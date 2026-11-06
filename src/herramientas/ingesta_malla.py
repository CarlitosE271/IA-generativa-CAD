"""
Ingesta de mallas del entorno: STL/STEP por componente + manifiesto -> bloques (cajas) para el A*.

Es la parte "Ingesta 3D" del plan. Son funciones puras (solo numpy; CadQuery se importa solo si hay un STEP),
asi que se pueden probar sin LangChain, sin trimesh y sin Streamlit.

CARPETA DE UNA ESCENA (data/escenas/<nombre>/):
    escena.json     <- manifiesto (lo escribe Maikol o quien arme la geometria)
    motor.stl ...   <- un archivo por componente (.stl o .step/.stp)

MANIFIESTO (escena.json):
    {
      "version": 1,
      "nombre": "escena_1",
      "unidades": "mm",                       # solo para STL ("mm", "cm", "m", "in"); un STEP ya trae sus unidades
      "componentes": [
        {"id": "motor", "nombre": "Motor", "archivo": "motor.stl", "tipo": "termico", "valor": 100}
      ],
      "puertos": {                            # opcional: extremos del ducto (en las MISMAS coordenadas de las mallas)
        "inicio": {"punto": [x, y, z], "ancho_mm": 40, "alto_mm": 25},
        "fin":    {"punto": [x, y, z], "ancho_mm": 40, "alto_mm": 25}
      }
    }
    tipo: termico | electrico | neutro.  valor: distancia de seguridad en mm (igual que en entorno.json; neutro = 0).

COMO SE USA LA MALLA (decision del proyecto: "mallas -> cajas por voxelizacion"):
    1. Cada malla se voxeliza: se marcan los voxeles que tocan la superficie y se rellena el interior.
    2. Los voxeles se funden en cajas (una caja es un bloque del entorno.json de Fer, con el tipo y valor del componente).
    3. Cada caja se recorta a la caja envolvente de su malla.

LIMITES (hay que decirlos con honestidad):
    - Es una aproximacion CONSERVADORA: las cajas cubren la pieza y sobran hasta ~1 voxel (paso_mm) por lado.
      El informe da la sobreestimacion (volumen de cajas / volumen de malla) para cuantificarlo.
    - Una pieza mas delgada que el paso no desaparece: ocupa al menos un voxel.
    - Una malla ABIERTA (con huecos) no se puede rellenar: se usa su caja envolvente y se avisa.
    - Un STL no trae unidades: hay que declararlas en el manifiesto; si no, se asumen mm (y se avisa si la escena
      se ve absurda: < 50 mm o > 50 m).
    - El A* de Fer recorre todas las cajas en cada celda: por eso se limita el total de cajas (max_cajas).
"""
from __future__ import annotations

import json
import math
import re
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np

UNIDADES_A_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4}
TIPOS_VALIDOS = ("termico", "electrico", "neutro")
EXTENSIONES_STL = (".stl",)
EXTENSIONES_STEP = (".step", ".stp")

MAX_VOXELES = 6_000_000  # tope de la rejilla de UN componente (memoria y tiempo del relleno)
EXTENSION_MIN_MM = 50.0  # una escena mas chica que esto probablemente esta en metros o pulgadas
EXTENSION_MAX_MM = 50_000.0  # una escena mas grande que esto (50 m) probablemente esta en micras o mm mal escalados


class IngestaError(ValueError):
    """Problema con una malla o con el manifiesto (mensaje pensado para leerse tal cual)."""


# ----------------------------------------------------------------------------------------------------
# Lectura / escritura de mallas
# ----------------------------------------------------------------------------------------------------
def leer_stl(ruta) -> np.ndarray:
    """STL binario o ASCII -> arreglo (n, 3, 3): n triangulos x 3 vertices x (x, y, z), en las unidades del archivo."""
    ruta = Path(ruta)
    datos = ruta.read_bytes()
    if len(datos) >= 84:
        n = struct.unpack("<I", datos[80:84])[0]
        if n > 0 and 84 + 50 * n == len(datos):  # el tamano exacto delata al binario
            dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
            tri = np.frombuffer(datos, dtype=dt, count=n, offset=84)["v"].astype(np.float64)
            return _validar_triangulos(tri, ruta.name)
    texto = datos.decode("utf-8", errors="ignore")
    vertices = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", texto)
    if not vertices or len(vertices) % 3:
        raise IngestaError(f"'{ruta.name}' no se pudo leer como STL (ni binario ni ASCII valido).")
    try:
        tri = np.array(vertices, dtype=np.float64).reshape(-1, 3, 3)
    except ValueError:
        raise IngestaError(f"'{ruta.name}' tiene vertices que no son numeros.")
    return _validar_triangulos(tri, ruta.name)


def _validar_triangulos(tri: np.ndarray, nombre: str) -> np.ndarray:
    if tri.size == 0:
        raise IngestaError(f"'{nombre}' no tiene triangulos.")
    if not np.all(np.isfinite(tri)):
        raise IngestaError(f"'{nombre}' tiene coordenadas no finitas (NaN o infinito).")
    return tri


def escribir_stl(ruta, tri: np.ndarray) -> None:
    """Escribe un STL binario (sirve para escenas de ejemplo y pruebas)."""
    tri = np.asarray(tri, dtype=np.float64).reshape(-1, 3, 3)
    dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    arr = np.zeros(len(tri), dtype=dt)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    largo = np.linalg.norm(normal, axis=1, keepdims=True)
    arr["n"] = np.divide(normal, largo, out=np.zeros_like(normal), where=largo > 0)
    arr["v"] = tri
    with open(ruta, "wb") as f:
        f.write(b"ingesta_malla".ljust(80, b" "))
        f.write(struct.pack("<I", len(tri)))
        f.write(arr.tobytes())


def leer_step(ruta, tolerancia_mm: float = 0.5) -> np.ndarray:
    """STEP -> triangulos (n, 3, 3) en mm (OCCT normaliza a mm al importar). Necesita CadQuery."""
    import cadquery as cq  # import perezoso: es pesada y solo hace falta con STEP

    ruta = Path(ruta)
    shape = cq.importers.importStep(str(ruta)).val()
    if not hasattr(shape, "tessellate"):
        raise IngestaError(f"'{ruta.name}' no contiene un solido o superficie valido.")
    vertices, triangulos = shape.tessellate(tolerancia_mm)
    if not triangulos:
        raise IngestaError(f"'{ruta.name}' no produjo triangulos al teselarlo.")
    v = np.array([[p.x, p.y, p.z] for p in vertices], dtype=np.float64)
    return _validar_triangulos(v[np.array(triangulos, dtype=int)], ruta.name)


def malla_caja(lo, hi) -> np.ndarray:
    """Malla cerrada de una caja (12 triangulos, normales hacia afuera)."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], dtype=np.float64)
    caras = [(0, 3, 2), (0, 2, 1), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (3, 7, 6), (3, 6, 2), (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5)]
    return v[np.array(caras)]


def malla_cilindro(centro, radio: float, alto: float, lados: int = 32) -> np.ndarray:
    """Malla cerrada de un cilindro con eje Z; 'centro' es el centro de la base (x, y, z_base)."""
    cx, cy, z0 = centro
    t = np.linspace(0.0, 2.0 * np.pi, lados, endpoint=False)
    inf = np.stack([cx + radio * np.cos(t), cy + radio * np.sin(t), np.full(lados, z0)], axis=1)
    sup = inf + np.array([0.0, 0.0, alto])
    c_inf, c_sup = np.array([cx, cy, z0]), np.array([cx, cy, z0 + alto])
    tri = []
    for i in range(lados):
        j = (i + 1) % lados
        tri += [(inf[i], inf[j], sup[j]), (inf[i], sup[j], sup[i]), (c_inf, inf[j], inf[i]), (c_sup, sup[i], sup[j])]
    return np.array(tri, dtype=np.float64)


# ----------------------------------------------------------------------------------------------------
# Propiedades de una malla
# ----------------------------------------------------------------------------------------------------
def info_malla(tri: np.ndarray) -> dict:
    """bbox, numero de triangulos, volumen (si se puede) y si la malla esta CERRADA (sin aristas de borde)."""
    puntos = tri.reshape(-1, 3)
    lo, hi = puntos.min(axis=0), puntos.max(axis=0)
    escala = float(np.max(hi - lo)) or 1.0
    # se sueldan los vertices repetidos (un STL guarda cada vertice una vez por triangulo)
    redondeado = np.round(puntos / (escala * 1e-7)).astype(np.int64)
    _, ids = np.unique(redondeado, axis=0, return_inverse=True)
    ids = ids.reshape(-1, 3)
    aristas = np.concatenate([ids[:, [0, 1]], ids[:, [1, 2]], ids[:, [2, 0]]])
    aristas = np.sort(aristas, axis=1)
    aristas = aristas[aristas[:, 0] != aristas[:, 1]]  # triangulos degenerados
    _, cuentas = np.unique(aristas, axis=0, return_counts=True)
    # cerrada = ninguna arista de borde. Cuenta PAR (2, o 4 si dos cuerpos comparten una arista) y no solo 2.
    cerrada = bool(len(cuentas) > 0 and np.all(cuentas % 2 == 0))
    volumen = abs(float(np.sum(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))) / 6.0))
    return {
        "n_triangulos": int(len(tri)),
        "bbox_min": [float(v) for v in lo],
        "bbox_max": [float(v) for v in hi],
        "cerrada": cerrada,
        "volumen_mm3": volumen if cerrada and volumen > 0 else None,
    }


# ----------------------------------------------------------------------------------------------------
# Voxelizacion y fusion en cajas
# ----------------------------------------------------------------------------------------------------
def voxelizar_superficie(tri: np.ndarray, paso: float, origen: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """Rejilla booleana (dims) con True en los voxeles que tocan la superficie.
    Se muestrea cada triangulo con puntos separados <= paso/2 (incluye vertices y aristas).
    El muestreo es anisotropico: cada triangulo se rota para que 'a' sea el vertice comun de sus dos aristas
    CORTAS, asi un triangulo largo y delgado (tipico de las tapas de un cilindro) no desperdicia miles de puntos."""
    rejilla = np.zeros(tuple(int(d) for d in dims), dtype=bool)
    l_ab = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
    l_bc = np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1)
    l_ca = np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)
    # vertice opuesto a la arista mas larga: 0 si la mas larga es bc, 1 si es ca, 2 si es ab
    opuesto = np.argmax(np.stack([l_bc, l_ca, l_ab], axis=1), axis=1)
    filas = np.arange(len(tri))
    a, b, c = (tri[filas, (opuesto + d) % 3] for d in (0, 1, 2))
    h = paso / 2.0
    ku = np.maximum(1, np.ceil(np.linalg.norm(b - a, axis=1) / h)).astype(np.int64)
    kv = np.maximum(1, np.ceil(np.linalg.norm(c - a, axis=1) / h)).astype(np.int64)
    clave = ku * 1_000_003 + kv
    tope = np.asarray(dims, dtype=np.int64) - 1
    for k in np.unique(clave):
        idx = np.flatnonzero(clave == k)
        nu, nv = int(ku[idx[0]]), int(kv[idx[0]])
        ii, jj = np.meshgrid(np.arange(nu + 1), np.arange(nv + 1), indexing="ij")
        ok = (ii / nu + jj / nv) <= 1.0 + 1e-9
        u, v = ii[ok] / nu, jj[ok] / nv
        w = 1.0 - u - v
        por_lote = max(1, int(2_000_000 // len(u)))
        for s in range(0, len(idx), por_lote):
            t = idx[s:s + por_lote]
            p = (w[None, :, None] * a[t][:, None, :] + u[None, :, None] * b[t][:, None, :]
                 + v[None, :, None] * c[t][:, None, :]).reshape(-1, 3)
            ind = np.floor((p - origen) / paso).astype(np.int64)
            ind = np.clip(ind, 0, tope)
            rejilla[ind[:, 0], ind[:, 1], ind[:, 2]] = True
    return rejilla


def rellenar_interior(superficie: np.ndarray) -> np.ndarray:
    """Voxeles ocupados = superficie + todo lo que NO se alcanza desde afuera (inundacion con 6 vecinos).
    Si la superficie tiene huecos, el 'exterior' se cuela y no se rellena nada (se detecta aparte)."""
    libre = np.pad(~superficie, 1, constant_values=True)
    exterior = np.zeros_like(libre)
    exterior[0, :, :] = exterior[-1, :, :] = True
    exterior[:, 0, :] = exterior[:, -1, :] = True
    exterior[:, :, 0] = exterior[:, :, -1] = True
    exterior &= libre
    while True:
        nuevo = exterior.copy()
        nuevo[1:, :, :] |= exterior[:-1, :, :]
        nuevo[:-1, :, :] |= exterior[1:, :, :]
        nuevo[:, 1:, :] |= exterior[:, :-1, :]
        nuevo[:, :-1, :] |= exterior[:, 1:, :]
        nuevo[:, :, 1:] |= exterior[:, :, :-1]
        nuevo[:, :, :-1] |= exterior[:, :, 1:]
        nuevo &= libre
        if np.array_equal(nuevo, exterior):
            break
        exterior = nuevo
    return ~exterior[1:-1, 1:-1, 1:-1]


def _consecutivos(valores):
    """[1,2,3,7,8] -> [(1,4), (7,9)] (rangos [a, b) de enteros consecutivos)."""
    valores = sorted(valores)
    rangos, ini, prev = [], valores[0], valores[0]
    for v in valores[1:]:
        if v == prev + 1:
            prev = v
            continue
        rangos.append((ini, prev + 1))
        ini = prev = v
    rangos.append((ini, prev + 1))
    return rangos


def voxeles_a_cajas(ocupado: np.ndarray) -> list[tuple]:
    """Funde los voxeles ocupados en cajas (i0, i1, j0, j1, k0, k1) con indices [inicio, fin).
    Primero tramos en Z, luego se unen tramos iguales vecinos en Y, luego en X (no es el minimo, pero es rapido)."""
    if not ocupado.any():
        return []
    relleno = np.pad(ocupado.astype(np.int8), ((0, 0), (0, 0), (1, 1)))
    d = np.diff(relleno, axis=2)
    inicios = np.argwhere(d == 1)  # (i, j, k0) en orden lexicografico
    fines = np.argwhere(d == -1)   # (i, j, k1) en el mismo orden: se corresponden uno a uno
    por_columna = defaultdict(list)  # (i, k0, k1) -> [j]
    for (i, j, k0), (_, _, k1) in zip(inicios, fines):
        por_columna[(int(i), int(k0), int(k1))].append(int(j))
    por_y = defaultdict(list)  # (j0, j1, k0, k1) -> [i]
    for (i, k0, k1), js in por_columna.items():
        for j0, j1 in _consecutivos(js):
            por_y[(j0, j1, k0, k1)].append(i)
    cajas = []
    for (j0, j1, k0, k1), is_ in por_y.items():
        for i0, i1 in _consecutivos(is_):
            cajas.append((i0, i1, j0, j1, k0, k1))
    return cajas


def malla_a_cajas(tri: np.ndarray, paso_mm: float, rellenar: bool = True) -> tuple[list[tuple], dict]:
    """Una malla -> cajas en coordenadas del mundo (x0, y0, z0, x1, y1, z1), recortadas a su caja envolvente.
    Devuelve (cajas, meta) con meta = {"metodo": "voxel" | "aabb", "advertencias": [...], "voxeles": n}."""
    puntos = tri.reshape(-1, 3)
    lo, hi = puntos.min(axis=0), puntos.max(axis=0)
    aabb = [(*[float(v) for v in lo], *[float(v) for v in hi])]
    dims = np.ceil((hi - lo) / paso_mm).astype(np.int64) + 1
    if int(np.prod(dims)) > MAX_VOXELES:
        raise IngestaError(
            f"Con paso {paso_mm:.3g} mm la rejilla de un componente tendria {int(np.prod(dims))} voxeles "
            f"(maximo {MAX_VOXELES}). Aumenta el paso."
        )
    superficie = voxelizar_superficie(tri, paso_mm, lo, dims)
    ocupado = rellenar_interior(superficie) if rellenar else superficie
    advertencias = []
    if rellenar and ocupado.sum() == superficie.sum() and not info_malla(tri)["cerrada"]:
        # nada se relleno y la malla no esta cerrada: la cascara hueca no sirve como obstaculo
        return aabb, {"metodo": "aabb", "voxeles": int(superficie.sum()),
                      "advertencias": ["malla ABIERTA (con huecos): no se pudo rellenar; se uso su caja envolvente"]}
    cajas = []
    for i0, i1, j0, j1, k0, k1 in voxeles_a_cajas(ocupado):
        x0, y0, z0 = lo + paso_mm * np.array([i0, j0, k0])
        x1, y1, z1 = lo + paso_mm * np.array([i1, j1, k1])
        cajas.append((float(max(x0, lo[0])), float(max(y0, lo[1])), float(max(z0, lo[2])),
                      float(min(x1, hi[0])), float(min(y1, hi[1])), float(min(z1, hi[2]))))
    return cajas, {"metodo": "voxel", "voxeles": int(ocupado.sum()), "advertencias": advertencias}


# ----------------------------------------------------------------------------------------------------
# Manifiesto y escena completa
# ----------------------------------------------------------------------------------------------------
def _nombre_archivo_seguro(nombre: str, carpeta: Path) -> Path:
    if not nombre or any(c in nombre for c in "/\\:") or Path(nombre).name != nombre:
        raise IngestaError(f"Archivo '{nombre}': solo se acepta el NOMBRE del archivo, sin carpetas ni rutas.")
    if not nombre.lower().endswith(EXTENSIONES_STL + EXTENSIONES_STEP):
        raise IngestaError(f"Archivo '{nombre}': debe terminar en .stl, .step o .stp.")
    ruta = (carpeta / nombre).resolve()
    if carpeta.resolve() not in ruta.parents:
        raise IngestaError(f"Archivo '{nombre}': ruta fuera de la carpeta de la escena.")
    if not ruta.is_file():
        raise IngestaError(f"No existe el archivo '{nombre}' en {carpeta}.")
    return ruta


def _punto(valor, campo: str) -> list[float]:
    if isinstance(valor, dict):
        valor = [valor.get("x"), valor.get("y"), valor.get("z")]
    try:
        p = [float(v) for v in valor]
    except (TypeError, ValueError):
        raise IngestaError(f"'{campo}' debe ser [x, y, z] o {{x, y, z}} con numeros.")
    if len(p) != 3 or not all(math.isfinite(v) for v in p):
        raise IngestaError(f"'{campo}' debe tener tres numeros finitos.")
    return p


def cargar_manifiesto(carpeta) -> dict:
    """Lee y valida escena.json. Devuelve el manifiesto normalizado (componentes con ruta resuelta)."""
    carpeta = Path(carpeta)
    ruta = carpeta / "escena.json"
    try:
        crudo = json.loads(ruta.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise IngestaError(f"Falta {ruta}. Cada escena necesita su escena.json.")
    except json.JSONDecodeError as e:
        raise IngestaError(f"escena.json no es JSON valido: {e}")
    if not isinstance(crudo, dict) or not isinstance(crudo.get("componentes"), list) or not crudo["componentes"]:
        raise IngestaError("escena.json debe traer una lista 'componentes' con al menos un componente.")

    unidades = str(crudo.get("unidades", "mm")).strip().lower()
    if unidades not in UNIDADES_A_MM:
        raise IngestaError(f"unidades '{unidades}' no valida; usa: {', '.join(UNIDADES_A_MM)}.")

    componentes, ids = [], set()
    for c in crudo["componentes"]:
        if not isinstance(c, dict):
            raise IngestaError("Cada componente debe ser un objeto con id, nombre, archivo y tipo.")
        for campo in ("id", "nombre", "archivo", "tipo"):
            if not str(c.get(campo, "")).strip():
                raise IngestaError(f"Un componente no trae '{campo}'.")
        cid = re.sub(r"[^a-z0-9_]+", "_", str(c["id"]).strip().lower()).strip("_")
        if not cid:
            raise IngestaError(f"id '{c['id']}' no es valido.")
        if cid in ids:
            raise IngestaError(f"id repetido: '{cid}'.")
        ids.add(cid)
        tipo = str(c["tipo"]).strip().lower()
        tipo = tipo.replace("é", "e")  # térmico / eléctrico con acento tambien valen
        if tipo not in TIPOS_VALIDOS:
            raise IngestaError(f"Componente '{cid}': tipo '{c['tipo']}' invalido; usa termico, electrico o neutro.")
        try:
            valor = float(c.get("valor", 0))
        except (TypeError, ValueError):
            raise IngestaError(f"Componente '{cid}': 'valor' debe ser numerico.")
        if not math.isfinite(valor) or valor < 0:
            raise IngestaError(f"Componente '{cid}': 'valor' debe ser un numero >= 0.")
        componentes.append({
            "id": cid, "nombre": str(c["nombre"]).strip(), "archivo": _nombre_archivo_seguro(str(c["archivo"]), carpeta),
            "tipo": tipo, "valor": 0.0 if tipo == "neutro" else valor,
        })

    puertos = {}
    for lado in ("inicio", "fin"):
        p = (crudo.get("puertos") or {}).get(lado)
        if p is None:
            continue
        if not isinstance(p, dict) or "punto" not in p:
            raise IngestaError(f"puertos.{lado} debe ser un objeto con 'punto'.")
        puerto = dict(p)
        puerto["punto"] = _punto(p["punto"], f"puertos.{lado}.punto")
        puertos[lado] = puerto

    return {"nombre": str(crudo.get("nombre") or carpeta.name), "unidades": unidades,
            "componentes": componentes, "puertos": puertos}


def _distancia_punto_caja(p, c) -> float:
    d = np.maximum.reduce([np.array(c[:3]) - p, np.zeros(3), p - np.array(c[3:])])
    return float(np.linalg.norm(d))


def escena_a_bloques(carpeta, paso_mm: float | None = None, max_cajas: int = 150) -> tuple[list[dict], dict]:
    """Carpeta de escena -> (bloques en formato entorno.json de Fer, informe).

    paso_mm: tamano del voxel. None = automatico (extension de la escena / 80, como la celda del A*, minimo 2 mm).
    max_cajas: tope del TOTAL de cajas; si se pasa, el paso crece (x1.25) hasta que quepa.
    """
    carpeta = Path(carpeta)
    man = cargar_manifiesto(carpeta)
    factor = UNIDADES_A_MM[man["unidades"]]
    advertencias = []

    mallas = []
    for c in man["componentes"]:
        es_step = c["archivo"].suffix.lower() in EXTENSIONES_STEP
        tri = leer_step(c["archivo"]) if es_step else leer_stl(c["archivo"]) * factor
        mallas.append((c, tri, info_malla(tri)))

    todo_lo = np.min([m[2]["bbox_min"] for m in mallas], axis=0)
    todo_hi = np.max([m[2]["bbox_max"] for m in mallas], axis=0)
    extension = float(np.max(todo_hi - todo_lo))
    if extension < EXTENSION_MIN_MM or extension > EXTENSION_MAX_MM:
        advertencias.append(
            f"La escena mide {extension:.1f} mm en su dimension mayor: revisa 'unidades' en escena.json "
            f"(un STL no trae unidades)."
        )

    paso = float(paso_mm) if paso_mm else max(extension / 80.0, 2.0)
    paso_inicial = paso
    for _ in range(25):
        por_componente = [malla_a_cajas(tri, paso) for _, tri, _ in mallas]
        total = sum(len(cajas) for cajas, _ in por_componente)
        if total <= max_cajas:
            break
        paso *= 1.25
    else:
        advertencias.append(f"No se logro bajar de {max_cajas} cajas ({total}); aumenta max_cajas o el paso.")

    bloques, componentes_info = [], []
    for (c, tri, info), (cajas, meta) in zip(mallas, por_componente):
        vol_cajas = sum((x1 - x0) * (y1 - y0) * (z1 - z0) for x0, y0, z0, x1, y1, z1 in cajas)
        vol_malla = info["volumen_mm3"]
        for n, (x0, y0, z0, x1, y1, z1) in enumerate(cajas, start=1):
            bloques.append({
                "id": f"{c['id']}_{n:03d}", "nombre": c["nombre"],
                "centro": {"x": (x0 + x1) / 2, "y": (y0 + y1) / 2, "z": (z0 + z1) / 2},
                "size": {"x": x1 - x0, "y": y1 - y0, "z": z1 - z0},
                "tipo": c["tipo"], "valor": c["valor"],
            })
        componentes_info.append({
            "id": c["id"], "nombre": c["nombre"], "archivo": c["archivo"].name, "tipo": c["tipo"], "valor": c["valor"],
            "n_triangulos": info["n_triangulos"], "malla_cerrada": info["cerrada"],
            "bbox_min_mm": [round(v, 2) for v in info["bbox_min"]], "bbox_max_mm": [round(v, 2) for v in info["bbox_max"]],
            "metodo": meta["metodo"], "n_cajas": len(cajas),
            "volumen_malla_mm3": round(vol_malla, 1) if vol_malla else None,
            "volumen_cajas_mm3": round(vol_cajas, 1),
            "sobreestimacion": round(vol_cajas / vol_malla, 3) if vol_malla else None,
            "advertencias": meta["advertencias"],
        })

    puertos = man["puertos"]
    for lado, p in puertos.items():
        pt = np.array(p["punto"])
        for b in bloques:
            s, ce = b["size"], b["centro"]
            caja = (ce["x"] - s["x"] / 2, ce["y"] - s["y"] / 2, ce["z"] - s["z"] / 2,
                    ce["x"] + s["x"] / 2, ce["y"] + s["y"] / 2, ce["z"] + s["z"] / 2)
            if _distancia_punto_caja(pt, caja) == 0.0:
                advertencias.append(f"El puerto '{lado}' {p['punto']} cae DENTRO de '{b['nombre']}' (caja {b['id']}): "
                                    "el A* no podra salir de ahi. Revisa coordenadas o unidades.")
                break

    informe = {
        "escena": man["nombre"], "unidades_declaradas": man["unidades"], "extension_mm": round(extension, 1),
        "paso_mm": round(paso, 2), "paso_inicial_mm": round(paso_inicial, 2), "max_cajas": max_cajas,
        "n_cajas_total": len(bloques), "componentes": componentes_info, "puertos": puertos, "advertencias": advertencias,
    }
    return bloques, informe


def guardar_entorno(bloques: list[dict], ruta) -> Path:
    """Escribe bloques en el mismo formato que ruteo_astar (version 1)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps({"version": 1, "bloques": bloques}, ensure_ascii=False, indent=2), encoding="utf-8")
    return ruta
