"""
Entorno de bloques y ruteo con A* en 3D.
Todo va en mm. Los bloques se guardan en data/entorno/entorno.json.

Funciones: agregar_bloque, listar_bloques, eliminar_bloque,
limpiar_entorno, calculo_ruta, configurar_ruta_entorno
"""

import heapq
import json
import math
import os
import re
import tempfile
import unicodedata

__all__ = [
    "agregar_bloque",
    "listar_bloques",
    "eliminar_bloque",
    "limpiar_entorno",
    "calculo_ruta",
    "configurar_ruta_entorno",
    "distancia_seguridad_mm",
]

RUTA_ENTORNO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "entorno", "entorno.json"
)

TIPOS_VALIDOS = ("termico", "electrico", "neutro")

# Cuantos mm hay que alejarse por cada punto de valor.
# Por ahora es 1 a 1 (100 -> 10 cm), falta la tabla de la norma.
DISTANCIA_MM_POR_UNIDAD = {
    "termico": 1.0,
    "electrico": 1.0,
    "neutro": 0.0,
}


def distancia_seguridad_mm(tipo, valor):
    return max(0.0, float(valor)) * DISTANCIA_MM_POR_UNIDAD.get(tipo, 0.0)


# ---- guardar y cargar el json ----

def configurar_ruta_entorno(ruta):
    global RUTA_ENTORNO
    RUTA_ENTORNO = os.path.abspath(ruta)
    return RUTA_ENTORNO


def _cargar():
    try:
        with open(RUTA_ENTORNO, "r", encoding="utf-8") as f:
            datos = json.load(f)
    except FileNotFoundError:
        return []
    except (ValueError, OSError):
        # si el json esta roto lo aparto y empiezo vacio
        try:
            os.replace(RUTA_ENTORNO, RUTA_ENTORNO + ".corrupto")
        except OSError:
            pass
        return []

    crudos = datos.get("bloques", []) if isinstance(datos, dict) else []
    bloques = []
    for crudo in crudos if isinstance(crudos, list) else []:
        try:
            bloques.append(_normalizar_bloque(crudo))
        except (ValueError, TypeError, KeyError):
            continue  # bloque mal formado, lo salto
    return bloques


def _guardar(bloques):
    carpeta = os.path.dirname(RUTA_ENTORNO)
    os.makedirs(carpeta, exist_ok=True)
    contenido = {"version": 1, "bloques": bloques}

    # escribo en un temporal y luego lo reemplazo
    fd, tmp = tempfile.mkstemp(dir=carpeta, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(contenido, f, ensure_ascii=False, indent=2)
        try:
            os.replace(tmp, RUTA_ENTORNO)
        except OSError:
            # OneDrive a veces bloquea el reemplazo, escribo directo
            with open(RUTA_ENTORNO, "w", encoding="utf-8") as f:
                json.dump(contenido, f, ensure_ascii=False, indent=2)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


# ---- validacion ----

def _a_xyz(valor, campo):
    # acepta {"x","y","z"} o [x, y, z] y regresa [x, y, z] en floats
    if isinstance(valor, dict):
        try:
            crudo = [valor["x"], valor["y"], valor["z"]]
        except KeyError:
            raise ValueError("'%s' debe tener las llaves x, y, z" % campo)
    elif isinstance(valor, (list, tuple)) and len(valor) == 3:
        crudo = list(valor)
    else:
        raise ValueError("'%s' debe ser {'x','y','z'} o una lista [x, y, z]" % campo)
    try:
        xyz = [float(v) for v in crudo]
    except (TypeError, ValueError):
        raise ValueError("'%s' debe contener solo números" % campo)
    if any(math.isnan(v) or math.isinf(v) for v in xyz):
        raise ValueError("'%s' contiene valores no finitos" % campo)
    return xyz


def _quitar_acentos(texto):
    texto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def _normalizar_tipo(tipo):
    t = _quitar_acentos(str(tipo).strip().lower())
    if t not in TIPOS_VALIDOS:
        raise ValueError(
            "tipo inválido '%s'; usa: termico, electrico o neutro" % tipo
        )
    return t


def _normalizar_bloque(b):
    centro = _a_xyz(b["centro"], "centro")
    size = _a_xyz(b["size"], "size")
    if any(v <= 0 for v in size):
        raise ValueError("'size' debe ser mayor que 0 en x, y y z")
    tipo = _normalizar_tipo(b["tipo"])
    try:
        valor = float(b.get("valor", 0))
    except (TypeError, ValueError):
        raise ValueError("'valor' debe ser numérico")
    if math.isnan(valor) or math.isinf(valor) or valor < 0:
        raise ValueError("'valor' debe ser un número >= 0")
    if tipo == "neutro":
        valor = 0.0
    return {
        "id": str(b["id"]),
        "nombre": str(b["nombre"]),
        "centro": {"x": centro[0], "y": centro[1], "z": centro[2]},
        "size": {"x": size[0], "y": size[1], "z": size[2]},
        "tipo": tipo,
        "valor": valor,
    }


def _generar_id(nombre, usados):
    base = _quitar_acentos(nombre.lower())
    base = re.sub(r"[^a-z0-9]+", "_", base).strip("_") or "bloque"
    n = 1
    while "%s_%d" % (base, n) in usados:
        n += 1
    return "%s_%d" % (base, n)


# ---- funciones del entorno ----

def agregar_bloque(nombre, centro, size, tipo, valor=0, id=None):
    # regresa el bloque creado, o lanza ValueError si los datos no sirven
    nombre = str(nombre).strip()
    if not nombre:
        raise ValueError("'nombre' no puede estar vacío")

    bloques = _cargar()
    usados = {b["id"] for b in bloques}
    if id is None:
        id = _generar_id(nombre, usados)
    else:
        id = str(id).strip()
        if not id:
            raise ValueError("'id' no puede estar vacío")
        if id in usados:
            raise ValueError("ya existe un bloque con id '%s'" % id)

    bloque = _normalizar_bloque(
        {"id": id, "nombre": nombre, "centro": centro, "size": size,
         "tipo": tipo, "valor": valor}
    )
    bloques.append(bloque)
    _guardar(bloques)
    return bloque


def listar_bloques():
    return _cargar()


def eliminar_bloque(id):
    # True si lo encontro y lo borro
    bloques = _cargar()
    restantes = [b for b in bloques if b["id"] != id]
    if len(restantes) == len(bloques):
        return False
    _guardar(restantes)
    return True


def limpiar_entorno():
    # regresa cuantos bloques habia
    n = len(_cargar())
    _guardar([])
    return n


# ---- geometria ----

def _caja(b):
    # (x0, y0, z0, x1, y1, z1)
    c, s = b["centro"], b["size"]
    return (
        c["x"] - s["x"] / 2.0, c["y"] - s["y"] / 2.0, c["z"] - s["z"] / 2.0,
        c["x"] + s["x"] / 2.0, c["y"] + s["y"] / 2.0, c["z"] + s["z"] / 2.0,
    )


def _dist_a_caja(p, c):
    # distancia de un punto a la superficie de la caja (0 si esta adentro)
    dx = max(c[0] - p[0], 0.0, p[0] - c[3])
    dy = max(c[1] - p[1], 0.0, p[1] - c[4])
    dz = max(c[2] - p[2], 0.0, p[2] - c[5])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _dentro(p, c, margen):
    return (c[0] - margen <= p[0] <= c[3] + margen
            and c[1] - margen <= p[1] <= c[4] + margen
            and c[2] - margen <= p[2] <= c[5] + margen)


def _dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


# ---- A* ----

# los 26 vecinos de una celda con su longitud
_VECINOS = [
    ((dx, dy, dz), math.sqrt(dx * dx + dy * dy + dz * dz))
    for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
]


def _buscar(meta, res, limites, evaluar, max_nodos):
    # el inicio es la celda (0, 0, 0)
    # regresa (celdas, motivo), motivo: "ok", "sin_ruta" o "limite"
    imin, imax, jmin, jmax, kmin, kmax = limites
    inicio = (0, 0, 0)

    def h(c):
        return res * math.sqrt((c[0] - meta[0]) ** 2 + (c[1] - meta[1]) ** 2
                               + (c[2] - meta[2]) ** 2)

    g = {inicio: 0.0}
    padre = {}
    cerrado = set()
    abierto = [(h(inicio), 0, inicio)]
    desempate = 1
    expandidos = 0

    while abierto:
        _, _, actual = heapq.heappop(abierto)
        if actual in cerrado:
            continue
        if actual == meta:
            camino = [actual]
            while actual in padre:
                actual = padre[actual]
                camino.append(actual)
            camino.reverse()
            return camino, "ok"
        cerrado.add(actual)
        expandidos += 1
        if expandidos > max_nodos:
            return None, "limite"

        ga = g[actual]
        for (dx, dy, dz), largo in _VECINOS:
            v = (actual[0] + dx, actual[1] + dy, actual[2] + dz)
            if v in cerrado:
                continue
            if not (imin <= v[0] <= imax and jmin <= v[1] <= jmax
                    and kmin <= v[2] <= kmax):
                continue
            bloqueada, mult = evaluar(v)
            if bloqueada:
                continue
            ng = ga + largo * res * mult
            if ng < g.get(v, float("inf")):
                g[v] = ng
                padre[v] = actual
                heapq.heappush(abierto, (ng + h(v), desempate, v))
                desempate += 1
    return None, "sin_ruta"


def _quitar_colineales(celdas):
    # deja solo los puntos donde la ruta cambia de direccion
    if len(celdas) <= 2:
        return celdas
    resultado = [celdas[0]]
    for i in range(1, len(celdas) - 1):
        d1 = (celdas[i][0] - celdas[i - 1][0], celdas[i][1] - celdas[i - 1][1],
              celdas[i][2] - celdas[i - 1][2])
        d2 = (celdas[i + 1][0] - celdas[i][0], celdas[i + 1][1] - celdas[i][1],
              celdas[i + 1][2] - celdas[i][2])
        if d1 != d2:
            resultado.append(celdas[i])
    resultado.append(celdas[-1])
    return resultado


# ---- calculo_ruta ----

_CONFIG_DEFECTO = {
    "resolucion_mm": None,         # tamaño de celda, None = automatico
    "radio_ducto_mm": 0.0,         # los bloques se agrandan este radio
    "peso_proximidad": 5.0,        # que tanto cuesta pasar cerca de lo termico/electrico
    "margen_mm": 300.0,            # espacio extra alrededor del entorno
    "max_nodos": 400000,           # tope de nodos para que no se cicle
    "distancia_cercania_mm": 100.0,  # para llenar bloques_cercanos
    "lambda_campo": 2.0,           # cuanto pesa campo_costo (solo si se pasa uno): costo = 1 + ... + lambda * C
}

# tope de celdas para evaluar campo_costo de una sola vez (cada celda es un punto que se manda a la funcion)
MAX_CELDAS_CAMPO = 1500000


def _resultado(estado, ruta=None, longitud=0.0, cercanos=None, mensaje=""):
    return {
        "ruta": ruta if ruta is not None else [],
        "longitud_mm": float(longitud),
        "estado": estado,
        "bloques_cercanos": cercanos if cercanos is not None else [],
        "mensaje": mensaje,
    }


def _leer_config(costo_ruta):
    cfg = dict(_CONFIG_DEFECTO)
    if costo_ruta is not None:
        if not isinstance(costo_ruta, dict):
            raise ValueError("'costo_ruta' debe ser un dict o None")
        for k, v in costo_ruta.items():
            if k not in cfg:
                raise ValueError(
                    "opción desconocida en costo_ruta: '%s' (válidas: %s)"
                    % (k, ", ".join(sorted(cfg)))
                )
            cfg[k] = v
    for k, v in cfg.items():
        if v is None and k == "resolucion_mm":
            continue
        try:
            cfg[k] = float(v)
        except (TypeError, ValueError):
            raise ValueError("costo_ruta['%s'] debe ser numérico" % k)
        if math.isnan(cfg[k]) or math.isinf(cfg[k]) or cfg[k] < 0:
            raise ValueError("costo_ruta['%s'] debe ser un número >= 0" % k)
    if cfg["resolucion_mm"] is not None and cfg["resolucion_mm"] <= 0:
        raise ValueError("costo_ruta['resolucion_mm'] debe ser > 0")
    cfg["max_nodos"] = int(cfg["max_nodos"])
    return cfg


def calculo_ruta(inicio, fin, costo_ruta=None, campo_costo=None):
    """
    inicio y fin: [x, y, z] o {"x":..,"y":..,"z":..} en mm.
    costo_ruta: dict opcional con las opciones de _CONFIG_DEFECTO.
    campo_costo: opcional. Funcion que recibe una LISTA de puntos [[x, y, z], ...] (mm) y regresa una lista
                 de costos en [0, 1], uno por punto (por ejemplo la red C_NN). Se llama UNA vez con todas
                 las celdas de la malla. El costo de entrar a una celda queda 1 + proximidad + lambda_campo * C,
                 asi que sigue siendo >= 1 y la heuristica (distancia en linea recta) sigue siendo valida.

    Regresa un dict con ruta, longitud_mm, estado ("ok", "sin_ruta" o "error"),
    bloques_cercanos y mensaje. No lanza excepciones.
    """
    try:
        return _calculo_ruta(inicio, fin, costo_ruta, campo_costo)
    except ValueError as e:
        return _resultado("error", mensaje=str(e))
    except Exception as e:
        return _resultado("error", mensaje="%s: %s" % (type(e).__name__, e))


def _calculo_ruta(inicio, fin, costo_ruta, campo_costo=None):
    cfg = _leer_config(costo_ruta)
    p0 = _a_xyz(inicio, "inicio")
    p1 = _a_xyz(fin, "fin")
    radio = cfg["radio_ducto_mm"]

    bloques = _cargar()
    cajas = []
    for b in bloques:
        cajas.append({
            "nombre": b["nombre"],
            "caja": _caja(b),
            "dseg": distancia_seguridad_mm(b["tipo"], b["valor"]),
        })

    # inicio y fin no pueden estar dentro de un bloque
    for etiqueta, p in (("inicio", p0), ("fin", p1)):
        for c in cajas:
            if _dentro(p, c["caja"], radio):
                return _resultado(
                    "error",
                    mensaje="El punto de %s está dentro (o demasiado cerca) del bloque '%s'"
                            % (etiqueta, c["nombre"]),
                )

    if _dist(p0, p1) < 1e-9:
        return _resultado("ok", [_redondear(p0)], 0.0, [], "")

    # sin bloques es linea recta
    if not cajas:
        return _resultado("ok", [_redondear(p0), _redondear(p1)], _dist(p0, p1), [], "")

    # zona de busqueda: todos los bloques + inicio + fin + margen
    lo = [min([c["caja"][i] for c in cajas] + [p0[i], p1[i]]) - cfg["margen_mm"]
          for i in range(3)]
    hi = [max([c["caja"][i + 3] for c in cajas] + [p0[i], p1[i]]) + cfg["margen_mm"]
          for i in range(3)]
    extension = max(hi[i] - lo[i] for i in range(3))

    res = cfg["resolucion_mm"]
    if res is None:
        res = max(extension / 80.0, 5.0)

    meta = tuple(int(round((p1[i] - p0[i]) / res)) for i in range(3))
    limites = []
    for i in range(3):
        cmin = min(int(math.ceil((lo[i] - p0[i]) / res)), meta[i])
        cmax = max(int(math.floor((hi[i] - p0[i]) / res)), meta[i])
        limites += [cmin, cmax]
    limites = tuple(limites)

    # una celda se bloquea si su cubo toca un bloque (por eso se suma res/2)
    inflado = radio + res / 2.0
    peso = cfg["peso_proximidad"]
    cache = {}

    # campo de costo externo: se evalua una sola vez para toda la malla (un lote), no celda por celda
    campo = None
    lam = cfg["lambda_campo"]
    if campo_costo is not None and lam > 0:
        i0, i1, j0, j1, k0, k1 = limites
        total = (i1 - i0 + 1) * (j1 - j0 + 1) * (k1 - k0 + 1)
        if total > MAX_CELDAS_CAMPO:
            raise ValueError(
                "La malla tiene %d celdas, demasiadas para campo_costo (maximo %d). "
                "Aumenta 'resolucion_mm'." % (total, MAX_CELDAS_CAMPO)
            )
        indices = [(i, j, k) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)
                   for k in range(k0, k1 + 1)]
        puntos_malla = [[p0[0] + i * res, p0[1] + j * res, p0[2] + k * res] for (i, j, k) in indices]
        valores = campo_costo(puntos_malla)
        if len(valores) != len(indices):
            raise ValueError("campo_costo devolvio %d valores para %d puntos" % (len(valores), len(indices)))
        campo = {c: min(1.0, max(0.0, float(v))) for c, v in zip(indices, valores)}

    def evaluar(celda):
        # regresa (bloqueada, multiplicador de costo)
        r = cache.get(celda)
        if r is not None:
            return r
        p = (p0[0] + celda[0] * res, p0[1] + celda[1] * res, p0[2] + celda[2] * res)
        es_extremo = celda == (0, 0, 0) or celda == meta
        bloqueada = False
        pen = 0.0
        for c in cajas:
            if not es_extremo and _dentro(p, c["caja"], inflado):
                bloqueada = True
                break
            if c["dseg"] > 0 and peso > 0:
                holgura = _dist_a_caja(p, c["caja"]) - radio
                if holgura < c["dseg"]:
                    f = (c["dseg"] - max(holgura, 0.0)) / c["dseg"]
                    pen += f * f
        extra = lam * campo.get(celda, 0.0) if campo is not None else 0.0
        r = (bloqueada, 1.0 + peso * pen + extra)
        cache[celda] = r
        return r

    celdas, motivo = _buscar(meta, res, limites, evaluar, int(cfg["max_nodos"]))
    if celdas is None:
        if motivo == "limite":
            return _resultado(
                "error",
                mensaje="Se alcanzó el límite de nodos (max_nodos). Aumenta "
                        "'resolucion_mm' (celdas más grandes) o 'max_nodos'.",
            )
        return _resultado(
            "sin_ruta",
            mensaje="No existe un camino libre entre inicio y fin con este radio de "
                    "ducto y esta resolución.",
        )

    def a_punto(celda):
        return [p0[0] + celda[0] * res, p0[1] + celda[1] * res, p0[2] + celda[2] * res]

    # puntos de todas las celdas, para ver que bloques quedaron cerca
    densos = [a_punto(c) for c in celdas]
    densos[0], densos[-1] = list(p0), list(p1)

    # puntos finales, solo donde cambia la direccion
    puntos = [a_punto(c) for c in _quitar_colineales(celdas)]
    puntos[0], puntos[-1] = list(p0), list(p1)
    puntos = [_redondear(p) for p in puntos]
    longitud = sum(_dist(puntos[i], puntos[i + 1]) for i in range(len(puntos) - 1))

    cercanos = []
    for c in cajas:
        dmin = min(_dist_a_caja(p, c["caja"]) for p in densos) - radio
        # se suma res porque la malla no es exacta
        if dmin <= max(c["dseg"], cfg["distancia_cercania_mm"]) + res:
            cercanos.append((dmin, c["nombre"]))
    cercanos.sort()
    nombres = []
    for _, nombre in cercanos:
        if nombre not in nombres:
            nombres.append(nombre)

    return _resultado("ok", puntos, round(longitud, 3), nombres, "")


def _redondear(p):
    return [round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3)]
