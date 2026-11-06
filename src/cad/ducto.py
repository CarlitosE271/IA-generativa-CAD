"""
Generacion de CAD del ducto (componente 5 del diagrama de arquitectura): convierte la ruta que
calcula el A* (una lista de puntos [x, y, z]) en un ducto rectangular HUECO y lo exporta a
STEP/STL.

Flujo:
  ruta del A* (esquinas del grid)
    -> suavizar_ruta        corta las esquinas (Chaikin) y remuestrea a paso uniforme
    -> validar_*            compara contra los parametros aprobados (R_min, area, relacion de aspecto)
    -> estaciones           un rectangulo por estacion, orientado con transporte paralelo
    -> loft (CadQuery)      sólido exterior, menos sólido interior = ducto hueco
    -> export               data/cad/<nombre>.step y .stl

Por que NO se usa sweep() de CadQuery: barrer un rectangulo por una curva 3D puede torcer el perfil
(el rectangulo gira sobre si mismo) o fallar. Aqui la orientacion de cada rectangulo se calcula
a mano con "marcos de transporte paralelo" (rotation-minimizing frames): el perfil nunca gira
innecesariamente. Despues se hace un loft entre los perfiles, que es justo el "loft entre perfiles
por estacion" de tu diagrama (y deja listo el camino para que la seccion cambie a lo largo de la
ruta mas adelante: rectangular -> circular).

Casi todo es matematica pura con numpy (sin CadQuery, sin LLM), igual que documentos.py y
geometria.py: por eso se puede probar sin CadQuery. Solo construir_ducto() y exportar() la usan.

Suposiciones que conviene tener claras (para defenderlas ante el panel):
  - r_min_mm se compara contra el radio de curvatura de la LINEA CENTRAL del ducto.
  - El suavizado corta esquinas: la curva resultante puede acercarse a un obstaculo mas que la
    ruta original del A*. Esta version NO revisa colisiones del ducto con el entorno.
  - Si la ruta del A* gira 90 grados en una sola celda del grid, el radio resultante puede ser
    chico y violar r_min_mm: no es un error de este modulo, es informacion util (el A* deberia
    imponer R_min en su costo, como dice tu plan).
  - El angulo de desmoldeo (draft angle) todavia no se valida: necesita definir una direccion de
    desmoldeo, que es una decision de manufactura que no hemos tomado.

Nota honesta: CadQuery no se pudo instalar en este entorno de trabajo, asi que construir_ducto()
se probo con un modulo falso (tests/test_ducto.py). Toda la parte numerica (suavizado, radios,
marcos, rectangulos) si se probo con numpy real. La primera corrida real con CadQuery es en tu
maquina: es probable que el loft pida algun ajuste, avisame el error.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

raiz = Path(__file__).resolve().parents[2]
ruta_cad = raiz / "data" / "cad"

_EPS = 1e-9


def asegurar_carpetas() -> None:
    ruta_cad.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Ruta: limpieza, suavizado y remuestreo
# --------------------------------------------------------------------------
def _como_puntos(ruta) -> np.ndarray:
    """Convierte la ruta a un arreglo (n, 3) y quita puntos consecutivos repetidos."""
    puntos = np.asarray(ruta, dtype=float)
    if puntos.ndim != 2 or puntos.shape[1] != 3:
        raise ValueError("La ruta debe ser una lista de puntos [x, y, z].")
    if len(puntos) > 1:
        distintos = np.linalg.norm(np.diff(puntos, axis=0), axis=1) > _EPS
        puntos = puntos[np.concatenate(([True], distintos))]
    if len(puntos) < 2:
        raise ValueError("La ruta necesita al menos dos puntos distintos.")
    return puntos


def _chaikin(puntos: np.ndarray, iteraciones: int) -> np.ndarray:
    """Corte de esquinas de Chaikin con extremos fijos: cada segmento P->Q se reemplaza por los
    puntos a 1/4 y 3/4. Converge a una B-spline cuadratica suave, y no necesita scipy."""
    for _ in range(iteraciones):
        p, q = puntos[:-1], puntos[1:]
        cuartos = 0.75 * p + 0.25 * q
        tres_cuartos = 0.25 * p + 0.75 * q
        intercalados = np.empty((2 * len(p), 3))
        intercalados[0::2] = cuartos
        intercalados[1::2] = tres_cuartos
        puntos = np.vstack((puntos[:1], intercalados, puntos[-1:]))
    return puntos


def _remuestrear(puntos: np.ndarray, paso_mm: float) -> np.ndarray:
    """Puntos a separacion (casi) uniforme medida a lo largo de la curva."""
    tramos = np.linalg.norm(np.diff(puntos, axis=0), axis=1)
    acumulado = np.concatenate(([0.0], np.cumsum(tramos)))
    largo = acumulado[-1]
    n = max(2, int(math.ceil(largo / paso_mm)) + 1)
    objetivo = np.linspace(0.0, largo, n)
    return np.column_stack([np.interp(objetivo, acumulado, puntos[:, k]) for k in range(3)])


def suavizar_ruta(ruta, iteraciones: int = 3, paso_mm: float = 5.0) -> np.ndarray:
    """Ruta del A* (esquinas del grid) -> curva suave remuestreada cada ~paso_mm.
    Los extremos (inicio y fin) se conservan exactos."""
    puntos = _como_puntos(ruta)
    if len(puntos) > 2:
        puntos = _chaikin(puntos, iteraciones)
    return _remuestrear(puntos, paso_mm)


def longitud_mm(puntos: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(puntos, axis=0), axis=1).sum())


# --------------------------------------------------------------------------
# Validacion contra los parametros aprobados
# --------------------------------------------------------------------------
def radios_de_curvatura(puntos: np.ndarray) -> np.ndarray:
    """Radio de curvatura (mm) en cada punto interior, con el circulo que pasa por tres puntos
    consecutivos: R = a*b*c / (2*|(p1-p0) x (p2-p1)|). Un tramo recto da infinito."""
    p0, p1, p2 = puntos[:-2], puntos[1:-1], puntos[2:]
    a = np.linalg.norm(p1 - p0, axis=1)
    b = np.linalg.norm(p2 - p1, axis=1)
    c = np.linalg.norm(p2 - p0, axis=1)
    cruz = np.linalg.norm(np.cross(p1 - p0, p2 - p1), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        radios = np.where(cruz > _EPS, a * b * c / (2.0 * cruz), np.inf)
    return radios


def validar_radio_minimo(puntos: np.ndarray, r_min_mm: float) -> dict:
    radios = radios_de_curvatura(puntos)
    if len(radios) == 0:
        return {"cumple": True, "radio_minimo_mm": None, "num_violaciones": 0, "posicion_peor_mm": None}
    i = int(np.argmin(radios))
    peor = float(radios[i])
    tramos = np.linalg.norm(np.diff(puntos, axis=0), axis=1)
    posicion = float(tramos[: i + 1].sum())  # distancia desde el inicio hasta el punto mas cerrado
    violaciones = int(np.sum(radios < r_min_mm))
    return {
        "cumple": violaciones == 0,
        "radio_minimo_mm": None if math.isinf(peor) else round(peor, 2),
        "num_violaciones": violaciones,
        "posicion_peor_mm": round(posicion, 1),
    }


def diametro_hidraulico(ancho_mm: float, alto_mm: float) -> float:
    """Dh = 4*Area / Perimetro = 2ab/(a+b) para un rectangulo."""
    return 2.0 * ancho_mm * alto_mm / (ancho_mm + alto_mm)


def validar_seccion(ancho_mm: float, alto_mm: float, parametros: dict) -> dict:
    """Area y relacion de aspecto de la seccion contra los parametros aprobados.
    Si falta un parametro, ese chequeo se marca 'no_verificado' en lugar de aprobarse en silencio."""
    area = ancho_mm * alto_mm
    aspecto = max(ancho_mm, alto_mm) / min(ancho_mm, alto_mm)
    resultado: dict = {
        "area_mm2": round(area, 2),
        "relacion_aspecto": round(aspecto, 3),
        "diametro_hidraulico_mm": round(diametro_hidraulico(ancho_mm, alto_mm), 2),
    }
    amin, amax = parametros.get("area_min_mm2"), parametros.get("area_max_mm2")
    if amin is None or amax is None:
        resultado["area_ok"] = "no_verificado"
    else:
        resultado["area_ok"] = bool(amin <= area <= amax)
    amax_aspecto = parametros.get("aspect_ratio_max")
    resultado["aspecto_ok"] = "no_verificado" if amax_aspecto is None else bool(aspecto <= amax_aspecto)
    return resultado


# --------------------------------------------------------------------------
# Marcos de transporte paralelo y rectangulos por estacion
# --------------------------------------------------------------------------
def _unitario(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < _EPS:
        raise ValueError("Vector nulo: la ruta tiene un tramo de longitud cero.")
    return v / n


def _tangentes(puntos: np.ndarray) -> np.ndarray:
    t = np.empty_like(puntos)
    t[1:-1] = puntos[2:] - puntos[:-2]
    t[0] = puntos[1] - puntos[0]
    t[-1] = puntos[-1] - puntos[-2]
    return np.array([_unitario(v) for v in t])


def marcos_transporte_paralelo(puntos: np.ndarray, arriba=(0.0, 0.0, 1.0)):
    """Para cada punto devuelve (tangente, normal, binormal) ortonormales. La normal arranca lo mas
    parecida posible a `arriba` y despues se transporta sin girar sobre la tangente (metodo de la doble
    reflexion). Asi el rectangulo no se tuerce a lo largo de la curva.
    Devuelve tres arreglos (n, 3): T, N, B."""
    t = _tangentes(puntos)
    n = np.empty_like(t)

    arriba = np.asarray(arriba, dtype=float)
    n0 = arriba - np.dot(arriba, t[0]) * t[0]
    if np.linalg.norm(n0) < 1e-6:  # el ducto arranca vertical: se elige otra referencia
        alterno = np.array([1.0, 0.0, 0.0])
        n0 = alterno - np.dot(alterno, t[0]) * t[0]
    n[0] = _unitario(n0)

    for i in range(len(puntos) - 1):
        v1 = puntos[i + 1] - puntos[i]
        c1 = np.dot(v1, v1)
        r_l = n[i] - (2.0 / c1) * np.dot(v1, n[i]) * v1
        t_l = t[i] - (2.0 / c1) * np.dot(v1, t[i]) * v1
        v2 = t[i + 1] - t_l
        c2 = np.dot(v2, v2)
        r_sig = r_l - (2.0 / c2) * np.dot(v2, r_l) * v2 if c2 > _EPS else r_l
        # re-proyecta sobre el plano perpendicular a la tangente para no acumular error numerico
        r_sig = r_sig - np.dot(r_sig, t[i + 1]) * t[i + 1]
        n[i + 1] = _unitario(r_sig)

    b = np.cross(t, n)
    return t, n, b


def estaciones_rectangulares(puntos, t, n, b, ancho_mm, alto_mm, extension_mm: float = 0.0):
    """Lista de rectangulos (cada uno con 4 esquinas 3D) perpendiculares a la ruta.
    `ancho` va en la direccion de la binormal y `alto` en la de la normal (que arranca vertical).
    Todas las estaciones usan el mismo orden de esquinas, lo que evita que el loft se tuerza.
    extension_mm > 0 agrega una estacion extra de cada lado, prolongando la ruta en linea recta: se usa
    para que el solido interior sobresalga y el corte deje los extremos del ducto abiertos."""
    centros, tt, nn, bb = list(puntos), list(t), list(n), list(b)
    if extension_mm > 0:
        centros = [puntos[0] - t[0] * extension_mm] + centros + [puntos[-1] + t[-1] * extension_mm]
        tt, nn, bb = [t[0]] + tt + [t[-1]], [n[0]] + nn + [n[-1]], [b[0]] + bb + [b[-1]]

    mw, mh = ancho_mm / 2.0, alto_mm / 2.0
    rectangulos = []
    for c, ni, bi in zip(centros, nn, bb):
        rectangulos.append(
            np.array(
                [
                    c + mw * bi + mh * ni,
                    c - mw * bi + mh * ni,
                    c - mw * bi - mh * ni,
                    c + mw * bi - mh * ni,
                ]
            )
        )
    return rectangulos


def _submuestrear(puntos: np.ndarray, max_estaciones: int) -> np.ndarray:
    """Un loft con cientos de perfiles es lento y ondulado: se conservan a lo mas `max_estaciones`,
    siempre incluyendo el primero y el ultimo punto."""
    if len(puntos) <= max_estaciones:
        return puntos
    indices = np.unique(np.linspace(0, len(puntos) - 1, max_estaciones).round().astype(int))
    return puntos[indices]


# --------------------------------------------------------------------------
# CadQuery: loft exterior - loft interior, y exportacion
# --------------------------------------------------------------------------
def _nombre_valido(nombre: str) -> str:
    """El nombre de archivo de salida: solo letras, numeros, guion y guion bajo."""
    if not nombre or not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", nombre):
        raise ValueError("El nombre solo puede tener letras, numeros, '-' y '_' (maximo 60 caracteres).")
    return nombre


def construir_ducto(puntos, ancho_mm, alto_mm, espesor_mm, max_estaciones: int = 40):
    """Devuelve el solido CadQuery del ducto hueco (extremos abiertos)."""
    import cadquery as cq  # import perezoso: es pesada y solo se usa aqui

    if espesor_mm <= 0 or espesor_mm >= min(ancho_mm, alto_mm) / 2.0:
        raise ValueError("El espesor de pared debe ser positivo y menor que la mitad de la dimension mas chica.")

    puntos = _submuestrear(np.asarray(puntos, dtype=float), max_estaciones)
    t, n, b = marcos_transporte_paralelo(puntos)

    def hacer_loft(rectangulos):
        alambres = []
        for esquinas in rectangulos:
            vectores = [cq.Vector(*map(float, p)) for p in esquinas]
            vectores.append(vectores[0])  # se repite el primero para cerrar el alambre
            alambres.append(cq.Wire.makePolygon(vectores))
        return cq.Solid.makeLoft(alambres, False)

    exterior = hacer_loft(estaciones_rectangulares(puntos, t, n, b, ancho_mm, alto_mm))
    interior = hacer_loft(
        estaciones_rectangulares(
            puntos, t, n, b,
            ancho_mm - 2 * espesor_mm, alto_mm - 2 * espesor_mm,
            extension_mm=espesor_mm + 1.0,
        )
    )
    return exterior.cut(interior)


def exportar(shape, nombre: str) -> dict:
    """Escribe data/cad/<nombre>.step y .stl."""
    import cadquery as cq

    asegurar_carpetas()
    nombre = _nombre_valido(nombre)
    archivos = {}
    for ext in ("step", "stl"):
        destino = ruta_cad / f"{nombre}.{ext}"
        cq.exporters.export(shape, str(destino))
        archivos[ext] = destino.name
    return archivos


# --------------------------------------------------------------------------
# Punto de entrada: ruta del A* -> ducto + reporte
# --------------------------------------------------------------------------
def generar_ducto(
    ruta,
    ancho_mm: float,
    alto_mm: float,
    espesor_mm: float,
    parametros: dict | None = None,
    nombre: str = "ducto",
    mm_por_celda: float = 1.0,
    exportar_archivos: bool = True,
    curva=None,
) -> dict:
    """Ruta del A* -> ducto rectangular hueco en data/cad/ + reporte de validacion (dict simple, listo
    para mostrarse o para que el LLM lo narre).

    ruta:         lista de [x, y, z]. Si el A* devuelve indices de celda, pasa mm_por_celda para escalar.
    parametros:   dict como PARAMETROS de agente.py (r_min_mm, area_min_mm2, area_max_mm2, aspect_ratio_max).
                  Lo que falte se reporta como 'no_verificado'.
    curva:        (n, 3) opcional. Si se da (por ejemplo desde cad.curva_ruta.construir_curva, con arcos de
                  radio exacto) se usa TAL CUAL como linea central y no se suaviza la ruta con Chaikin.
    Siempre genera el ducto, aun si no cumple r_min: decide el humano, y el reporte dice por que."""
    parametros = parametros or {}
    nombre = _nombre_valido(nombre)

    if curva is None:
        escalada = _como_puntos(ruta) * mm_por_celda
        paso = max(1.0, min(ancho_mm, alto_mm) / 4.0)  # fino, para medir bien la curvatura
        curva = suavizar_ruta(escalada, paso_mm=paso)
    else:
        curva = _como_puntos(curva)

    r_min = parametros.get("r_min_mm")
    if r_min is None:
        radio = validar_radio_minimo(curva, 0.0)
        radio["cumple"] = "no_verificado"
    else:
        radio = validar_radio_minimo(curva, float(r_min))
        radio["r_min_exigido_mm"] = float(r_min)

    shape = construir_ducto(curva, ancho_mm, alto_mm, espesor_mm)

    reporte = {
        "nombre": nombre,
        "longitud_mm": round(longitud_mm(curva), 1),
        "seccion": validar_seccion(ancho_mm, alto_mm, parametros),
        "radio_curvatura": radio,
        "espesor_mm": float(espesor_mm),
        "volumen_mm3": round(float(shape.Volume()), 1),
        "solido_valido": bool(shape.isValid()),
    }
    if exportar_archivos:
        reporte["archivos"] = exportar(shape, nombre)
    return reporte
