"""
Convierte la ruta que calcula el A* (una lista de puntos [x, y, z]) en un ducto rectangular hueco y lo exporta a
STEP/STL.

Flujo:
  ruta del A* 
    -> suavizar_ruta        corta las esquinas usndo Chaikin y remuestrea a paso uniforme
    -> validar_*            compara contra los parametros aprobados
    -> estaciones           un rectangulo por estacion, orientado con transporte paralelo
    -> loft                 sólido exterior, menos sólido interior = ducto hueco
    -> export               data/cad/<nombre>.step y .stl
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


def validar_seccion(ancho_mm: float, alto_mm: float, parametros: dict, espesor_mm: float | None = None) -> dict:
    """Area y relacion de aspecto de la seccion contra los parametros aprobados.
    Si falta un parametro, ese chequeo se marca 'no_verificado' en lugar de aprobarse en silencio.

    ancho/alto son medidas EXTERIORES. Con espesor_mm, las comprobaciones (area, aspecto, D_h) se hacen sobre la seccion
    INTERIOR (ancho - 2e, alto - 2e), que es por donde pasa el aire; sin espesor_mm se usan las exteriores (como antes).
    Con espesor 2 mm en un ducto de 40 x 25, el area exterior es 1000 mm2 y la de paso 756 mm2: no es un detalle."""
    if espesor_mm is not None:
        ancho_mm, alto_mm = ancho_mm - 2.0 * espesor_mm, alto_mm - 2.0 * espesor_mm
        if min(ancho_mm, alto_mm) <= 0:
            raise ValueError("El espesor de pared deja una seccion interior sin espacio.")
    area = ancho_mm * alto_mm
    aspecto = max(ancho_mm, alto_mm) / min(ancho_mm, alto_mm)
    resultado: dict = {
        "area_mm2": round(area, 2),
        "relacion_aspecto": round(aspecto, 3),
        "diametro_hidraulico_mm": round(diametro_hidraulico(ancho_mm, alto_mm), 2),
        "medida": "interior" if espesor_mm is not None else "exterior",
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
# Transicion de seccion a lo largo de la ruta
# --------------------------------------------------------------------------
LEYES_TRANSICION = ("suave", "lineal")


def fraccion_de_transicion(u, ley: str = "suave") -> np.ndarray:
    """Fraccion del cambio de seccion (0 = puerto inicial, 1 = puerto final) para u en [0, 1] (distancia relativa)."""
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    if ley == "lineal":
        return u
    if ley == "suave":
        return u * u * (3.0 - 2.0 * u)
    raise ValueError(f"Ley de transicion desconocida: {ley!r}. Opciones: {', '.join(LEYES_TRANSICION)}.")


def _secciones_validas(ancho_mm, alto_mm, ancho_fin_mm, alto_fin_mm) -> tuple[float, float, float, float]:
    """Completa la seccion final (si no se da, es igual a la inicial) y exige todo > 0 y fin completo o ausente."""
    if (ancho_fin_mm is None) != (alto_fin_mm is None):
        raise ValueError("Para cambiar la seccion en el puerto final hay que dar ancho_fin_mm Y alto_fin_mm.")
    a1 = float(ancho_mm if ancho_fin_mm is None else ancho_fin_mm)
    h1 = float(alto_mm if alto_fin_mm is None else alto_fin_mm)
    a0, h0 = float(ancho_mm), float(alto_mm)
    if min(a0, h0, a1, h1) <= 0:
        raise ValueError("Ancho y alto de las secciones deben ser positivos.")
    return a0, h0, a1, h1


def validar_secciones_control(secciones) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[(pos, ancho, alto), ...] -> (pos, ancho, alto) como arreglos. Exige >= 2 secciones, pos estrictamente creciente
    empezando en 0 y terminando en 1, y anchos y altos positivos."""
    try:
        m = np.asarray([[float(c) for c in fila] for fila in secciones], dtype=float)
    except (TypeError, ValueError):
        raise ValueError("secciones debe ser una lista de (pos, ancho_mm, alto_mm).") from None
    if m.ndim != 2 or m.shape[1] != 3 or len(m) < 2:
        raise ValueError("secciones necesita al menos 2 secciones, cada una (pos, ancho_mm, alto_mm).")
    if not np.all(np.isfinite(m)):
        raise ValueError("secciones tiene valores no numericos (nan o infinito).")
    pos, ancho, alto = m[:, 0], m[:, 1], m[:, 2]
    if abs(pos[0]) > 1e-9 or abs(pos[-1] - 1.0) > 1e-9:
        raise ValueError("La primera seccion debe estar en pos=0 (puerto inicial) y la ultima en pos=1 (puerto final).")
    if np.any(np.diff(pos) <= 0):
        raise ValueError("Las posiciones de las secciones deben crecer estrictamente.")
    if ancho.min() <= 0 or alto.min() <= 0:
        raise ValueError("Ancho y alto de las secciones deben ser positivos.")
    return pos, ancho, alto


def _pendientes_pchip(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pendientes de Fritsch-Carlson (como PCHIP de scipy): 0 en los extremos locales, media armonica ponderada en el
    resto, y formula de tres puntos que conserva la forma en los extremos."""
    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros(len(x))
    if len(x) == 2:
        d[:] = delta[0]
        return d
    for k in range(1, len(x) - 1):
        if delta[k - 1] * delta[k] > 0.0:
            w1, w2 = 2.0 * h[k] + h[k - 1], h[k] + 2.0 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])

    def extremo(h0, h1, d0, d1):
        v = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if np.sign(v) != np.sign(d0):
            return 0.0
        if np.sign(d0) != np.sign(d1) and abs(v) > 3.0 * abs(d0):
            return 3.0 * d0
        return v

    d[0] = extremo(h[0], h[1], delta[0], delta[1])
    d[-1] = extremo(h[-1], h[-2], delta[-1], delta[-2])
    return d


def interpolar_pchip(x, y, u) -> np.ndarray:
    """Valor en u de la cubica monotona (PCHIP) que pasa por los nodos (x, y). Solo numpy."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    u = np.clip(np.asarray(u, dtype=float), x[0], x[-1])
    d = _pendientes_pchip(x, y)
    i = np.clip(np.searchsorted(x, u, side="right") - 1, 0, len(x) - 2)
    h = x[i + 1] - x[i]
    t = (u - x[i]) / h
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * y[i] + (t3 - 2 * t2 + t) * h * d[i]
            + (-2 * t3 + 3 * t2) * y[i + 1] + (t3 - t2) * h * d[i + 1])


def validar_rectos(rectos_mm) -> tuple[float, float]:
    """None -> (0, 0). Si no: dos longitudes >= 0 (mm) de los tramos rectos de entrada y salida."""
    if rectos_mm is None:
        return 0.0, 0.0
    try:
        l0, l1 = (float(v) for v in rectos_mm)
    except (TypeError, ValueError):
        raise ValueError("rectos_mm debe ser (tramo_inicial_mm, tramo_final_mm).") from None
    if not (math.isfinite(l0) and math.isfinite(l1)) or l0 < 0 or l1 < 0:
        raise ValueError("Los tramos rectos deben ser numeros >= 0.")
    return l0, l1


def perfil_de_seccion(puntos: np.ndarray, ancho_mm=None, alto_mm=None, ancho_fin_mm=None, alto_fin_mm=None,
                      ley: str = "suave", secciones=None, rectos_mm=None) -> tuple[np.ndarray, np.ndarray]:
    """(anchos, altos) de la seccion en cada punto de la linea central, segun la distancia recorrida.
    Modo clasico: ancho/alto (+ fin opcional) con la ley 'suave' o 'lineal'.
    Modo por estaciones: secciones=[(pos, ancho, alto), ...] (ancho/alto/fin deben ser None); ley 'suave' = PCHIP,
    'lineal' = tramos rectos.
    rectos_mm=(l0, l1): la seccion es constante en los primeros l0 mm y los ultimos l1 mm, y el cambio (u de 0 a 1)
    ocurre solo entre ellos."""
    if ley not in LEYES_TRANSICION:
        raise ValueError(f"Ley de transicion desconocida: {ley!r}. Opciones: {', '.join(LEYES_TRANSICION)}.")
    l0, l1 = validar_rectos(rectos_mm)
    tramos = np.linalg.norm(np.diff(puntos, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(tramos)))
    if l0 + l1 > 0.0:
        util = s[-1] - l0 - l1
        if util <= 1e-6:
            raise ValueError(f"Los tramos rectos ({l0:g} + {l1:g} mm) ocupan todo el ducto ({s[-1]:.0f} mm).")
        u = np.clip((s - l0) / util, 0.0, 1.0)
    else:
        u = s / s[-1] if s[-1] > 0 else np.zeros_like(s)
    if secciones is not None:
        if any(v is not None for v in (ancho_mm, alto_mm, ancho_fin_mm, alto_fin_mm)):
            raise ValueError("Da las secciones POR ESTACIONES (secciones=...) o ancho/alto (+ fin), no las dos cosas.")
        pos, ancho, alto = validar_secciones_control(secciones)
        if ley == "lineal":
            return np.interp(u, pos, ancho), np.interp(u, pos, alto)
        return interpolar_pchip(pos, ancho, u), interpolar_pchip(pos, alto, u)
    if ancho_mm is None or alto_mm is None:
        raise ValueError("Falta ancho_mm y alto_mm (o secciones=...).")
    a0, h0, a1, h1 = _secciones_validas(ancho_mm, alto_mm, ancho_fin_mm, alto_fin_mm)
    f = fraccion_de_transicion(u, ley)
    return a0 + (a1 - a0) * f, h0 + (h1 - h0) * f


def cambio_max_relativo_por(valores: np.ndarray, s: np.ndarray, distancia_mm: float = 50.0) -> float:
    """Mayor |V(s + d) - V(s)| / V(s) en la curva (misma definicion que la ficha de los ductos de referencia)."""
    if len(valores) < 2 or s[-1] <= 0.0:
        return 0.0
    destino = np.interp(s + distancia_mm, s, valores)
    sirve = s + distancia_mm <= s[-1] + 1e-9
    if not sirve.any():  # ducto mas corto que la ventana: se compara extremo contra extremo
        return float(abs(valores[-1] - valores[0]) / max(valores[0], 1e-12))
    return float(np.max(np.abs(destino[sirve] - valores[sirve]) / np.maximum(valores[sirve], 1e-12)))


def validar_transicion(puntos: np.ndarray, ancho_mm, alto_mm, ancho_fin_mm, alto_fin_mm,
                       ley: str, parametros: dict, espesor_mm: float | None = None, secciones=None,
                       rectos_mm=None) -> dict:
    """Area, relacion de aspecto y cambio de area a lo largo del ducto, con las medidas NOMINALES que se piden.
    Con espesor_mm se usa la seccion interior (la de paso del aire), igual que validar_seccion.
    Con secciones=[(pos, ancho, alto), ...] el perfil es por estaciones (ancho/alto/fin deben ser None).
    Lo que falte en parametros se marca 'no_verificado'."""
    anchos, altos = perfil_de_seccion(puntos, ancho_mm, alto_mm, ancho_fin_mm, alto_fin_mm, ley, secciones, rectos_mm)
    if espesor_mm is not None:
        anchos, altos = anchos - 2.0 * espesor_mm, altos - 2.0 * espesor_mm
        if min(anchos.min(), altos.min()) <= 0:
            raise ValueError("El espesor de pared deja una seccion interior sin espacio.")
    tramos = np.linalg.norm(np.diff(puntos, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(tramos)))
    area = anchos * altos
    aspecto = np.maximum(anchos, altos) / np.minimum(anchos, altos)
    dh = 2.0 * anchos * altos / (anchos + altos)
    # semiangulo equivalente de un difusor/contraccion conica: tan(theta) = (dDh/ds) / 2
    ddh = np.gradient(dh, s) if len(s) > 1 and s[-1] > 0 else np.zeros_like(dh)
    ang = np.degrees(np.arctan(ddh / 2.0))
    r = {
        "ley": ley, "medida": "interior" if espesor_mm is not None else "exterior",
        "area_inicio_mm2": round(float(area[0]), 1), "area_fin_mm2": round(float(area[-1]), 1),
        "area_min_mm2": round(float(area.min()), 1), "area_max_mm2": round(float(area.max()), 1),
        "relacion_aspecto_max": round(float(aspecto.max()), 3),
        "cambio_max_de_area_por_50mm": round(cambio_max_relativo_por(area, s, 50.0), 3),
        "semiangulo_difusor_max_grados": round(float(max(ang.max(), 0.0)), 2),
        "semiangulo_contraccion_max_grados": round(float(max(-ang.min(), 0.0)), 2),
    }
    if rectos_mm is not None and any(validar_rectos(rectos_mm)):
        r["tramos_rectos_mm"] = list(validar_rectos(rectos_mm))
    if secciones is not None:
        r["n_secciones"] = len(secciones)
        r["interpolacion"] = "lineal" if ley == "lineal" else "pchip"
    amin, amax = parametros.get("area_min_mm2"), parametros.get("area_max_mm2")
    r["area_ok"] = "no_verificado" if amin is None or amax is None else bool(amin <= area.min() and area.max() <= amax)
    amax_asp = parametros.get("aspect_ratio_max")
    r["aspecto_ok"] = "no_verificado" if amax_asp is None else bool(aspecto.max() <= amax_asp)
    return r


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

    # ancho/alto pueden ser un numero (seccion constante) o un arreglo con un valor por punto de 'puntos'
    # (transicion de seccion); las estaciones de extension repiten el valor del extremo.
    anchos = np.broadcast_to(np.asarray(ancho_mm, dtype=float), (len(puntos),)).copy()
    altos = np.broadcast_to(np.asarray(alto_mm, dtype=float), (len(puntos),)).copy()
    if extension_mm > 0:
        anchos = np.concatenate(([anchos[0]], anchos, [anchos[-1]]))
        altos = np.concatenate(([altos[0]], altos, [altos[-1]]))
    rectangulos = []
    for c, ni, bi, w, h in zip(centros, nn, bb, anchos / 2.0, altos / 2.0):
        rectangulos.append(
            np.array(
                [
                    c + w * bi + h * ni,
                    c - w * bi + h * ni,
                    c - w * bi - h * ni,
                    c + w * bi - h * ni,
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


def construir_ducto(puntos, ancho_mm, alto_mm, espesor_mm, max_estaciones: int = 40,
                    ancho_fin_mm=None, alto_fin_mm=None, ley_transicion: str = "suave", secciones=None,
                    rectos_mm=None):
    """Devuelve el solido CadQuery del ducto hueco (extremos abiertos).
    ancho/alto = seccion del puerto inicial; ancho_fin_mm/alto_fin_mm (opcionales, juntos) = seccion del puerto
    final: la seccion cambia a lo largo de la ruta segun ley_transicion ('suave' o 'lineal').
    secciones=[(pos, ancho, alto), ...] (con ancho/alto/fin = None): perfil por estaciones (ver docstring del modulo).
    rectos_mm=(l0, l1): tramos rectos de los puertos donde la seccion no cambia."""
    import cadquery as cq  # import perezoso: es pesada y solo se usa aqui

    puntos = _submuestrear(np.asarray(puntos, dtype=float), max_estaciones)
    anchos, altos = perfil_de_seccion(puntos, ancho_mm, alto_mm, ancho_fin_mm, alto_fin_mm, ley_transicion, secciones,
                                      rectos_mm)
    if espesor_mm <= 0 or espesor_mm >= min(anchos.min(), altos.min()) / 2.0:
        raise ValueError("El espesor de pared debe ser positivo y menor que la mitad de la dimension mas chica "
                         "(en cualquier punto del ducto).")

    t, n, b = marcos_transporte_paralelo(puntos)

    def hacer_loft(rectangulos):
        alambres = []
        for esquinas in rectangulos:
            vectores = [cq.Vector(*map(float, p)) for p in esquinas]
            vectores.append(vectores[0])  # se repite el primero para cerrar el alambre
            alambres.append(cq.Wire.makePolygon(vectores))
        return cq.Solid.makeLoft(alambres, False)

    exterior = hacer_loft(estaciones_rectangulares(puntos, t, n, b, anchos, altos))
    interior = hacer_loft(
        estaciones_rectangulares(
            puntos, t, n, b,
            anchos - 2 * espesor_mm, altos - 2 * espesor_mm,
            extension_mm=espesor_mm + 1.0,
        )
    )
    return exterior.cut(interior)


def malla_numpy(puntos, ancho_mm, alto_mm, espesor_mm, max_estaciones: int = 40,
                ancho_fin_mm=None, alto_fin_mm=None, ley_transicion: str = "suave", secciones=None,
                rectos_mm=None) -> np.ndarray:
    """Malla de triangulos (n, 3, 3) CERRADA del mismo ducto hueco, hecha solo con numpy y con las MISMAS estaciones que
    usa construir_ducto (marcos, perfil de seccion, espesor). No sustituye al CAD (no es STEP); sirve para
    (1) medir con la ficha tecnica un ducto sin necesitar CadQuery y (2) comprobar de forma independiente que
    las estaciones estan bien: si la malla sale torcida o con la seccion equivocada, el loft de CadQuery tambien."""
    puntos = _submuestrear(np.asarray(puntos, dtype=float), max_estaciones)
    anchos, altos = perfil_de_seccion(puntos, ancho_mm, alto_mm, ancho_fin_mm, alto_fin_mm, ley_transicion, secciones,
                                      rectos_mm)
    if espesor_mm <= 0 or espesor_mm >= min(anchos.min(), altos.min()) / 2.0:
        raise ValueError("El espesor de pared debe ser positivo y menor que la mitad de la dimension mas chica "
                         "(en cualquier punto del ducto).")
    t, n, b = marcos_transporte_paralelo(puntos)
    ext = np.array(estaciones_rectangulares(puntos, t, n, b, anchos, altos))                          # (k, 4, 3)
    interior = np.array(estaciones_rectangulares(puntos, t, n, b, anchos - 2 * espesor_mm, altos - 2 * espesor_mm))
    k = len(puntos)
    tri = []
    for anillos, hacia_afuera in ((ext, True), (interior, False)):
        for i in range(k - 1):
            for j in range(4):
                jj = (j + 1) % 4
                a_, b_, c_, d_ = anillos[i][j], anillos[i][jj], anillos[i + 1][jj], anillos[i + 1][j]
                for tr in ((a_, b_, c_), (a_, c_, d_)):
                    tr = np.array(tr)
                    normal = np.cross(tr[1] - tr[0], tr[2] - tr[0])
                    radial = tr.mean(axis=0) - puntos[i]
                    radial -= np.dot(radial, t[i]) * t[i]               # solo la parte perpendicular al eje
                    # exterior: normal hacia afuera del eje; interior: hacia el eje (el material queda entre las dos)
                    if (np.dot(normal, radial) < 0) == hacia_afuera:
                        tr = tr[[0, 2, 1]]
                    tri.append(tr)
    for i, signo in ((0, -1.0), (k - 1, 1.0)):                          # tapas anulares en los extremos
        for j in range(4):
            jj = (j + 1) % 4
            for tr in ((ext[i][j], ext[i][jj], interior[i][jj]), (ext[i][j], interior[i][jj], interior[i][j])):
                tr = np.array(tr)
                if np.dot(np.cross(tr[1] - tr[0], tr[2] - tr[0]), signo * t[i]) < 0:
                    tr = tr[[0, 2, 1]]
                tri.append(tr)
    return np.array(tri)


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
    ancho_fin_mm: float | None = None,
    alto_fin_mm: float | None = None,
    ley_transicion: str = "suave",
    secciones=None,
    rectos_mm=None,
) -> dict:
    """Ruta del A* -> ducto rectangular hueco en data/cad/ + reporte de validacion (dict simple, listo
    para mostrarse o para que el LLM lo narre).

    secciones:    [(pos, ancho_mm, alto_mm), ...] perfil por estaciones (pos 0 = puerto inicial, 1 = final). Si se da,
                  ancho_mm/alto_mm/ancho_fin_mm/alto_fin_mm deben ser None; 'seccion' y 'seccion_fin' del reporte son
                  la primera y la ultima, y 'transicion' lleva n_secciones e interpolacion.

    ancho_mm/alto_mm: seccion del puerto inicial. ancho_fin_mm/alto_fin_mm (opcionales, los dos juntos): seccion
                  del puerto final; si se dan, la seccion cambia a lo largo de la ruta (ley_transicion 'suave' o
                  'lineal') y el reporte agrega 'seccion_fin' y 'transicion'. Sin ellos, todo igual que antes.

    rectos_mm:    (l0, l1) mm de tramo recto en cada puerto; la seccion no cambia ahi (ver docstring del modulo).

    ruta:         lista de [x, y, z]. Si el A* devuelve indices de celda, pasa mm_por_celda para escalar.
    parametros:   dict como PARAMETROS de agente.py (r_min_mm, area_min_mm2, area_max_mm2, aspect_ratio_max).
                  Lo que falte se reporta como 'no_verificado'.
    curva:        (n, 3) opcional. Si se da (por ejemplo desde cad.curva_ruta.construir_curva, con arcos de
                  radio exacto) se usa TAL CUAL como linea central y no se suaviza la ruta con Chaikin.
    Siempre genera el ducto, aun si no cumple r_min: decide el humano, y el reporte dice por que."""
    parametros = parametros or {}
    nombre = _nombre_valido(nombre)
    if ley_transicion not in LEYES_TRANSICION:
        raise ValueError(f"Ley de transicion desconocida: {ley_transicion!r}. Opciones: {', '.join(LEYES_TRANSICION)}.")
    if secciones is not None:
        if any(v is not None for v in (ancho_mm, alto_mm, ancho_fin_mm, alto_fin_mm)):
            raise ValueError("Da las secciones POR ESTACIONES (secciones=...) o ancho/alto (+ fin), no las dos cosas.")
        _, anchos_c, altos_c = validar_secciones_control(secciones)
        a0, h0, a1, h1 = float(anchos_c[0]), float(altos_c[0]), float(anchos_c[-1]), float(altos_c[-1])
        hay_transicion = bool(np.any(anchos_c != a0) or np.any(altos_c != h0))
        menor = float(min(anchos_c.min(), altos_c.min()))
    else:
        a0, h0, a1, h1 = _secciones_validas(ancho_mm, alto_mm, ancho_fin_mm, alto_fin_mm)
        hay_transicion = (a0, h0) != (a1, h1)
        menor = min(a0, h0, a1, h1)

    if curva is None:
        escalada = _como_puntos(ruta) * mm_por_celda
        paso = max(1.0, menor / 4.0)  # fino, para medir bien la curvatura
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

    shape = construir_ducto(curva, ancho_mm, alto_mm, espesor_mm, ancho_fin_mm=ancho_fin_mm, alto_fin_mm=alto_fin_mm,
                            ley_transicion=ley_transicion, secciones=secciones, rectos_mm=rectos_mm)

    reporte = {
        "nombre": nombre,
        "longitud_mm": round(longitud_mm(curva), 1),
        "seccion": validar_seccion(a0, h0, parametros, espesor_mm),
        "radio_curvatura": radio,
        "espesor_mm": float(espesor_mm),
        "volumen_mm3": round(float(shape.Volume()), 1),
        "solido_valido": bool(shape.isValid()),
    }
    if hay_transicion:
        reporte["seccion_fin"] = validar_seccion(a1, h1, parametros, espesor_mm)
        if secciones is not None:
            reporte["transicion"] = validar_transicion(curva, None, None, None, None, ley_transicion, parametros,
                                                       espesor_mm, secciones=secciones, rectos_mm=rectos_mm)
        else:
            reporte["transicion"] = validar_transicion(curva, a0, h0, a1, h1, ley_transicion, parametros, espesor_mm,
                                                       rectos_mm=rectos_mm)
    if exportar_archivos:
        reporte["archivos"] = exportar(shape, nombre)
    return reporte
