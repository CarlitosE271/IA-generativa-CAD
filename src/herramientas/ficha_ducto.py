"""
Ficha tecnica de un ducto a partir de su malla (STL): puertos, linea central, longitud, radios de curvatura y
perfil de seccion. Sirve para medir un ducto de REFERENCIA (los que da Maikol) y, con las mismas funciones, el
ducto que genere la IA, de modo que se comparen con las mismas reglas.

Solo numpy (como ingesta_malla.py): no necesita scipy ni trimesh.

COMO FUNCIONA
  1. Se voxeliza la pared del ducto (voxeles que tocan la superficie) y se queda la pieza principal (los clips o
     piezas sueltas quedan fuera).
  2. Se mide la distancia GEODESICA por la pared desde un extremo (como caminar sobre la superficie). Los dos
     voxeles mas lejanos entre si son los extremos del ducto.
  3. Se agrupan los voxeles por distancia geodesica y se promedian: eso da una linea central aproximada.
  4. Se corta el ducto con planos PERPENDICULARES a esa linea y en cada corte se mide la seccion
     (ancho, alto, area, diametro hidraulico). Se repite 2 veces recentrando.
  5. Con la linea central suavizada se calcula la curvatura (radio minimo) y la longitud.

LIMITES (decirlos con honestidad):
  - Las dimensiones son EXTERIORES (la pared mide ~1.6 mm); la interior estimada resta 2 x espesor. Costillas, orejas
    y clips inflan algunas secciones; por eso se usan percentiles y no el maximo.
  - La precision es del orden del voxel (paso_mm, 3 mm por defecto): sirve para comparar ductos entre si, NO para
    dimensionar. Se valido contra mediciones en CAD (ver tests y la conversacion con Maikol).
  - Un ducto con ramas (mas de dos salidas) tiene mas de dos extremos: aqui se toman los dos MAS LEJANOS.
  - Un STL no trae unidades: se asume mm (como el resto del proyecto).
  - Una seccion con tabique central (dos canales) se mide por fuera: el tabique no se descuenta del area.
"""
from __future__ import annotations

import math

import numpy as np

from herramientas import ingesta_malla as im

RADIO_MAX_SECCION_MM = 110.0  # solo se miden los voxeles a menos de esto del eje (deja fuera orejas y clips)
SALTO_MAX_RECENTRADO_MM = 8.0  # tope de cuanto puede moverse el centro en un paso de 6 mm (tras los primeros pasos)
PERCENTIL_SECCION = 3.0       # las dimensiones usan los percentiles 3 y 97 (no el maximo, que lo dominan las costillas)

_OFFS = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]
_PESOS = np.array([math.sqrt(dx * dx + dy * dy + dz * dz) for dx, dy, dz in _OFFS])


# ----------------------------------------------------------------------------------------------------
# Grafo de voxeles
# ----------------------------------------------------------------------------------------------------
def tabla_vecinos(idx: np.ndarray) -> np.ndarray:
    """(N, 26) con el indice del vecino en cada una de las 26 direcciones (-1 si no hay voxel ahi)."""
    n = len(idx)
    dims = idx.max(axis=0) + 3
    lin = ((idx[:, 0] + 1) * dims[1] + (idx[:, 1] + 1)) * dims[2] + (idx[:, 2] + 1)
    orden = np.argsort(lin)
    lin_ord = lin[orden]
    nb = np.full((n, len(_OFFS)), -1, dtype=np.int64)
    for k, (dx, dy, dz) in enumerate(_OFFS):
        q = lin + (dx * dims[1] + dy) * dims[2] + dz
        pos = np.clip(np.searchsorted(lin_ord, q), 0, n - 1)
        ok = lin_ord[pos] == q
        nb[ok, k] = orden[pos[ok]]
    return nb


def componentes(nb: np.ndarray) -> np.ndarray:
    """Etiqueta de componente conexa (26 vecinos) de cada voxel."""
    n = len(nb)
    lab = np.full(n, -1, dtype=np.int64)
    c = 0
    for s in range(n):
        if lab[s] >= 0:
            continue
        lab[s] = c
        frente = np.array([s])
        while len(frente):
            v = nb[frente].ravel()
            v = np.unique(v[v >= 0])
            v = v[lab[v] < 0]
            lab[v] = c
            frente = v
        c += 1
    return lab


def distancia_geodesica(nb: np.ndarray, fuente: int, paso: float) -> np.ndarray:
    """Distancia (mm) caminando de voxel en voxel (26 vecinos) desde 'fuente'. inf = inalcanzable."""
    d = np.full(len(nb), np.inf)
    d[fuente] = 0.0
    frente = np.array([fuente])
    while len(frente):
        mejorados = []
        for k in range(nb.shape[1]):
            v = nb[frente, k]
            ok = v >= 0
            if not ok.any():
                continue
            cand = d[frente[ok]] + _PESOS[k] * paso
            destino = v[ok]
            mejora = cand < d[destino]
            if mejora.any():
                np.minimum.at(d, destino[mejora], cand[mejora])
                mejorados.append(destino[mejora])
        frente = np.unique(np.concatenate(mejorados)) if mejorados else np.array([], dtype=np.int64)
    return d


# ----------------------------------------------------------------------------------------------------
# Geometria 2D de una seccion
# ----------------------------------------------------------------------------------------------------
def casco_convexo(q: np.ndarray) -> np.ndarray:
    """Casco convexo 2D (cadena monotona). Devuelve los vertices en orden antihorario."""
    p = np.unique(np.round(q, 6), axis=0)
    if len(p) < 3:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def cruz(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    inf, sup = [], []
    for pt in p:
        while len(inf) >= 2 and cruz(inf[-2], inf[-1], pt) <= 0:
            inf.pop()
        inf.append(pt)
    for pt in p[::-1]:
        while len(sup) >= 2 and cruz(sup[-2], sup[-1], pt) <= 0:
            sup.pop()
        sup.append(pt)
    return np.array(inf[:-1] + sup[:-1])


def area_perimetro(h: np.ndarray) -> tuple[float, float]:
    if len(h) < 3:
        return 0.0, 0.0
    x, y = h[:, 0], h[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    per = float(np.sum(np.linalg.norm(h - np.roll(h, -1, axis=0), axis=1)))
    return area, per


def _base_plano(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(t, ref)
    u /= np.linalg.norm(u)
    return u, np.cross(t, u)


def medir_seccion(P: np.ndarray, centro: np.ndarray, t: np.ndarray, grosor: float, radio_max: float,
                  espesor_pared: float, pct: float = 1.0, valores: np.ndarray | None = None):
    """Corta con el plano (centro, normal t). Devuelve dict de la seccion o None si hay muy pocos puntos.
    ancho = extension mayor, alto = extension menor (ejes principales de la seccion, con percentiles pct/100-pct).
    'valores' (opcional, uno por punto): se devuelve su promedio sobre el corte en 'valor_medio'."""
    rel = P - centro
    sel = (np.abs(rel @ t) <= grosor / 2.0) & (np.linalg.norm(rel, axis=1) <= radio_max)
    if sel.sum() < 12:
        return None
    u0, v0 = _base_plano(t)
    q = np.stack([rel[sel] @ u0, rel[sel] @ v0], axis=1)
    c2 = q.mean(axis=0)
    _, vec = np.linalg.eigh(np.cov((q - c2).T))
    e1, e2 = vec[:, 1], vec[:, 0]  # e1 = direccion de mayor varianza
    a, b = (q - c2) @ e1, (q - c2) @ e2
    amin, amax = np.percentile(a, [pct, 100 - pct])
    bmin, bmax = np.percentile(b, [pct, 100 - pct])
    ancho, alto = float(amax - amin), float(bmax - bmin)
    if alto > ancho:
        ancho, alto = alto, ancho
        amin, amax, bmin, bmax = bmin, bmax, amin, amax
        e1, e2 = e2, e1
    area, per = area_perimetro(casco_convexo(np.stack([a, b], axis=1)))
    dc = ((amin + amax) / 2.0) * e1 + ((bmin + bmax) / 2.0) * e2 + c2
    nuevo_centro = centro + dc[0] * u0 + dc[1] * v0
    t_in = espesor_pared
    area_int = max(area - per * t_in, 0.0)
    per_int = max(per - 8.0 * t_in, 1e-9)
    return {
        "centro": nuevo_centro, "ancho_ext_mm": ancho, "alto_ext_mm": alto,
        "ancho_int_mm": max(ancho - 2 * t_in, 0.0), "alto_int_mm": max(alto - 2 * t_in, 0.0),
        "area_ext_mm2": area, "area_int_mm2": area_int,
        "diam_hidraulico_mm": 4.0 * area_int / per_int, "n_puntos": int(sel.sum()),
        "valor_medio": float(np.mean(valores[sel])) if valores is not None else None,
    }


# ----------------------------------------------------------------------------------------------------
# Linea central y curvatura
# ----------------------------------------------------------------------------------------------------
def _suavizar(c: np.ndarray, ventana: int) -> np.ndarray:
    if len(c) < 3 or ventana < 2:
        return c.copy()
    k = ventana // 2
    out = c.copy()
    for i in range(len(c)):
        i0, i1 = max(0, i - k), min(len(c), i + k + 1)
        out[i] = c[i0:i1].mean(axis=0)
    out[0], out[-1] = c[0], c[-1]
    return out


def _tangentes(c: np.ndarray, saltos: int = 3) -> np.ndarray:
    t = np.zeros_like(c)
    for i in range(len(c)):
        a, b = max(0, i - saltos), min(len(c) - 1, i + saltos)
        v = c[b] - c[a]
        t[i] = v / max(np.linalg.norm(v), 1e-12)
    return t


def _mediana_movil(c: np.ndarray, ventana: int) -> np.ndarray:
    """Mediana movil por coordenada: quita picos de 1-2 puntos (orejas, clips) sin redondear las curvas."""
    k = ventana // 2
    out = c.copy()
    for i in range(len(c)):
        out[i] = np.median(c[max(0, i - k): i + k + 1], axis=0)
    out[0], out[-1] = c[0], c[-1]
    return out


def _quitar_ganchos(c: np.ndarray, grados: float = 45.0, previos: int = 5) -> np.ndarray:
    """Quita puntos al final de cada extremo cuyo rumbo se desvia mas de 'grados' del rumbo de los 'previos' pasos
    anteriores: es el recentrado persiguiendo una oreja o una brida, no el ducto."""
    c = c.copy()
    cos_min = math.cos(math.radians(grados))
    for _ in range(4):
        for lado in (0, 1):
            if len(c) < previos + 3:
                return c
            v = c if lado == 0 else c[::-1]
            ref = v[0] - v[previos]          # rumbo hacia el extremo, medido en el tramo previo
            ult = v[0] - v[1]
            nr, nu = np.linalg.norm(ref), np.linalg.norm(ult)
            if nr > 1e-9 and nu > 1e-9 and np.dot(ref, ult) / (nr * nu) < cos_min:
                c = c[1:] if lado == 0 else c[:-1]
    return c


def longitud_polilinea(c: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1))) if len(c) > 1 else 0.0


def radios_de_curvatura(c: np.ndarray, separacion_mm: float = 50.0) -> np.ndarray:
    """Radio local (mm) en cada punto de la linea central: circulo por tres puntos separados 'separacion_mm' de arco.
    inf = recta. Los extremos (a menos de 'separacion_mm' del final) quedan en nan."""
    seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    r = np.full(len(c), np.nan)
    for i in range(len(c)):
        if s[i] - separacion_mm < 0 or s[i] + separacion_mm > s[-1]:
            continue
        a = c[int(np.searchsorted(s, s[i] - separacion_mm))]
        b = c[i]
        d = c[min(int(np.searchsorted(s, s[i] + separacion_mm)), len(c) - 1)]
        ab, bd, ad = np.linalg.norm(b - a), np.linalg.norm(d - b), np.linalg.norm(d - a)
        area2 = np.linalg.norm(np.cross(b - a, d - a))  # = 2 x area del triangulo
        r[i] = np.inf if area2 < 1e-9 else (ab * bd * ad) / (2.0 * area2)
    return r


def marchar(P: np.ndarray, semilla: np.ndarray, t0: np.ndarray, paso_s: float, espesor_pared: float,
            grosor: float, dist_geodesica: np.ndarray, sentido: int, tolerancia_mm: float = 40.0,
            max_pasos: int = 1000, k_tangente: int = 4) -> np.ndarray:
    """Avanza desde 'semilla' en la direccion t0: en cada paso corta con un plano perpendicular y recentra.
    Devuelve los centros (incluye la semilla). Esto evita el sesgo de las franjas geodesicas, que cerca de los
    extremos se tuercen.
    'dist_geodesica' (una por voxel) y 'sentido' (+1 si al avanzar crece, -1 si decrece) sirven de freno: en el
    final de un ducto real hay bridas y el corte puede regresar por el mismo tubo. Si el promedio geodesico del corte
    retrocede mas de 'tolerancia_mm' respecto al maximo alcanzado, se corta el camino en ese maximo (el extremo).
    Tambien se detiene cuando el corte ya no tiene voxeles o el recentrado quiere retroceder."""
    cs = [np.asarray(semilla, float)]
    t = np.asarray(t0, float) / np.linalg.norm(t0)
    mejor, mejor_i = -np.inf, 0
    for _ in range(max_pasos):
        pred = cs[-1] + t * paso_s
        m = medir_seccion(P, pred, t, grosor, RADIO_MAX_SECCION_MM, espesor_pared, PERCENTIL_SECCION,
                          valores=dist_geodesica)
        if m is None:
            break
        c = m["centro"]
        desvio = c - pred
        if len(cs) > 3 and np.linalg.norm(desvio) > SALTO_MAX_RECENTRADO_MM:
            # la linea central de un ducto cambia suave: un salto grande es una oreja o brida que desplazo el corte
            c = pred + desvio * (SALTO_MAX_RECENTRADO_MM / np.linalg.norm(desvio))
        if float(np.dot(c - cs[-1], t)) < 0.3 * paso_s:  # el recentrado no puede hacernos retroceder
            break
        cs.append(c)
        progreso = sentido * m["valor_medio"]
        if progreso > mejor:
            mejor, mejor_i = progreso, len(cs) - 1
        elif progreso < mejor - tolerancia_mm:  # el camino dio la vuelta
            break
        v = cs[-1] - cs[max(0, len(cs) - 1 - k_tangente)]
        t = v / max(np.linalg.norm(v), 1e-12)
    return np.array(cs[: mejor_i + 1] if mejor_i > 0 else cs)


# ----------------------------------------------------------------------------------------------------
# Ficha completa
# ----------------------------------------------------------------------------------------------------
def espesor_de_pared(tri: np.ndarray) -> float | None:
    """Espesor aproximado = volumen / (area superficial / 2). None si la malla no es cerrada."""
    info = im.info_malla(tri)
    if not info["volumen_mm3"]:
        return None
    area = 0.5 * float(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())
    return float(info["volumen_mm3"] / (area / 2.0))


def ficha_de_malla(tri: np.ndarray, paso_mm: float = 3.0, espesor_pared_mm: float | None = None,
                   separacion_radio_mm: float = 50.0, paso_perfil_mm: float = 6.0) -> dict:
    """Ficha tecnica de un ducto (triangulos en mm). Devuelve un dict serializable a JSON salvo la clave '_interno'."""
    t_pared = espesor_pared_mm or espesor_de_pared(tri) or 1.6
    lo = tri.reshape(-1, 3).min(axis=0)
    hi = tri.reshape(-1, 3).max(axis=0)
    dims = np.ceil((hi - lo) / paso_mm).astype(np.int64) + 1
    if int(np.prod(dims)) > im.MAX_VOXELES * 3:
        raise im.IngestaError("El ducto es demasiado grande para este paso: aumenta paso_mm.")
    sup = im.voxelizar_superficie(tri, paso_mm, lo, dims)
    idx = np.argwhere(sup)
    nb = tabla_vecinos(idx)
    lab = componentes(nb)
    cuentas = np.bincount(lab)
    principal = int(np.argmax(cuentas))
    advertencias = []
    if len(cuentas) > 1:
        sueltas = sorted((int(c) for c in cuentas if c != cuentas[principal]), reverse=True)[:5]
        advertencias.append(f"{len(cuentas) - 1} pieza(s) suelta(s) ignoradas (voxeles: {sueltas})")
    # se vuelve a armar el grafo solo con la pieza principal
    idx = idx[lab == principal]
    nb = tabla_vecinos(idx)
    P = lo + (idx + 0.5) * paso_mm  # centro de cada voxel

    d0 = distancia_geodesica(nb, 0, paso_mm)
    a = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
    da = distancia_geodesica(nb, a, paso_mm)
    b = int(np.argmax(np.where(np.isfinite(da), da, -1)))
    s = da

    # linea central inicial: centroides de franjas de distancia geodesica
    paso_s = paso_perfil_mm
    nbin = int(s[np.isfinite(s)].max() // paso_s) + 1
    centros = []
    for k in range(nbin):
        m = (s >= k * paso_s) & (s < (k + 1) * paso_s)
        if m.sum() >= 6:
            centros.append(P[m].mean(axis=0))
    centros = np.array(centros)
    if len(centros) < 8:
        raise im.IngestaError("No se pudo trazar la linea central (muy pocos voxeles): revisa unidades o el paso.")
    centros = _suavizar(centros, 9)

    # la linea central definitiva se traza CAMINANDO desde el centro del ducto hacia cada extremo, recentrando en
    # cada corte perpendicular (radio_max chico: deja fuera orejas y clips que inflan la seccion)
    grosor = max(paso_mm * 2, 6.0)

    def trazar(fraccion):
        """Camina desde un punto de la linea aproximada (fraccion de su largo) hacia los dos extremos.
        s crece de 'a' hacia 'b', asi que 'adelante' (hacia b) lleva sentido +1."""
        m0 = min(max(int(len(centros) * fraccion), 5), len(centros) - 6)
        k = 8
        tt = centros[min(m0 + k, len(centros) - 1)] - centros[max(m0 - k, 0)]
        tt = tt / max(np.linalg.norm(tt), 1e-12)
        semilla = centros[m0]
        for _ in range(2):
            m = medir_seccion(P, semilla, tt, grosor, RADIO_MAX_SECCION_MM, t_pared, PERCENTIL_SECCION)
            if m:
                semilla = m["centro"]
        adelante = marchar(P, semilla, tt, paso_s, t_pared, grosor, s, +1)
        atras = marchar(P, semilla, -tt, paso_s, t_pared, grosor, s, -1)
        camino = np.concatenate([atras[::-1], adelante[1:]])
        error = float(np.linalg.norm(camino[0] - P[a]) + np.linalg.norm(camino[-1] - P[b]))
        return camino, error

    # un punto de partida malo (zona estrecha, brida) puede frenar la marcha: se prueban varios y se toma el primero
    # que llega a los dos extremos (a menos de 130 mm de ellos; los extremos del voxel pueden ser orejas)
    crudos, mejor_error = None, np.inf
    for fraccion in (0.5, 0.35, 0.65, 0.25, 0.75):
        camino, error = trazar(fraccion)
        if error < mejor_error:
            crudos, mejor_error = camino, error
        if (np.linalg.norm(camino[0] - P[a]) <= 130.0 and np.linalg.norm(camino[-1] - P[b]) <= 130.0):
            break
    for extremo, objetivo, nombre in ((crudos[0], P[a], "inicio"), (crudos[-1], P[b], "final")):
        if np.linalg.norm(objetivo - extremo) > 130.0:
            advertencias.append(f"la linea central no llego al extremo del {nombre} (quedo a mas de 130 mm): "
                                "revisa si el ducto tiene ramas o cortes")
    crudos = _quitar_ganchos(_mediana_movil(crudos, 7))
    centros = _suavizar(crudos, 5)
    secciones = []
    for c, ti in zip(centros, _tangentes(centros)):
        secciones.append(medir_seccion(P, c, ti, grosor, RADIO_MAX_SECCION_MM, t_pared, PERCENTIL_SECCION))

    s_arco = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(centros, axis=0), axis=1))])
    radios = radios_de_curvatura(centros, separacion_radio_mm)
    r_valid = radios[np.isfinite(radios)]
    # el radio depende de cuanto se suavice la linea central: se reporta tambien el rango (suavizado fino / grueso)
    rango_r = []
    for ventana in (5, 15):
        rv = radios_de_curvatura(_suavizar(crudos, ventana), separacion_radio_mm)
        rv = rv[np.isfinite(rv)]
        if len(rv):
            rango_r.append(round(float(rv.min()), 1))

    def perfil(i):
        m = secciones[i]
        if m is None:
            return None
        return {"s_mm": round(float(s_arco[i]), 1), "centro_mm": [round(float(v), 1) for v in m["centro"]],
                "ancho_int_mm": round(m["ancho_int_mm"], 1), "alto_int_mm": round(m["alto_int_mm"], 1),
                "ancho_ext_mm": round(m["ancho_ext_mm"], 1), "alto_ext_mm": round(m["alto_ext_mm"], 1),
                "area_int_mm2": round(m["area_int_mm2"], 0), "diam_hidraulico_mm": round(m["diam_hidraulico_mm"], 1)}

    def puerto(sentido):
        """Puerto en un extremo. Se mide entre 20 y 80 mm hacia adentro (la brida y el borde distorsionan el final);
        la normal sale de la direccion de la linea central en ese tramo. sentido 0 = inicio, 1 = final."""
        if sentido == 0:
            sa, sb = s_arco[0] + 20.0, s_arco[0] + 80.0
        else:
            sa, sb = s_arco[-1] - 20.0, s_arco[-1] - 80.0
        ia, ib = int(np.argmin(np.abs(s_arco - sa))), int(np.argmin(np.abs(s_arco - sb)))
        normal = centros[ia] - centros[ib]  # de adentro hacia el extremo = hacia afuera del ducto
        normal = normal / max(np.linalg.norm(normal), 1e-12)
        tramo = [f for f in (perfil(i) for i in range(min(ia, ib), max(ia, ib) + 1)) if f]
        med = lambda k: round(float(np.median([f[k] for f in tramo])), 1) if tramo else None
        return {"punto_mm": [round(float(v), 1) for v in centros[ia]],
                "normal_hacia_afuera": [round(float(v), 3) for v in normal],
                "seccion_20_a_80mm": {"ancho_int_mm": med("ancho_int_mm"), "alto_int_mm": med("alto_int_mm"),
                                      "area_int_mm2": med("area_int_mm2"), "diam_hidraulico_mm": med("diam_hidraulico_mm")}}

    filas = [perfil(i) for i in range(len(centros))]
    filas_ok = [f for f in filas if f]
    areas = np.array([f["area_int_mm2"] for f in filas_ok])
    # se ignoran los 60 mm de cada extremo (bridas, collares, orejas) para las estadisticas del cuerpo
    cuerpo = [f for f in filas_ok if 60.0 <= f["s_mm"] <= s_arco[-1] - 60.0]
    a_cuerpo = np.array([f["area_int_mm2"] for f in cuerpo]) if cuerpo else areas
    aspectos = np.array([f["ancho_int_mm"] / max(f["alto_int_mm"], 1e-9) for f in cuerpo]) if cuerpo else np.array([np.nan])
    # cambio relativo de area por cada 50 mm
    cambio50 = []
    for f in cuerpo:
        g = [h for h in cuerpo if abs(h["s_mm"] - (f["s_mm"] + 50.0)) <= paso_s / 2]
        if g:
            cambio50.append(abs(g[0]["area_int_mm2"] - f["area_int_mm2"]) / max(f["area_int_mm2"], 1.0))

    # R / D_h en las curvas: un codo cerrado con seccion grande pierde mas presion que uno suave con seccion chica.
    # D_h = mediana en +-25 mm alrededor de cada punto (el D_h de un solo corte lo mueven las orejas).
    s_fil = np.array([f["s_mm"] for f in filas_ok])
    dh_fil = np.array([f["diam_hidraulico_mm"] for f in filas_ok])
    r_sobre_dh = []
    for i, r_i in enumerate(radios):
        if not np.isfinite(r_i) or not (60.0 <= s_arco[i] <= s_arco[-1] - 60.0):
            continue
        cerca = dh_fil[np.abs(s_fil - s_arco[i]) <= 25.0]
        if len(cerca):
            r_sobre_dh.append(r_i / max(float(np.median(cerca)), 1e-9))

    ficha = {
        "n_triangulos": int(len(tri)), "paso_mm": paso_mm, "espesor_pared_mm": round(t_pared, 2),
        "extremos_mm": [[round(float(v), 1) for v in P[a]], [round(float(v), 1) for v in P[b]]],
        "longitud_linea_central_mm": round(float(s_arco[-1]), 1),
        "radio_curvatura_minimo_mm": round(float(r_valid.min()), 1) if len(r_valid) else None,
        "radio_curvatura_minimo_rango_mm": rango_r,
        "radio_curvatura_p10_mm": round(float(np.percentile(r_valid, 10)), 1) if len(r_valid) else None,
        "area_int_mm2": {"min": round(float(a_cuerpo.min()), 0), "mediana": round(float(np.median(a_cuerpo)), 0),
                         "max": round(float(a_cuerpo.max()), 0)},
        "relacion_aspecto": {"mediana": round(float(np.nanmedian(aspectos)), 2), "max": round(float(np.nanmax(aspectos)), 2)},
        "cambio_max_de_area_por_50mm": round(float(max(cambio50)), 3) if cambio50 else None,
        # el maximo lo domina UN corte malo (orejas, brida); el percentil 95 es el numero para comparar ductos
        "cambio_area_por_50mm_p95": round(float(np.percentile(cambio50, 95)), 3) if cambio50 else None,
        "radio_sobre_diam_hidraulico_min": round(float(min(r_sobre_dh)), 2) if r_sobre_dh else None,
        "puertos": {"extremo_1": puerto(0), "extremo_2": puerto(1)},
        "perfil": filas_ok,
        "advertencias": advertencias,
        "_interno": {"centros": centros, "radios": radios, "s_arco": s_arco},
    }
    return ficha
