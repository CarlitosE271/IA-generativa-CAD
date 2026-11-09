"""
Protocolo de evaluacion con los ductos de REFERENCIA (las 3 piezas de Maikol), SIN entorno.

Que se hace por cada referencia
  1. Se mide la referencia con la ficha tecnica (herramientas/ficha_ducto.py)  ->  data/referencia/<pieza>/ficha.json
  2. Se arma un ESCENARIO: puerto inicial y final (posicion y seccion) tomados de esa ficha, entorno vacio.
  3. Se genera el ducto con el pipeline (A* -> curva -> CAD con transicion de seccion).
  4. Se mide el ducto GENERADO con la misma ficha (sobre la malla exportada) y se juzga:
        - cumplimiento de los parametros aprobados (aprobado / no aprobado / dudoso)
        - parecido en lo que SI es comparable sin entorno: area de paso, aspecto, suavidad del area
        - ajuste a los puertos: distancia al puerto de la referencia y angulo de salida vs la normal del puerto

Que NO se compara, y por que (decirlo al panel)
  - Largo, radio de curvatura y R/D_h: dependen de los obstaculos que rodea el ducto, y no conocemos el entorno de
    ninguna referencia. Sin obstaculos el A* da una recta (Fer: "sin bloques es linea recta"); que la recta sea mas
    corta que la referencia NO la hace mejor. Estas cifras se reportan, pero marcadas como no comparables.
  - Este protocolo evalua la generacion del CAD (transicion de seccion, cumplimiento, ajuste a puertos). NO evalua la
    inteligencia de ruteo (A* + campo de costo): eso necesita escenas con obstaculos (ver scripts/crear_escena_ejemplo.py).

Perfil por estaciones (modo por defecto)
  Con solo las secciones de los dos puertos, un cuerpo mas plano o mas chico que AMBOS puertos no se puede reproducir
  (segunda pieza: cuerpo ~117 x 34 mm, puertos de 4100 y 7200 mm2). Por eso el escenario toma ademas ~10 secciones del
  cuerpo de la referencia (mediana por ventana del perfil de la ficha) y el ducto generado las interpola con PCHIP.
  CUIDADO CON LA CIRCULARIDAD: el perfil sale de la MISMA referencia contra la que se compara; el resultado prueba que
  el generador sabe reproducir un perfil dado (y que la ficha mide de forma consistente), NO que la IA lo "invente".
  Con --solo-puertos se recupera el modo anterior (dos secciones) para comparar los dos.

Limitaciones conocidas
  - Con tramos rectos en los puertos (por defecto) el ducto sale en la direccion de la normal; con --sin-tramos el
    pipeline no la impone y el angulo de salida solo se MIDE (puede ser grande). Los tramos miden 1.5 x D_h del puerto
    (minimo 85 mm, tope 30 % de la distancia entre puertos): ese largo es una eleccion de diseno, no un dato de GM;
    el minimo de 85 mm esta puesto para cubrir la ventana 20-80 mm con que la ficha mide el puerto.
  - La orientacion de la seccion alrededor del eje (que lado es el ancho) no se impone: se usa el ancho/alto de la ficha.
  - Los puertos de la ficha salen de una medicion con voxeles de 3 mm; si una referencia tiene orejas o bridas en el
    extremo (la primera pieza), la seccion del puerto no es fiable y se marca para corregirla con la medida del CAD.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

from herramientas import metricas_ducto as mt

N_ESTACIONES = 10               # secciones del cuerpo que se copian de la referencia
MARGEN_EXTREMO_MM = 80.0        # no se toman estaciones en los primeros/ultimos 80 mm: ahi esta el puerto (y las orejas)
VENTANA_MIN_MM = 30.0           # una estacion necesita al menos esta longitud de cuerpo para ser mediana de algo
SANGRIA_PUERTO_MM = 20.0        # la ficha mide el puerto 20 mm DENTRO del extremo
FACTOR_TRAMO = 1.5              # tramo recto de cada puerto = 1.5 x el diametro hidraulico interior del puerto...
TRAMO_MIN_MM = 85.0             # ...pero no menos de 85 mm: cubre la ventana 20-80 mm donde la ficha mide el puerto
SECCION_CONSTANTE_MM = 80.0     # la seccion del puerto solo vale ~80 mm: en la referencia la boca se estrecha enseguida
FRACCION_MAX_TRAMO = 0.3        # ni mas del 30 % de la distancia entre puertos (si no, los dos tramos se comerian el ducto)
FACTOR_PUERTO_DUDOSO = 2.5      # puerto con area > 2.5 x la mediana del cuerpo: casi seguro orejas o bridas
LARGOS_EMPALME_MM = (30.0, 45.0, 60.0, 80.0, 100.0, 130.0, 160.0, 200.0, 260.0)   # largo (mm) del empalme tangente tramo recto -> cuerpo

# comparables sin conocer el entorno / dependientes del entorno (ver docstring)
COMPARABLES = ("area_cuerpo_mediana_mm2", "area_cuerpo_min_mm2", "area_cuerpo_max_mm2", "area_puerto_1_mm2",
               "area_puerto_2_mm2", "relacion_aspecto_mediana", "cambio_area_por_50mm_p95",
               "cambio_max_de_area_por_50mm")
DEPENDEN_DEL_ENTORNO = ("longitud_mm", "radio_curvatura_minimo_mm", "radio_sobre_diam_hidraulico_min")


def _mediana(valores: list[float]) -> float:
    v = sorted(valores)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def _perfil_orientado(ficha: dict) -> list[dict]:
    """Perfil de la ficha empezando en el extremo_1 (el sentido de la linea central de la ficha puede venir al reves)."""
    perfil = ficha.get("perfil") or []
    if not perfil:
        return []
    largo = float(ficha.get("longitud_linea_central_mm") or 0.0)
    p1, p2 = ficha["puertos"]["extremo_1"]["punto_mm"], ficha["puertos"]["extremo_2"]["punto_mm"]
    if math.dist(perfil[0]["centro_mm"], p1) > math.dist(perfil[0]["centro_mm"], p2):
        return [{**q, "s_mm": largo - q["s_mm"]} for q in reversed(perfil)]
    return perfil


def _alisar(puntos, sigma_mm: float, paso_mm: float):
    """Suavizado gaussiano de una polilinea 3D muestreada cada paso_mm, con los dos extremos fijos."""
    import numpy as np

    p = np.asarray(puntos, dtype=float)
    k = max(1, int(round(sigma_mm / paso_mm)))
    x = np.arange(-3 * k, 3 * k + 1)
    w = np.exp(-0.5 * (x / k) ** 2)
    w /= w.sum()
    pad = np.vstack([2 * p[0] - p[1:3 * k + 1][::-1], p, 2 * p[-1] - p[-3 * k - 1:-1][::-1]])   # reflejo: no encoge los extremos
    liso = np.column_stack([np.convolve(pad[:, c], w, mode="same") for c in range(3)])[3 * k:3 * k + len(p)]
    liso[0], liso[-1] = p[0], p[-1]
    return liso


def _empalmar(tramo, cuerpo, largo_mm: float, paso_mm: float):
    """Une un tramo recto con el cuerpo de la curva con un empalme TANGENTE (cubica de Hermite).
    tramo: puntos desde el exterior hasta la union (su ultimo punto == cuerpo[0]); cuerpo: curva que sigue.
    El empalme sale con la direccion del tramo recto y llega al cuerpo, `largo_mm` de arco adelante, con la tangente
    del cuerpo: no hay quiebre en la union (un quiebre es un radio local de ~paso_mm, que viola cualquier R_min)."""
    import numpy as np

    tramo, cuerpo = np.asarray(tramo, dtype=float), np.asarray(cuerpo, dtype=float)
    a = tramo[-1]
    dir_a = a - tramo[0]
    if np.linalg.norm(dir_a) < 1e-6:                      # sin tramo recto: no hay direccion que respetar
        return np.vstack([tramo[:-1], cuerpo])
    dir_a /= np.linalg.norm(dir_a)
    acum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(cuerpo, axis=0), axis=1))])
    j = int(np.searchsorted(acum, largo_mm))
    j = min(max(j, 2), len(cuerpo) - 2)
    b = cuerpo[j]
    dir_b = cuerpo[j + 1] - cuerpo[j - 1]
    dir_b /= np.linalg.norm(dir_b)
    m = float(np.linalg.norm(b - a))
    k = max(4, int(np.ceil(m / paso_mm)) * 2)
    u = np.linspace(0.0, 1.0, k + 1)[:, None]
    h = ((2 * u**3 - 3 * u**2 + 1) * a + (u**3 - 2 * u**2 + u) * m * dir_a
         + (-2 * u**3 + 3 * u**2) * b + (u**3 - u**2) * m * dir_b)
    return np.vstack([tramo[:-1], h[:-1], cuerpo[j:]])


def _remuestrear(curva, paso_mm: float):
    """Misma curva con puntos equiespaciados en longitud de arco (el radio de curvatura de 3 puntos lo necesita)."""
    import numpy as np

    c = np.asarray(curva, dtype=float)
    acum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    n = max(8, int(np.ceil(acum[-1] / paso_mm)) + 1)
    u = np.linspace(0.0, acum[-1], n)
    return np.column_stack([np.interp(u, acum, c[:, i]) for i in range(3)])



def curva_de_referencia(ficha: dict, escenario: dict, r_min_mm: float | None = None, paso_mm: float = 5.0,
                        margen_mm: float = 15.0) -> list[list[float]]:
    """La linea central MEDIDA de la referencia como CURVA ya hecha (no como ruta con esquinas): inicio -> tramo recto
    por la normal -> linea central alisada -> tramo recto -> fin, muestreada cada paso_mm.
    La linea medida trae ruido (picos de 10 mm en las orejas), asi que se alisa con una gaussiana cada vez mas ancha
    (sigma 10, 15, 20, 30, 40, 60 mm) hasta que el radio de curvatura minimo sea >= r_min_mm (si se da).
    Por que curva y no ruta: una ruta con esquinas obliga a redondearlas con arcos de r_min, y la referencia tiene
    curvas suaves y seguidas que no caben asi (el pipeline contesta 'infactible: 9 esquinas no caben').
    Es circular a proposito: la curva es de la referencia; lo que se evalua es la fidelidad del CAD, no el ruteo."""
    import numpy as np
    from cad import curva_ruta as cr
    from cad import ducto as dd

    perfil = _perfil_orientado(ficha)
    if len(perfil) < 3:
        raise ValueError("La ficha no trae perfil para sacar la linea central.")
    centros = np.array([q["centro_mm"] for q in perfil], dtype=float)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(centros, axis=0), axis=1))])
    pd = escenario.get("puertos_ducto") or {}
    t = cr.tramos_de_puertos(escenario["inicio"], escenario["fin"], pd)
    ini_a, fin_a = np.array(t["inicio_astar"]), np.array(t["fin_astar"])
    l0 = float((pd.get("inicio") or {}).get("tramo_mm") or 0.0)
    l1 = float((pd.get("fin") or {}).get("tramo_mm") or 0.0)
    medio = centros[(s > l0 + margen_mm) & (s < s[-1] - l1 - margen_mm)]
    cruda = np.vstack([ini_a, medio, fin_a])
    acum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(cruda, axis=0), axis=1))])
    n = max(8, int(np.ceil(acum[-1] / paso_mm)) + 1)
    u = np.linspace(0.0, acum[-1], n)
    remuestreada = np.column_stack([np.interp(u, acum, cruda[:, c]) for c in range(3)])

    def recta(a, b):
        k = max(1, int(np.ceil(np.linalg.norm(b - a) / paso_mm)))
        return np.array([a + (b - a) * q / k for q in range(k + 1)])

    inicio, fin = np.array(escenario["inicio"], dtype=float), np.array(escenario["fin"], dtype=float)
    mitad = float(acum[-1]) / 2.0 - paso_mm
    empalmes = [v for v in LARGOS_EMPALME_MM if v <= mitad] or [max(mitad, 3 * paso_mm)]
    curva, radio_logrado = None, None
    for sigma in (10.0, 15.0, 20.0, 30.0, 40.0, 60.0):
        liso = _alisar(remuestreada, sigma, paso_mm)
        for largo in (empalmes[:1] if r_min_mm is None else empalmes):
            c1 = _empalmar(recta(inicio, ini_a), liso, largo, paso_mm)               # inicio -> recto -> empalme -> cuerpo
            c2 = _empalmar(recta(fin, fin_a), c1[::-1], largo, paso_mm)[::-1]       # lo mismo por el otro extremo
            candidata = _remuestrear(c2, paso_mm)
            radios = dd.radios_de_curvatura(candidata)
            minimo = float(np.min(radios)) if len(radios) else float("inf")
            if curva is None or minimo > radio_logrado:
                curva, radio_logrado = candidata, minimo
            if r_min_mm is None or minimo >= r_min_mm:
                return [[round(float(c), 2) for c in p] for p in candidata]
    raise ValueError(f"No se logro una curva de referencia con radio >= {r_min_mm} mm ni alisando (sigma hasta 60 mm) "
                     f"ni empalmando los tramos rectos (hasta {empalmes[-1]:.0f} mm): el mejor radio fue "
                     f"{radio_logrado:.1f} mm. Los tramos rectos de los puertos no son compatibles con la linea central medida.")


def estaciones_de_cuerpo(ficha: dict, n_max: int = N_ESTACIONES,
                        rectos_mm: tuple[float, float] = (0.0, 0.0)) -> list[tuple[float, float, float]]:
    """Secciones INTERIORES del cuerpo de la referencia: [(pos, ancho_int, alto_int), ...] con pos en (0, 1).
    Cada una es la mediana del perfil de la ficha dentro de una ventana (las medianas ignoran los saltos sueltos de la
    medicion). Sin estaciones (ducto demasiado corto o sin perfil) devuelve [].
    rectos_mm = (l0, l1): tramos rectos del ducto generado, donde la seccion es constante. Las ventanas empiezan despues
    de ellos y pos se mide entre ellos: pos = (s - l0) / (largo - l0 - l1), igual que perfil_de_seccion(rectos_mm=...).
    Sin tramos, pos = s / largo."""
    perfil = ficha.get("perfil") or []
    largo = float(ficha.get("longitud_linea_central_mm") or 0.0)
    l0, l1 = rectos_mm
    m0, m1 = max(MARGEN_EXTREMO_MM, l0), max(MARGEN_EXTREMO_MM, l1)
    util = largo - m0 - m1
    if not perfil or util < VENTANA_MIN_MM or largo - l0 - l1 <= 0:
        return []
    perfil = _perfil_orientado(ficha)   # empieza en el extremo_1
    n = max(1, min(n_max, int(util // VENTANA_MIN_MM)))
    ancho_ventana = util / n
    salida = []
    for k in range(n):
        a, b = m0 + k * ancho_ventana, m0 + (k + 1) * ancho_ventana
        dentro = [q for q in perfil if a <= q["s_mm"] < b]
        if not dentro:
            continue
        # ancho y AREA medianos; el alto sale de ellos. Con medianas separadas de ancho y alto, una ventana donde el ancho
        # baja y el alto sube da un area que la referencia nunca tuvo (en la segunda pieza: 7900 en vez de 4500 mm2).
        ancho_m, area_m = _mediana([q["ancho_int_mm"] for q in dentro]), _mediana([q["area_int_mm2"] for q in dentro])
        salida.append((round((0.5 * (a + b) - l0) / (largo - l0 - l1), 4), round(ancho_m, 1), round(area_m / ancho_m, 1)))
    return salida


def escenario_desde_ficha(ficha: dict, nombre: str, secciones_interiores: dict | None = None,
                          con_estaciones: bool = True, factor_tramo: float | None = FACTOR_TRAMO,
                          ruta_referencia: bool = False, r_min_mm: float | None = None) -> dict:
    """Escenario sin entorno a partir de la ficha de una referencia.
    inicio/fin = el EXTREMO real de cada puerto (el punto de la ficha esta 20 mm adentro, se lleva 20 mm hacia afuera
    por la normal). ancho/alto = medida EXTERIOR (interior de la ficha + 2 x espesor).
    secciones_interiores: {1: (ancho_int, alto_int), 2: (...)} corrige la seccion de un puerto con la medida del CAD.
    con_estaciones=True agrega "secciones": [(pos, ancho_ext, alto_ext), ...] = puerto 1 (pos 0), estaciones del cuerpo
    de la referencia y puerto 2 (pos 1); es lo que usa el generador en el modo por estaciones. ancho_mm/alto_mm/fin
    siguen describiendo los puertos (modo de dos secciones). Sin estaciones utiles, "secciones" queda en None.
    factor_tramo (None o 0 = sin tramos): cada puerto lleva un tramo recto a lo largo de su normal de
    max(factor_tramo x D_h interior del puerto, TRAMO_MIN_MM) mm, con tope de FRACCION_MAX_TRAMO x distancia entre
    puertos. ruta_referencia=True agrega "curva" = la linea central medida de la referencia (ver curva_de_referencia;
    r_min_mm: el alisado sube hasta que el radio minimo sea >= r_min_mm).
    Sale en "puertos_ducto" (para ducto_desde_entorno) y "rectos_mm" (= los mm de SECCION constante, que son
    min(tramo, SECCION_CONSTANTE_MM): el tramo recto fija la direccion, la seccion constante solo cubre la boca)."""
    t = float(ficha["espesor_pared_mm"])
    mediana = ficha["area_int_mm2"]["mediana"]
    puertos, avisos = [], []
    for numero, clave in ((1, "extremo_1"), (2, "extremo_2")):
        p = ficha["puertos"][clave]
        sec = p["seccion_20_a_80mm"]
        ancho_int, alto_int = sec["ancho_int_mm"], sec["alto_int_mm"]
        corregido = bool(secciones_interiores and numero in secciones_interiores)
        if corregido:
            ancho_int, alto_int = secciones_interiores[numero]
        elif sec["area_int_mm2"] > FACTOR_PUERTO_DUDOSO * mediana:
            avisos.append(f"Puerto {numero}: area {sec['area_int_mm2']:.0f} mm2 contra {mediana:.0f} mm2 del cuerpo; "
                          "probablemente son orejas o bridas del extremo. Confirma con la medida del CAD "
                          "(--seccion NOMBRE:PUERTO=ANCHOxALTO) antes de fiarte.")
        puertos.append({
            "punto_medido_mm": p["punto_mm"], "normal_hacia_afuera": p["normal_hacia_afuera"],
            "extremo_mm": [round(c + SANGRIA_PUERTO_MM * n, 1) for c, n in zip(p["punto_mm"], p["normal_hacia_afuera"])],
            "ancho_ext_mm": round(ancho_int + 2 * t, 1), "alto_ext_mm": round(alto_int + 2 * t, 1),
            "seccion_corregida_con_cad": corregido,
        })
    puertos_ducto, rectos = None, (0.0, 0.0)
    if factor_tramo:
        distancia = math.dist(puertos[0]["extremo_mm"], puertos[1]["extremo_mm"])
        largos = []
        for numero, pto in enumerate(puertos, start=1):
            a_, h_ = pto["ancho_ext_mm"] - 2 * t, pto["alto_ext_mm"] - 2 * t
            dh = 2.0 * a_ * h_ / (a_ + h_)
            deseado = max(factor_tramo * dh, TRAMO_MIN_MM)
            tope = FRACCION_MAX_TRAMO * distancia
            if deseado > tope:
                avisos.append(f"Puerto {numero}: el tramo recto pedido ({deseado:.0f} mm) pasa del {FRACCION_MAX_TRAMO:.0%} de la "
                              f"distancia entre puertos; se reduce a {tope:.0f} mm.")
            largos.append(round(min(deseado, tope), 1))
        const = [min(l, SECCION_CONSTANTE_MM) for l in largos]
        puertos_ducto = {"inicio": {"normal": puertos[0]["normal_hacia_afuera"], "tramo_mm": largos[0], "seccion_constante_mm": const[0]},
                         "fin": {"normal": puertos[1]["normal_hacia_afuera"], "tramo_mm": largos[1], "seccion_constante_mm": const[1]}}
        rectos = (const[0], const[1])
    secciones = None
    if con_estaciones:
        cuerpo = estaciones_de_cuerpo(ficha, rectos_mm=rectos)
        if cuerpo:
            secciones = ([(0.0, puertos[0]["ancho_ext_mm"], puertos[0]["alto_ext_mm"])]
                         + [(pos, round(a + 2 * t, 1), round(h + 2 * t, 1)) for pos, a, h in cuerpo]
                         + [(1.0, puertos[1]["ancho_ext_mm"], puertos[1]["alto_ext_mm"])])
        else:
            avisos.append("Sin estaciones de cuerpo (ducto muy corto o ficha sin perfil): se usan solo las dos secciones de los puertos.")
    escenario = {"nombre": nombre, "espesor_mm": t, "puerto_1": puertos[0], "puerto_2": puertos[1], "avisos": avisos,
            "inicio": puertos[0]["extremo_mm"], "fin": puertos[1]["extremo_mm"],
            "ancho_mm": puertos[0]["ancho_ext_mm"], "alto_mm": puertos[0]["alto_ext_mm"],
            "ancho_fin_mm": puertos[1]["ancho_ext_mm"], "alto_fin_mm": puertos[1]["alto_ext_mm"],
            "secciones": secciones, "puertos_ducto": puertos_ducto, "rectos_mm": rectos if puertos_ducto else None,
            "tramos_rectos_geometricos_mm": [puertos_ducto["inicio"]["tramo_mm"], puertos_ducto["fin"]["tramo_mm"]] if puertos_ducto else None,
            "curva": None}
    if ruta_referencia:
        escenario["curva"] = curva_de_referencia(ficha, escenario, r_min_mm)
    return escenario


def errores_de_puertos(generada: dict, referencia: dict) -> list[dict]:
    """Por puerto (emparejados por cercania): distancia entre el punto medido en la generada y en la referencia, y
    angulo entre sus normales. Mide el AJUSTE a los puertos y la limitacion de que la normal no se impone.
    La distancia se separa en AXIAL (a lo largo de la normal de la referencia) y LATERAL (perpendicular). El punto de
    puerto de la ficha es "20 mm de arco hacia adentro desde donde la linea central marchada empieza", y ese inicio
    varia ~15 mm segun donde el marchado deja de poder medir la seccion: la componente axial tiene esa holgura del
    instrumento; la lateral y el angulo no."""
    pg, pr = generada["puertos"], referencia["puertos"]
    d = lambda a, b: math.dist(pg[a]["punto_mm"], pr[b]["punto_mm"])  # noqa: E731
    pares = [("extremo_1", "extremo_1"), ("extremo_2", "extremo_2")]
    if d("extremo_1", "extremo_2") + d("extremo_2", "extremo_1") < d("extremo_1", "extremo_1") + d("extremo_2", "extremo_2"):
        pares = [("extremo_1", "extremo_2"), ("extremo_2", "extremo_1")]
    salida = []
    for a, b in pares:
        v = [x - y for x, y in zip(pg[a]["punto_mm"], pr[b]["punto_mm"])]
        n = pr[b]["normal_hacia_afuera"]
        nn = math.sqrt(sum(c * c for c in n)) or 1.0
        axial = sum(c * k for c, k in zip(v, n)) / nn                                  # a lo largo de la normal de la referencia
        lateral = math.sqrt(max(sum(c * c for c in v) - axial * axial, 0.0))           # perpendicular a ella
        salida.append({
            "puerto_referencia": b, "distancia_mm": round(d(a, b), 1),
            "axial_mm": round(axial, 1), "lateral_mm": round(lateral, 1),
            "angulo_normal_grados": round(mt.angulo_grados(pg[a]["normal_hacia_afuera"], pr[b]["normal_hacia_afuera"]), 1),
        })
    return salida


def referencia_corregida(ficha_referencia: dict, escenario: dict) -> tuple[dict, list[int]]:
    """Copia de la ficha de la referencia donde el area/seccion de un puerto corregido con la medida del CAD (--seccion)
    reemplaza a la medida con orejas. Si no, comparar la generada (sin orejas) contra el 15490 mm2 de la ficha da una
    razon de 0.31 que parece un defecto y no lo es. Solo se corrige el puerto; las estadisticas del cuerpo (area
    maxima, cambio maximo de area) siguen contaminadas por las orejas y NO se pueden corregir desde aqui."""
    ref, corregidos = copy.deepcopy(ficha_referencia), []
    t = escenario["espesor_mm"]
    for numero, clave in ((1, "extremo_1"), (2, "extremo_2")):
        pto = escenario[f"puerto_{numero}"]
        if not pto.get("seccion_corregida_con_cad"):
            continue
        a, h = pto["ancho_ext_mm"] - 2 * t, pto["alto_ext_mm"] - 2 * t
        ref["puertos"][clave]["seccion_20_a_80mm"].update(
            ancho_int_mm=round(a, 1), alto_int_mm=round(h, 1), area_int_mm2=round(a * h, 1),
            diam_hidraulico_mm=round(2 * a * h / (a + h), 1))
        corregidos.append(numero)
    return ref, corregidos


def evaluar_escenario(escenario: dict, ficha_referencia: dict, parametros: dict, generar, medir) -> dict:
    """generar(escenario, parametros) -> ruta del STL del ducto generado;  medir(ruta_stl) -> ficha del ducto generado.
    Se inyectan para poder probar el protocolo sin CadQuery."""
    stl = generar(escenario, parametros)
    ficha = medir(stl)
    ficha_referencia, corregidos = referencia_corregida(ficha_referencia, escenario)
    comp = mt.comparar(ficha, ficha_referencia)
    return {
        "escenario": escenario["nombre"],
        "stl": str(stl),
        "cumplimiento": mt.cumplimiento(ficha, parametros),
        "comparable": {k: comp[k] for k in COMPARABLES},
        "no_comparable_sin_entorno": {k: comp[k] for k in DEPENDEN_DEL_ENTORNO},
        "puertos": errores_de_puertos(ficha, ficha_referencia),
        "perfil": mt.error_perfil(ficha, ficha_referencia),
        "puertos_de_referencia_corregidos_con_cad": corregidos,
        "ruta_de_referencia": bool(escenario.get("curva")),
        "tramos_rectos_mm": escenario.get("tramos_rectos_geometricos_mm"),
        "seccion_constante_mm": list(escenario["rectos_mm"]) if escenario.get("rectos_mm") else None,
        "modo_seccion": "estaciones" if escenario.get("secciones") else "dos_secciones",
        "advertencias": [*escenario["avisos"], *ficha["advertencias"]],
        "_ficha": ficha,
    }


def a_json(resultado: dict) -> dict:
    return {k: v for k, v in resultado.items() if not k.startswith("_")}


def carpetas_de_referencia(raiz: Path) -> list[Path]:
    return sorted(p.parent for p in Path(raiz).glob("*/ficha.json"))
