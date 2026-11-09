"""
Metricas para juzgar un ducto (el que genera la IA) y para compararlo con un ducto de REFERENCIA.
Todo sale de la ficha tecnica (herramientas/ficha_ducto.py), que mide una MALLA: asi el ducto generado y el de
referencia se miden con exactamente las mismas reglas, y se mide lo que se exporto (STEP/STL), no lo que se pidio.

Dos preguntas distintas, que no hay que mezclar:
  1. CUMPLIMIENTO  ->  cumplimiento(ficha, parametros): el ducto respeta lo que el humano aprobo (R_min, area de
     paso, relacion de aspecto)? Es aprobado / no aprobado / dudoso, nunca "casi".
  2. PARECIDO      ->  comparar(generada, referencia): se comporta como el ducto de referencia (largo, radios,
     seccion, cambio de area)? Son razones generada/referencia, no un puntaje: no existe una "nota" sin un criterio
     de GM que diga cuanta diferencia es aceptable.

Incertidumbre (importante para el panel): la ficha mide con voxeles de ~3 mm. El radio de curvatura minimo depende del
suavizado (la ficha da un rango) y puede variar ~20 %; las dimensiones de seccion ~3 mm. Por eso cumplimiento() marca
"dudoso" cuando el valor medido queda a menos de la incertidumbre del limite, en vez de aprobar o rechazar en falso.

No incluye: perdida de presion (eso es CFD o un modelo analitico que no hemos construido), angulo de desmoldeo
(falta definir la direccion de desmoldeo, decision de manufactura).
"""
from __future__ import annotations

import math

INCERTIDUMBRE_RADIO = 0.20     # el radio minimo medido puede variar ~20 % con el suavizado de la linea central
INCERTIDUMBRE_AREA = 0.10      # el area interior medida con voxeles de 3 mm: ~10 %
INCERTIDUMBRE_ASPECTO = 0.10


def resumen(ficha: dict) -> dict:
    """Las cifras que se comparan, en un dict plano (sale de ficha_de_malla o de un ficha.json)."""
    p = ficha["puertos"]
    return {
        "longitud_mm": ficha["longitud_linea_central_mm"],
        "radio_curvatura_minimo_mm": ficha["radio_curvatura_minimo_mm"],
        "radio_sobre_diam_hidraulico_min": ficha.get("radio_sobre_diam_hidraulico_min"),
        "area_cuerpo_mediana_mm2": ficha["area_int_mm2"]["mediana"],
        "area_cuerpo_min_mm2": ficha["area_int_mm2"]["min"],
        "area_cuerpo_max_mm2": ficha["area_int_mm2"]["max"],
        "area_puerto_1_mm2": p["extremo_1"]["seccion_20_a_80mm"]["area_int_mm2"],
        "area_puerto_2_mm2": p["extremo_2"]["seccion_20_a_80mm"]["area_int_mm2"],
        "relacion_aspecto_mediana": ficha["relacion_aspecto"]["mediana"],
        "cambio_area_por_50mm_p95": ficha.get("cambio_area_por_50mm_p95"),
        "cambio_max_de_area_por_50mm": ficha["cambio_max_de_area_por_50mm"],
    }


def _tres_estados(medido, limite, minimo: bool, incertidumbre: float):
    """True / False / 'dudoso'. minimo=True: debe ser >= limite; False: debe ser <= limite."""
    if medido is None or limite is None:
        return "no_verificado"
    margen = abs(limite) * incertidumbre
    if minimo:
        return True if medido >= limite + margen else (False if medido < limite - margen else "dudoso")
    return True if medido <= limite - margen else (False if medido > limite + margen else "dudoso")


def cumplimiento(ficha: dict, parametros: dict) -> dict:
    """Compara lo MEDIDO en la malla con los parametros aprobados. Cada chequeo: valor medido, limite, resultado.
    Falta un parametro -> 'no_verificado' (nunca se aprueba en silencio)."""
    r_min = parametros.get("r_min_mm")
    amin, amax = parametros.get("area_min_mm2"), parametros.get("area_max_mm2")
    asp = parametros.get("aspect_ratio_max")
    rango = ficha.get("radio_curvatura_minimo_rango_mm") or []
    radio = min([ficha["radio_curvatura_minimo_mm"], *rango]) if ficha.get("radio_curvatura_minimo_mm") else None
    a = ficha["area_int_mm2"]
    # el radio de una ruta recta no existe: sin curvas el radio minimo no limita
    radio_res = "no_aplica_sin_curvas" if radio is None and r_min is not None else \
        _tres_estados(radio, r_min, True, INCERTIDUMBRE_RADIO)
    return {
        "radio_minimo": {"medido_mm": radio, "limite_mm": r_min, "cumple": radio_res},
        "area_minima": {"medido_mm2": a["min"], "limite_mm2": amin, "cumple": _tres_estados(a["min"], amin, True, INCERTIDUMBRE_AREA)},
        "area_maxima": {"medido_mm2": a["max"], "limite_mm2": amax, "cumple": _tres_estados(a["max"], amax, False, INCERTIDUMBRE_AREA)},
        "relacion_aspecto": {"medido": ficha["relacion_aspecto"]["max"], "limite": asp,
                             "cumple": _tres_estados(ficha["relacion_aspecto"]["max"], asp, False, INCERTIDUMBRE_ASPECTO)},
        "nota": "Medido en la malla con voxeles de %s mm; 'dudoso' = dentro de la incertidumbre de la medicion." % ficha.get("paso_mm"),
    }


def comparar(generada: dict, referencia: dict) -> dict:
    """Metrica por metrica: valor de la referencia, de la generada y la razon generada/referencia.
    Una razon cerca de 1 = se parece. No se resume en un solo numero a proposito."""
    g, r = resumen(generada), resumen(referencia)
    # el orden extremo_1/extremo_2 de la ficha es arbitrario: se emparejan los puertos por cercania
    pg, pr = generada["puertos"], referencia["puertos"]
    dist = lambda a, b: math.dist(pg[a]["punto_mm"], pr[b]["punto_mm"])  # noqa: E731
    if dist("extremo_1", "extremo_2") + dist("extremo_2", "extremo_1") < dist("extremo_1", "extremo_1") + dist("extremo_2", "extremo_2"):
        g["area_puerto_1_mm2"], g["area_puerto_2_mm2"] = g["area_puerto_2_mm2"], g["area_puerto_1_mm2"]
    salida = {}
    for k in g:
        vg, vr = g[k], r[k]
        razon = None if vg is None or vr in (None, 0) else round(vg / vr, 3)
        salida[k] = {"referencia": vr, "generada": vg, "razon": razon}
    return salida


def _curva_de_perfil(ficha: dict, invertir: bool):
    """(u, area_int, aspecto) del perfil de la ficha, con u = s / largo en [0, 1] (invertido si hace falta)."""
    largo = float(ficha["longitud_linea_central_mm"])
    q = ficha.get("perfil") or []
    u = [p["s_mm"] / largo for p in q]
    area = [p["area_int_mm2"] for p in q]
    aspecto = [max(p["ancho_int_mm"], p["alto_int_mm"]) / max(min(p["ancho_int_mm"], p["alto_int_mm"]), 1e-9) for p in q]
    if invertir:
        u, area, aspecto = [1.0 - x for x in reversed(u)], area[::-1], aspecto[::-1]
    return u, area, aspecto


def error_perfil(generada: dict, referencia: dict, u_min: float = 0.1, u_max: float = 0.9, n: int = 200) -> dict | None:
    """Que tanto se parece el perfil de area (y de aspecto) de la generada al de la referencia a lo largo del ducto.
    Ambos perfiles se llevan a distancia relativa u = s / largo y se comparan en [u_min, u_max] (se dejan fuera los
    extremos, donde la ficha mide mal: boca abierta, orejas). Error relativo = |generada - referencia| / referencia.
    Se reporta mediana y percentil 90. Ojo: la ficha tiene ~10 % de incertidumbre en area, y el perfil de la
    referencia trae saltos sueltos de la medicion; errores de ese orden no se distinguen del ruido.
    Devuelve None si alguna ficha no tiene perfil."""
    import numpy as np

    if not generada.get("perfil") or not referencia.get("perfil"):
        return None
    pg, pr = generada["puertos"], referencia["puertos"]
    dist = lambda a, b: math.dist(pg[a]["punto_mm"], pr[b]["punto_mm"])  # noqa: E731
    invertir = dist("extremo_1", "extremo_2") + dist("extremo_2", "extremo_1") < dist("extremo_1", "extremo_1") + dist("extremo_2", "extremo_2")
    ug, ag, sg = _curva_de_perfil(generada, invertir)
    ur, ar, sr = _curva_de_perfil(referencia, False)
    u = np.linspace(u_min, u_max, n)
    ag_u, ar_u = np.interp(u, ug, ag), np.interp(u, ur, ar)
    sg_u, sr_u = np.interp(u, ug, sg), np.interp(u, ur, sr)
    e_area = np.abs(ag_u - ar_u) / np.maximum(ar_u, 1e-9)
    e_asp = np.abs(sg_u - sr_u) / np.maximum(sr_u, 1e-9)
    return {
        "u_desde_hasta": [u_min, u_max],
        "error_area_mediana": round(float(np.median(e_area)), 3), "error_area_p90": round(float(np.percentile(e_area, 90)), 3),
        "error_aspecto_mediana": round(float(np.median(e_asp)), 3), "error_aspecto_p90": round(float(np.percentile(e_asp, 90)), 3),
        "area_min_generada_sobre_referencia": round(float(ag_u.min() / max(ar_u.min(), 1e-9)), 3),
    }


def angulo_grados(u, v) -> float:
    """Angulo entre dos vectores 3D, en grados."""
    nu, nv = math.sqrt(sum(x * x for x in u)), math.sqrt(sum(x * x for x in v))
    c = sum(a * b for a, b in zip(u, v)) / max(nu * nv, 1e-12)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))
