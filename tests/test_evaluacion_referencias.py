"""
Pruebas del protocolo de evaluacion con ductos de referencia (src/herramientas/evaluacion_referencias.py).
Se usa un ducto sintetico como 'referencia' y la malla numpy (sin CadQuery) como 'generador'.
Ejecutar desde la raiz del proyecto:  pytest tests/test_evaluacion_referencias.py -v
"""
import copy
import math

import pytest

from cad import ducto as d
from herramientas import evaluacion_referencias as ev
from herramientas import ficha_ducto as fd
from herramientas import ingesta_malla as im
from test_ficha_ducto import PARED, tubo_rectangular


@pytest.fixture(scope="module")
def ficha_ref():
    return fd.ficha_de_malla(tubo_rectangular(ancho=lambda u: 120.0 - 40.0 * u, alto=lambda u: 50.0),
                             paso_mm=3.0, espesor_pared_mm=PARED)


def test_el_escenario_toma_extremos_y_secciones_exteriores_de_la_ficha(ficha_ref):
    esc = ev.escenario_desde_ficha(ficha_ref, "tubo")
    p1 = ficha_ref["puertos"]["extremo_1"]
    sec = p1["seccion_20_a_80mm"]
    # el extremo real esta 20 mm mas afuera que el punto medido, siguiendo la normal
    esperado = [c + 20.0 * n for c, n in zip(p1["punto_mm"], p1["normal_hacia_afuera"])]
    assert esc["inicio"] == pytest.approx(esperado, abs=0.1)
    assert esc["ancho_mm"] == pytest.approx(sec["ancho_int_mm"] + 2 * PARED, abs=0.1)
    assert esc["alto_mm"] == pytest.approx(sec["alto_int_mm"] + 2 * PARED, abs=0.1)
    assert esc["espesor_mm"] == PARED and esc["avisos"] == []


def test_un_puerto_con_area_enorme_se_marca_como_dudoso_y_se_puede_corregir_con_la_medida_del_cad(ficha_ref):
    mala = copy.deepcopy(ficha_ref)
    mala["puertos"]["extremo_2"]["seccion_20_a_80mm"].update(ancho_int_mm=200.0, alto_int_mm=75.0, area_int_mm2=15000.0)
    esc = ev.escenario_desde_ficha(mala, "tubo")
    assert any("Puerto 2" in a and "orejas" in a for a in esc["avisos"])
    arreglado = ev.escenario_desde_ficha(mala, "tubo", {2: (80.0, 40.0)})
    assert arreglado["avisos"] == [] and arreglado["puerto_2"]["seccion_corregida_con_cad"] is True
    assert arreglado["ancho_fin_mm"] == pytest.approx(80.0 + 2 * PARED) and arreglado["alto_fin_mm"] == pytest.approx(40.0 + 2 * PARED)


def test_los_errores_de_puertos_son_cero_contra_si_mismo_aunque_los_extremos_vengan_al_reves(ficha_ref):
    al_reves = dict(ficha_ref)
    al_reves["puertos"] = {"extremo_1": ficha_ref["puertos"]["extremo_2"], "extremo_2": ficha_ref["puertos"]["extremo_1"]}
    for e in ev.errores_de_puertos(al_reves, ficha_ref):
        assert e["distancia_mm"] == pytest.approx(0.0, abs=1e-6) and e["angulo_normal_grados"] == pytest.approx(0.0, abs=1e-3)


def test_el_protocolo_completo_sin_cadquery_separa_lo_comparable_de_lo_que_depende_del_entorno(ficha_ref, tmp_path):
    esc = ev.escenario_desde_ficha(ficha_ref, "tubo")

    def generar(e, params):
        curva = d.suavizar_ruta([e["inicio"], e["fin"]], paso_mm=5.0)
        tri = d.malla_numpy(curva, e["ancho_mm"], e["alto_mm"], e["espesor_mm"], max_estaciones=60,
                            ancho_fin_mm=e["ancho_fin_mm"], alto_fin_mm=e["alto_fin_mm"])
        ruta = tmp_path / "ducto.stl"
        im.escribir_stl(ruta, tri)
        return ruta

    medir = lambda stl: fd.ficha_de_malla(im.leer_stl(stl), paso_mm=3.0, espesor_pared_mm=esc["espesor_mm"])  # noqa: E731
    res = ev.evaluar_escenario(esc, ficha_ref, {"area_min_mm2": 2000, "area_max_mm2": 9000, "aspect_ratio_max": 4.0}, generar, medir)

    assert set(res["comparable"]) == set(ev.COMPARABLES)
    assert set(res["no_comparable_sin_entorno"]) == set(ev.DEPENDEN_DEL_ENTORNO)
    assert "longitud_mm" not in res["comparable"]            # el largo de la recta NO se compara con el de la referencia
    assert res["cumplimiento"]["area_minima"]["cumple"] is True and len(res["puertos"]) == 2
    # los puertos de la referencia tienen areas ~5000 y ~3000 mm2: la generada debe reproducirlos
    assert res["comparable"]["area_puerto_1_mm2"]["razon"] == pytest.approx(1.0, rel=0.25)
    assert res["comparable"]["area_puerto_2_mm2"]["razon"] == pytest.approx(1.0, rel=0.25)
    assert "_ficha" in res and "_ficha" not in ev.a_json(res)


def test_carpetas_de_referencia_solo_lista_las_que_tienen_ficha(tmp_path):
    (tmp_path / "a").mkdir(); (tmp_path / "a" / "ficha.json").write_text("{}")
    (tmp_path / "b").mkdir()
    assert [p.name for p in ev.carpetas_de_referencia(tmp_path)] == ["a"]


# ---- perfil por estaciones -------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ficha_plana():
    """Referencia con puertos grandes (~130 x 70) y cuerpo plano (~60 x 40): dos secciones no pueden reproducirla."""
    import math
    return fd.ficha_de_malla(
        tubo_rectangular(ancho=lambda u: 130.0 - 70.0 * math.sin(math.pi * u), alto=lambda u: 70.0 - 30.0 * math.sin(math.pi * u)),
        paso_mm=3.0, espesor_pared_mm=PARED)


def _generador(tmp_path):
    def generar(e, params):
        curva = d.suavizar_ruta([e["inicio"], e["fin"]], paso_mm=5.0)
        if e.get("secciones"):
            tri = d.malla_numpy(curva, None, None, e["espesor_mm"], max_estaciones=60, secciones=e["secciones"])
        else:
            tri = d.malla_numpy(curva, e["ancho_mm"], e["alto_mm"], e["espesor_mm"], max_estaciones=60,
                                ancho_fin_mm=e["ancho_fin_mm"], alto_fin_mm=e["alto_fin_mm"])
        ruta = tmp_path / "ducto.stl"
        im.escribir_stl(ruta, tri)
        return ruta
    return generar


def test_las_estaciones_del_cuerpo_salen_ordenadas_dentro_del_ducto_y_con_el_cuerpo_plano(ficha_plana):
    est = ev.estaciones_de_cuerpo(ficha_plana)
    largo = ficha_plana["longitud_linea_central_mm"]
    assert 3 <= len(est) <= ev.N_ESTACIONES
    pos = [p for p, _, _ in est]
    assert pos == sorted(pos) and len(set(pos)) == len(pos)
    assert pos[0] > ev.MARGEN_EXTREMO_MM / largo - 1e-6 and pos[-1] < 1 - ev.MARGEN_EXTREMO_MM / largo + 1e-6
    centro = min(est, key=lambda e: abs(e[0] - 0.5))
    assert centro[1] == pytest.approx(60.0 - 2 * PARED, abs=8.0) and centro[2] == pytest.approx(40.0 - 2 * PARED, abs=8.0)


def test_el_escenario_trae_secciones_exteriores_de_0_a_1_y_se_puede_desactivar(ficha_plana):
    esc = ev.escenario_desde_ficha(ficha_plana, "plana")
    sec = esc["secciones"]
    assert sec[0][0] == 0.0 and sec[-1][0] == 1.0 and len(sec) >= 5
    assert sec[0][1:] == (esc["ancho_mm"], esc["alto_mm"]) and sec[-1][1:] == (esc["ancho_fin_mm"], esc["alto_fin_mm"])
    d.validar_secciones_control(sec)                      # lo que arma el escenario es aceptado por el generador
    assert ev.escenario_desde_ficha(ficha_plana, "plana", con_estaciones=False)["secciones"] is None


def test_un_ducto_demasiado_corto_no_tiene_estaciones_y_avisa(ficha_plana):
    corta = copy.deepcopy(ficha_plana)
    corta["longitud_linea_central_mm"] = 150.0
    esc = ev.escenario_desde_ficha(corta, "corta")
    assert esc["secciones"] is None and any("Sin estaciones" in a for a in esc["avisos"])


def test_el_error_de_perfil_es_cero_contra_si_mismo_aunque_los_extremos_vengan_al_reves(ficha_plana):
    from herramientas import metricas_ducto as mt
    assert mt.error_perfil(ficha_plana, ficha_plana)["error_area_mediana"] == pytest.approx(0.0, abs=1e-9)
    al_reves = dict(ficha_plana)
    n = len(ficha_plana["perfil"])
    largo = ficha_plana["longitud_linea_central_mm"]
    al_reves["perfil"] = [{**q, "s_mm": largo - q["s_mm"]} for q in reversed(ficha_plana["perfil"])]
    al_reves["puertos"] = {"extremo_1": ficha_plana["puertos"]["extremo_2"], "extremo_2": ficha_plana["puertos"]["extremo_1"]}
    assert len(al_reves["perfil"]) == n
    e = mt.error_perfil(al_reves, ficha_plana)
    assert e["error_area_mediana"] == pytest.approx(0.0, abs=1e-6) and e["error_aspecto_p90"] == pytest.approx(0.0, abs=1e-6)
    assert mt.error_perfil({**ficha_plana, "perfil": []}, ficha_plana) is None


def test_con_estaciones_el_ducto_generado_reproduce_el_perfil_y_con_dos_secciones_no(ficha_plana, tmp_path):
    params = {}
    medir = lambda stl: fd.ficha_de_malla(im.leer_stl(stl), paso_mm=3.0, espesor_pared_mm=PARED)  # noqa: E731
    con = ev.evaluar_escenario(ev.escenario_desde_ficha(ficha_plana, "plana"), ficha_plana, params, _generador(tmp_path), medir)
    sin = ev.evaluar_escenario(ev.escenario_desde_ficha(ficha_plana, "plana", con_estaciones=False), ficha_plana, params,
                               _generador(tmp_path), medir)
    assert con["modo_seccion"] == "estaciones" and sin["modo_seccion"] == "dos_secciones"
    assert con["perfil"]["error_area_mediana"] < 0.20
    assert sin["perfil"]["error_area_mediana"] > 0.60                      # 2 secciones: el cuerpo plano sale casi tan ancho como los puertos
    assert con["perfil"]["error_area_mediana"] < 0.5 * sin["perfil"]["error_area_mediana"]
    assert con["perfil"]["area_min_generada_sobre_referencia"] == pytest.approx(1.0, abs=0.25)


# ---- tramos rectos en los puertos (evaluacion) ----------------------------------------------------------------------
def test_el_escenario_trae_tramos_rectos_por_la_normal_con_largo_min_y_tope(ficha_plana):
    esc = ev.escenario_desde_ficha(ficha_plana, "plana")
    pd = esc["puertos_ducto"]
    for clave, puerto in (("inicio", esc["puerto_1"]), ("fin", esc["puerto_2"])):
        assert pd[clave]["normal"] == puerto["normal_hacia_afuera"]
        assert ev.TRAMO_MIN_MM <= pd[clave]["tramo_mm"] <= ev.FRACCION_MAX_TRAMO * math.dist(esc["inicio"], esc["fin"]) + 1e-6
        assert pd[clave]["seccion_constante_mm"] == min(pd[clave]["tramo_mm"], ev.SECCION_CONSTANTE_MM)
    assert esc["rectos_mm"] == (pd["inicio"]["seccion_constante_mm"], pd["fin"]["seccion_constante_mm"])
    # factor_tramo=None -> como antes
    assert ev.escenario_desde_ficha(ficha_plana, "plana", factor_tramo=None)["puertos_ducto"] is None
    # un factor enorme se recorta con aviso
    grande = ev.escenario_desde_ficha(ficha_plana, "plana", factor_tramo=50.0)
    assert any("se reduce a" in a for a in grande["avisos"])
    assert grande["puertos_ducto"]["inicio"]["tramo_mm"] == pytest.approx(
        ev.FRACCION_MAX_TRAMO * math.dist(grande["inicio"], grande["fin"]), abs=0.1)


def test_las_estaciones_se_miden_entre_los_tramos_de_seccion_constante(ficha_plana):
    sin = ev.estaciones_de_cuerpo(ficha_plana)
    con = ev.estaciones_de_cuerpo(ficha_plana, rectos_mm=(80.0, 80.0))
    assert len(sin) == len(con) and all(0.0 < p < 1.0 for p, _, _ in con)
    # mismas ventanas (el margen ya era 80 mm), pero pos se reescala al tramo libre: se aleja de los extremos menos
    assert con[0][0] < sin[0][0] and con[-1][0] > sin[-1][0]
    d.validar_secciones_control([(0.0, 1.0, 1.0)] + [(p, a, h) for p, a, h in con] + [(1.0, 1.0, 1.0)])


def test_con_tramos_rectos_el_ducto_generado_sale_por_la_normal_y_reproduce_el_perfil(ficha_plana, tmp_path):
    from cad import curva_ruta as cr

    def generar(e, params):
        t = cr.tramos_de_puertos(e["inicio"], e["fin"], e["puertos_ducto"])
        curva, _ = cr.redondear_esquinas([e["inicio"], t["inicio_astar"], t["fin_astar"], e["fin"]], 47.0, 5.0)
        tri = d.malla_numpy(curva, None, None, e["espesor_mm"], max_estaciones=80, secciones=e["secciones"],
                            rectos_mm=t["rectos_mm"])
        ruta = tmp_path / "ducto_t.stl"
        im.escribir_stl(ruta, tri)
        return ruta

    medir = lambda stl: fd.ficha_de_malla(im.leer_stl(stl), paso_mm=3.0, espesor_pared_mm=PARED)  # noqa: E731
    res = ev.evaluar_escenario(ev.escenario_desde_ficha(ficha_plana, "plana"), ficha_plana, {}, generar, medir)
    assert res["tramos_rectos_mm"] is not None
    assert all(p["angulo_normal_grados"] < 12.0 for p in res["puertos"])
    assert all(p["distancia_mm"] < 25.0 for p in res["puertos"])
    assert res["perfil"]["error_area_mediana"] < 0.25


def test_la_referencia_corregida_reemplaza_solo_el_puerto_con_orejas_y_no_toca_el_original(ficha_ref):
    mala = copy.deepcopy(ficha_ref)
    mala["puertos"]["extremo_2"]["seccion_20_a_80mm"].update(ancho_int_mm=200.0, alto_int_mm=75.0, area_int_mm2=15000.0)
    esc = ev.escenario_desde_ficha(mala, "tubo", {2: (80.0, 40.0)})
    ref, corregidos = ev.referencia_corregida(mala, esc)
    sec = ref["puertos"]["extremo_2"]["seccion_20_a_80mm"]
    assert corregidos == [2] and sec["area_int_mm2"] == pytest.approx(3200.0)
    assert sec["diam_hidraulico_mm"] == pytest.approx(2 * 80 * 40 / 120.0, abs=0.1)
    assert mala["puertos"]["extremo_2"]["seccion_20_a_80mm"]["area_int_mm2"] == 15000.0      # el original no cambia
    assert ref["puertos"]["extremo_1"] == mala["puertos"]["extremo_1"]
    assert ev.referencia_corregida(mala, ev.escenario_desde_ficha(mala, "tubo"))[1] == []


# ---- linea central de la referencia como curva dada ----------------------------------------------------------------------
def test_la_curva_de_referencia_sale_por_la_normal_sigue_la_linea_central_y_cumple_r_min(ficha_ref):
    import numpy as np
    from cad import ducto as dd
    esc = ev.escenario_desde_ficha(ficha_ref, "tubo", ruta_referencia=True, r_min_mm=45.0)
    curva = np.array(esc["curva"])
    assert np.allclose(curva[0], esc["inicio"], atol=0.01) and np.allclose(curva[-1], esc["fin"], atol=0.01)
    # arranca por la normal del puerto: los primeros 60 mm son una recta hacia adentro
    n = np.array(esc["puertos_ducto"]["inicio"]["normal"], dtype=float)
    n /= np.linalg.norm(n)
    primeros = curva[np.linalg.norm(curva - curva[0], axis=1) < 60.0]
    assert np.allclose((primeros - curva[0]) @ n, -np.linalg.norm(primeros - curva[0], axis=1), atol=0.5)
    assert dd.radios_de_curvatura(curva).min() >= 45.0 - 1e-6
    assert dd.longitud_mm(curva) == pytest.approx(ficha_ref["longitud_linea_central_mm"], rel=0.08)
    # sin ruta_referencia no hay curva
    assert ev.escenario_desde_ficha(ficha_ref, "tubo")["curva"] is None


def test_el_alisado_no_encoge_los_extremos_ni_cambia_una_recta():
    import numpy as np
    recta = np.column_stack([np.arange(0, 300, 5.0), np.zeros(60), np.zeros(60)])
    assert np.allclose(ev._alisar(recta, 20.0, 5.0), recta, atol=1e-9)
    ruidosa = recta.copy()
    ruidosa[1:-1, 1] += np.random.default_rng(0).normal(0, 3.0, 58)
    liso = ev._alisar(ruidosa, 20.0, 5.0)
    assert np.allclose(liso[[0, -1]], ruidosa[[0, -1]]) and liso[:, 1].std() < 0.4 * ruidosa[:, 1].std()


def test_con_la_curva_de_la_referencia_el_largo_y_el_radio_si_se_parecen(ficha_ref, tmp_path):
    import numpy as np
    esc = ev.escenario_desde_ficha(ficha_ref, "tubo", ruta_referencia=True, r_min_mm=45.0)

    def generar(e, params):
        tri = d.malla_numpy(np.array(e["curva"]), None, None, e["espesor_mm"], max_estaciones=80, secciones=e["secciones"],
                            rectos_mm=e["rectos_mm"])
        ruta = tmp_path / "ducto_rr.stl"
        im.escribir_stl(ruta, tri)
        return ruta

    medir = lambda stl: fd.ficha_de_malla(im.leer_stl(stl), paso_mm=3.0, espesor_pared_mm=PARED)  # noqa: E731
    res = ev.evaluar_escenario(esc, ficha_ref, {}, generar, medir)
    assert res["ruta_de_referencia"] is True
    assert res["no_comparable_sin_entorno"]["longitud_mm"]["razon"] == pytest.approx(1.0, abs=0.08)
    radio = res["no_comparable_sin_entorno"]["radio_curvatura_minimo_mm"]
    assert radio["razon"] == pytest.approx(1.0, abs=0.35)                    # el tubo de prueba tiene un codo de R = 150
    assert all(p["distancia_mm"] < 25.0 and p["angulo_normal_grados"] < 12.0 for p in res["puertos"])


# --- puertos OBLICUOS: la linea central medida no sale por la normal del puerto ------------------------------------------
def _ficha_oblicua(largo_mm=900.0, pendiente=0.84, n=181):
    """Ficha minima a mano: la linea central sale del puerto ~40 grados fuera de la normal y llega recta al otro puerto
    (y(x) = a x (1 - x/L)^2). Los dos puertos tienen su normal sobre el eje x."""
    import numpy as np
    x = np.linspace(0.0, largo_mm, n)
    y = pendiente * x * (1.0 - x / largo_mm) ** 2
    centros = np.column_stack([x, y, np.zeros(n)])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(centros, axis=0), axis=1))])
    return {
        "longitud_linea_central_mm": float(s[-1]),
        "perfil": [{"s_mm": float(si), "centro_mm": c.tolist()} for si, c in zip(s, centros)],
        "puertos": {"extremo_1": {"punto_mm": [0.0, 0.0, 0.0]}, "extremo_2": {"punto_mm": [largo_mm, 0.0, 0.0]}},
    }


def _escenario_oblicuo(largo_mm=900.0, tramo=100.0):
    return {"inicio": [0.0, 0.0, 0.0], "fin": [largo_mm, 0.0, 0.0],
            "puertos_ducto": {"inicio": {"normal": [-1.0, 0.0, 0.0], "tramo_mm": tramo},
                              "fin": {"normal": [1.0, 0.0, 0.0], "tramo_mm": tramo}}}


def test_el_empalme_es_tangente_al_tramo_recto_y_al_cuerpo_y_no_hace_quiebres():
    import numpy as np
    recto = np.array([[-100.0 + 5.0 * i, 0.0, 0.0] for i in range(21)])                  # termina en el origen, rumbo +x
    ang = np.radians(45.0)
    cuerpo = np.array([[5.0 * i * math.cos(ang), 5.0 * i * math.sin(ang), 0.0] for i in range(60)])
    c = ev._remuestrear(ev._empalmar(recto, cuerpo, 160.0, 5.0), 5.0)   # un giro de 45 grados pide un empalme largo
    t = np.diff(c, axis=0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    giro = np.degrees(np.arccos(np.clip((t[1:] * t[:-1]).sum(axis=1), -1, 1)))
    assert giro.max() < 7.0                                  # sin quiebre: el cuerpo tiene 45 grados de diferencia
    assert np.allclose(c[0], recto[0]) and np.allclose(c[-1], cuerpo[-1])
    assert np.allclose(t[0], [1, 0, 0], atol=1e-6)           # sale por la direccion del tramo recto
    assert d.radios_de_curvatura(c).min() > 45.0
    corto = ev._remuestrear(ev._empalmar(recto, cuerpo, 30.0, 5.0), 5.0)           # uno corto SI queda cerrado: por eso hay que buscar el largo
    assert d.radios_de_curvatura(corto).min() < 20.0


def test_con_puertos_oblicuos_la_curva_de_referencia_no_tiene_quiebres_y_cumple_r_min():
    """Regresion: antes los tramos rectos se pegaban al cuerpo con un quiebre de 30-48 grados en un paso de 5 mm
    (radio local ~5 mm): el bucle de alisado nunca llegaba a r_min y no avisaba."""
    import numpy as np
    ficha, esc = _ficha_oblicua(), _escenario_oblicuo()
    curva = np.array(ev.curva_de_referencia(ficha, esc, r_min_mm=45.0))
    r = d.validar_radio_minimo(curva, 45.0)
    assert r["cumple"] and r["radio_minimo_mm"] >= 45.0
    t = np.diff(curva, axis=0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    assert np.degrees(np.arccos(np.clip((t[1:] * t[:-1]).sum(axis=1), -1, 1))).max() < 10.0   # ningun quiebre
    # extremos fijos y arranque por la normal invertida (el ducto entra: +x en el inicio, -x en el fin)
    assert np.allclose(curva[0], esc["inicio"], atol=0.01) and np.allclose(curva[-1], esc["fin"], atol=0.01)
    assert np.allclose(t[0], [1, 0, 0], atol=1e-3) and np.allclose(t[-1], [1, 0, 0], atol=1e-3)
    assert np.abs(curva[np.linalg.norm(curva - curva[0], axis=1) < 90.0][:, 1]).max() < 0.5      # el tramo recto sigue recto
    assert d.longitud_mm(curva) == pytest.approx(ficha["longitud_linea_central_mm"], rel=0.06)


def test_si_la_curva_de_referencia_no_puede_cumplir_r_min_lo_dice_en_vez_de_devolver_un_quiebre():
    with pytest.raises(ValueError, match="No se logro una curva de referencia"):
        ev.curva_de_referencia(_ficha_oblicua(), _escenario_oblicuo(), r_min_mm=5000.0)


def test_el_error_de_puerto_se_separa_en_axial_y_lateral():
    ref = {"puertos": {"extremo_1": {"punto_mm": [0.0, 0.0, 0.0], "normal_hacia_afuera": [-1.0, 0.0, 0.0]},
                       "extremo_2": {"punto_mm": [900.0, 0.0, 0.0], "normal_hacia_afuera": [1.0, 0.0, 0.0]}}}
    gen = {"puertos": {"extremo_1": {"punto_mm": [17.0, 3.0, 4.0], "normal_hacia_afuera": [-1.0, 0.0, 0.0]},
                       "extremo_2": {"punto_mm": [900.0, 0.0, 0.0], "normal_hacia_afuera": [1.0, 0.0, 0.0]}}}
    e1, e2 = ev.errores_de_puertos(gen, ref)
    assert e1["axial_mm"] == pytest.approx(-17.0, abs=0.1) and e1["lateral_mm"] == pytest.approx(5.0, abs=0.1)
    assert e1["distancia_mm"] == pytest.approx(math.hypot(17.0, 5.0), abs=0.1)
    assert e2["axial_mm"] == pytest.approx(0.0, abs=1e-6) and e2["lateral_mm"] == pytest.approx(0.0, abs=1e-6)
