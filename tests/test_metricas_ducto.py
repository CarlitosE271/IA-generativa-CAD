"""
Pruebas de src/herramientas/metricas_ducto.py (cumplimiento y comparacion contra un ducto de referencia).
Se usan ductos sinteticos de medidas conocidas (el generador de tests/test_ficha_ducto.py).
Ejecutar desde la raiz del proyecto:  pytest tests/test_metricas_ducto.py -v
"""
import pytest

from herramientas import ficha_ducto as fd
from herramientas import metricas_ducto as mt
from test_ficha_ducto import PARED, tubo_rectangular


@pytest.fixture(scope="module")
def ficha_100x50():
    return fd.ficha_de_malla(tubo_rectangular(), paso_mm=3.0, espesor_pared_mm=PARED)


@pytest.fixture(scope="module")
def ficha_60x30():
    return fd.ficha_de_malla(tubo_rectangular(ancho=lambda u: 60.0, alto=lambda u: 30.0), paso_mm=3.0,
                             espesor_pared_mm=PARED)


def test_resumen_trae_las_cifras_de_comparacion(ficha_100x50):
    r = mt.resumen(ficha_100x50)
    assert r["longitud_mm"] == pytest.approx(835.6, rel=0.05)
    assert r["area_cuerpo_mediana_mm2"] == pytest.approx(96 * 46, rel=0.15)
    assert r["relacion_aspecto_mediana"] == pytest.approx(96 / 46, rel=0.2)
    assert r["cambio_area_por_50mm_p95"] is not None and r["cambio_area_por_50mm_p95"] < 0.15
    assert r["radio_sobre_diam_hidraulico_min"] == pytest.approx(150 / (4 * 96 * 46 / (2 * (96 + 46))), rel=0.35)


def test_cumple_cuando_los_limites_estan_holgados(ficha_100x50):
    c = mt.cumplimiento(ficha_100x50, {"r_min_mm": 100, "area_min_mm2": 3000, "area_max_mm2": 6000, "aspect_ratio_max": 3.0})
    assert [c[k]["cumple"] for k in ("radio_minimo", "area_minima", "area_maxima", "relacion_aspecto")] == [True] * 4


def test_no_cumple_cuando_se_viola_con_claridad(ficha_100x50):
    c = mt.cumplimiento(ficha_100x50, {"r_min_mm": 300, "area_min_mm2": 9000, "area_max_mm2": 3000, "aspect_ratio_max": 1.2})
    assert [c[k]["cumple"] for k in ("radio_minimo", "area_minima", "area_maxima", "relacion_aspecto")] == [False] * 4


def test_dudoso_cuando_el_limite_cae_dentro_de_la_incertidumbre_de_la_medicion(ficha_100x50):
    # radio medido ~143 mm: un limite de 140 no se puede afirmar ni negar con una medicion con ~20 % de incertidumbre
    assert mt.cumplimiento(ficha_100x50, {"r_min_mm": 140})["radio_minimo"]["cumple"] == "dudoso"
    area_max = ficha_100x50["area_int_mm2"]["max"]
    assert mt.cumplimiento(ficha_100x50, {"area_max_mm2": area_max * 1.02})["area_maxima"]["cumple"] == "dudoso"


def test_sin_parametros_no_aprueba_en_silencio(ficha_100x50):
    c = mt.cumplimiento(ficha_100x50, {})
    assert all(c[k]["cumple"] == "no_verificado" for k in ("radio_minimo", "area_minima", "area_maxima", "relacion_aspecto"))


def test_comparar_un_ducto_consigo_mismo_da_razon_1_y_con_otro_se_nota(ficha_100x50, ficha_60x30):
    iguales = mt.comparar(ficha_100x50, ficha_100x50)
    assert all(v["razon"] == pytest.approx(1.0) for v in iguales.values() if v["razon"] is not None)
    c = mt.comparar(ficha_60x30, ficha_100x50)
    assert c["area_cuerpo_mediana_mm2"]["razon"] == pytest.approx((56 * 26) / (96 * 46), rel=0.2)  # ~0.33
    assert c["longitud_mm"]["razon"] == pytest.approx(1.0, rel=0.05)  # misma ruta


def test_angulo_entre_vectores():
    assert mt.angulo_grados([1, 0, 0], [0, 1, 0]) == pytest.approx(90.0)
    assert mt.angulo_grados([1, 0, 0], [2, 0, 0]) == pytest.approx(0.0, abs=1e-6)
    assert mt.angulo_grados([1, 0, 0], [-1, 0, 0]) == pytest.approx(180.0)


def test_comparar_empareja_los_puertos_por_cercania_aunque_la_ficha_los_numere_al_reves(ficha_100x50):
    al_reves = dict(ficha_100x50)
    al_reves["puertos"] = {"extremo_1": ficha_100x50["puertos"]["extremo_2"], "extremo_2": ficha_100x50["puertos"]["extremo_1"]}
    c = mt.comparar(al_reves, ficha_100x50)
    assert c["area_puerto_1_mm2"]["razon"] == pytest.approx(1.0) and c["area_puerto_2_mm2"]["razon"] == pytest.approx(1.0)
