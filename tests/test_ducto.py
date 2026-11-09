"""
Pruebas de src/cad/ducto.py.

La parte numerica (suavizado, radios de curvatura, marcos de transporte paralelo, rectangulos por
estacion, validaciones) se prueba con numpy REAL. CadQuery no se pudo instalar en este entorno de
trabajo, asi que construir_ducto()/exportar() se prueban con un modulo 'cadquery' falso que respeta la
API documentada (Vector, Wire.makePolygon, Solid.makeLoft, Shape.cut, Volume, isValid, exporters.export).

Ejecutar desde la raiz del proyecto:  pytest tests -v
"""
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from cad import ducto as d


def _arco_xy(radio, grados=90, n=200):
    """Puntos sobre un arco de circunferencia en el plano XY (z=0), de radio conocido."""
    ang = np.linspace(0, math.radians(grados), n)
    return np.column_stack([radio * np.sin(ang), radio * (1 - np.cos(ang)), np.zeros(n)])


# ---- limpieza de la ruta ---------------------------------------------------------------------
def test_ruta_invalida_se_rechaza():
    with pytest.raises(ValueError):
        d._como_puntos([[0, 0, 0]])  # un solo punto
    with pytest.raises(ValueError):
        d._como_puntos([[0, 0, 0], [0, 0, 0]])  # dos puntos iguales = un solo punto distinto
    with pytest.raises(ValueError):
        d._como_puntos([[0, 0], [1, 1]])  # falta z


def test_quita_puntos_consecutivos_repetidos():
    p = d._como_puntos([[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0]])
    assert len(p) == 3


# ---- suavizado ---------------------------------------------------------------------------------
def test_suavizar_conserva_inicio_y_fin_exactos():
    ruta = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [10, 10, 10]]
    curva = d.suavizar_ruta(ruta, paso_mm=2.0)
    assert np.allclose(curva[0], ruta[0])
    assert np.allclose(curva[-1], ruta[-1])


def test_suavizar_redondea_la_esquina_y_la_separacion_es_uniforme():
    ruta = [[0, 0, 0], [50, 0, 0], [50, 50, 0]]  # una esquina viva de 90 grados
    curva = d.suavizar_ruta(ruta, iteraciones=4, paso_mm=1.0)
    # la esquina viva (50,0,0) ya no esta en la curva: se cortó
    assert np.min(np.linalg.norm(curva - np.array([50, 0, 0]), axis=1)) > 1.0
    pasos = np.linalg.norm(np.diff(curva, axis=0), axis=1)
    assert pasos.max() - pasos.min() < 0.2 * pasos.mean()


def test_ruta_recta_de_dos_puntos_no_se_deforma():
    curva = d.suavizar_ruta([[0, 0, 0], [100, 0, 0]], paso_mm=10.0)
    assert np.allclose(curva[:, 1:], 0)
    assert d.longitud_mm(curva) == pytest.approx(100.0)


# ---- radio de curvatura -------------------------------------------------------------------------
def test_radio_de_un_arco_conocido():
    radios = d.radios_de_curvatura(_arco_xy(radio=80.0))
    assert np.allclose(radios, 80.0, rtol=1e-3)


def test_tramo_recto_tiene_radio_infinito():
    recta = np.column_stack([np.linspace(0, 100, 20), np.zeros(20), np.zeros(20)])
    assert np.all(np.isinf(d.radios_de_curvatura(recta)))


def test_validar_radio_minimo_detecta_violacion_y_cumplimiento():
    arco = _arco_xy(radio=80.0)
    ok = d.validar_radio_minimo(arco, r_min_mm=60.0)
    assert ok["cumple"] is True and ok["num_violaciones"] == 0
    mal = d.validar_radio_minimo(arco, r_min_mm=100.0)
    assert mal["cumple"] is False and mal["num_violaciones"] > 0
    assert mal["radio_minimo_mm"] == pytest.approx(80.0, rel=1e-2)


def test_validar_radio_en_una_recta_cumple_sin_radio_minimo():
    recta = np.column_stack([np.linspace(0, 100, 20), np.zeros(20), np.zeros(20)])
    r = d.validar_radio_minimo(recta, 50.0)
    assert r["cumple"] is True and r["radio_minimo_mm"] is None


# ---- seccion --------------------------------------------------------------------------------------
def test_diametro_hidraulico_rectangulo():
    assert d.diametro_hidraulico(60, 30) == pytest.approx(40.0)  # 2*60*30/90


def test_validar_seccion_contra_parametros():
    params = {"area_min_mm2": 310, "area_max_mm2": 900, "aspect_ratio_max": 3}
    ok = d.validar_seccion(30, 20, params)  # 600 mm2, aspecto 1.5
    assert ok["area_ok"] is True and ok["aspecto_ok"] is True
    mal = d.validar_seccion(100, 10, params)  # 1000 mm2, aspecto 10
    assert mal["area_ok"] is False and mal["aspecto_ok"] is False


def test_validar_seccion_no_aprueba_en_silencio_lo_que_no_puede_verificar():
    r = d.validar_seccion(30, 20, {})
    assert r["area_ok"] == "no_verificado" and r["aspecto_ok"] == "no_verificado"


# ---- marcos de transporte paralelo ------------------------------------------------------------------
def _ortonormales(t, n, b):
    for v in (t, n, b):
        assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-6)
    assert np.allclose(np.einsum("ij,ij->i", t, n), 0.0, atol=1e-6)
    assert np.allclose(np.einsum("ij,ij->i", t, b), 0.0, atol=1e-6)
    assert np.allclose(np.einsum("ij,ij->i", n, b), 0.0, atol=1e-6)


def test_marcos_en_curva_plana_mantienen_la_normal_vertical():
    # ruta en el plano XY con "arriba" = Z: el rectangulo no debe girar sobre su eje
    t, n, b = d.marcos_transporte_paralelo(_arco_xy(80.0))
    _ortonormales(t, n, b)
    assert np.allclose(n, [0, 0, 1], atol=1e-6)


def test_marcos_en_curva_3d_son_ortonormales_y_sin_saltos():
    s = np.linspace(0, 2 * math.pi, 150)
    helice = np.column_stack([60 * np.cos(s), 60 * np.sin(s), 15 * s])
    t, n, b = d.marcos_transporte_paralelo(helice)
    _ortonormales(t, n, b)
    saltos = np.linalg.norm(np.diff(n, axis=0), axis=1)
    assert saltos.max() < 0.2  # sin giros bruscos de un punto al siguiente (el perfil no se tuerce)


def test_marcos_cuando_el_ducto_arranca_vertical():
    vertical = np.column_stack([np.zeros(10), np.zeros(10), np.linspace(0, 90, 10)])
    t, n, b = d.marcos_transporte_paralelo(vertical)
    _ortonormales(t, n, b)  # no debe fallar aunque 'arriba' sea paralelo a la tangente


# ---- rectangulos por estacion --------------------------------------------------------------------------
def test_rectangulos_tienen_el_tamano_pedido_y_son_perpendiculares_a_la_ruta():
    pts = np.column_stack([np.linspace(0, 100, 11), np.zeros(11), np.zeros(11)])
    t, n, b = d.marcos_transporte_paralelo(pts)
    rects = d.estaciones_rectangulares(pts, t, n, b, ancho_mm=60.0, alto_mm=30.0)
    assert len(rects) == 11
    for r, tangente in zip(rects, t):
        lado_ancho = np.linalg.norm(r[0] - r[1])
        lado_alto = np.linalg.norm(r[1] - r[2])
        assert lado_ancho == pytest.approx(60.0) and lado_alto == pytest.approx(30.0)
        assert np.dot(r[0] - r[1], tangente) == pytest.approx(0.0, abs=1e-9)
        assert np.dot(r[1] - r[2], tangente) == pytest.approx(0.0, abs=1e-9)


def test_extension_agrega_una_estacion_de_cada_lado_en_linea_recta():
    pts = np.column_stack([np.linspace(0, 100, 11), np.zeros(11), np.zeros(11)])
    t, n, b = d.marcos_transporte_paralelo(pts)
    rects = d.estaciones_rectangulares(pts, t, n, b, 20.0, 10.0, extension_mm=5.0)
    assert len(rects) == 13
    assert rects[0].mean(axis=0)[0] == pytest.approx(-5.0)
    assert rects[-1].mean(axis=0)[0] == pytest.approx(105.0)


def test_submuestrear_respeta_el_maximo_y_conserva_extremos():
    pts = np.column_stack([np.linspace(0, 500, 400), np.zeros(400), np.zeros(400)])
    sub = d._submuestrear(pts, 40)
    assert len(sub) <= 40
    assert np.allclose(sub[0], pts[0]) and np.allclose(sub[-1], pts[-1])
    assert d._submuestrear(pts[:10], 40) is not None and len(d._submuestrear(pts[:10], 40)) == 10


# ---- nombre de salida -----------------------------------------------------------------------------------
def test_nombre_de_archivo_solo_acepta_caracteres_seguros():
    assert d._nombre_valido("ducto_1-a") == "ducto_1-a"
    for malo in ["", "../x", "a/b", "a\\b", "ducto.step", "a b", "x" * 61]:
        with pytest.raises(ValueError):
            d._nombre_valido(malo)


# ---- CadQuery simulado ----------------------------------------------------------------------------------
class _SolidoFalso:
    def __init__(self, nombre, n_perfiles=0, operandos=()):
        self.nombre, self.n_perfiles, self.operandos = nombre, n_perfiles, operandos

    def cut(self, otro):
        return _SolidoFalso("corte", operandos=(self, otro))

    def Volume(self):
        return 1234.5

    def isValid(self):
        return True


def _instalar_cadquery_falso(monkeypatch):
    registro = {"lofts": [], "exportados": []}

    def makePolygon(vectores):
        assert vectores[0] is vectores[-1] or vectores[0].coords == vectores[-1].coords  # alambre cerrado
        return types.SimpleNamespace(n=len(vectores))

    def makeLoft(alambres, ruled=False):
        s = _SolidoFalso("loft", n_perfiles=len(alambres))
        registro["lofts"].append(s)
        return s

    def export(shape, ruta):
        registro["exportados"].append(ruta)
        Path(ruta).write_text("CAD FALSO")

    class Vector:
        def __init__(self, x, y, z):
            self.coords = (x, y, z)

    modulo = types.ModuleType("cadquery")
    modulo.Vector = Vector
    modulo.Wire = types.SimpleNamespace(makePolygon=makePolygon)
    modulo.Solid = types.SimpleNamespace(makeLoft=makeLoft)
    modulo.exporters = types.SimpleNamespace(export=export)
    monkeypatch.setitem(sys.modules, "cadquery", modulo)
    return registro


def test_construir_ducto_hace_loft_exterior_e_interior_y_los_resta(monkeypatch):
    registro = _instalar_cadquery_falso(monkeypatch)
    pts = d.suavizar_ruta([[0, 0, 0], [100, 0, 0], [100, 100, 0]], paso_mm=5.0)

    resultado = d.construir_ducto(pts, ancho_mm=40.0, alto_mm=20.0, espesor_mm=2.0, max_estaciones=30)

    exterior, interior = registro["lofts"]
    assert resultado.nombre == "corte" and resultado.operandos == (exterior, interior)
    assert exterior.n_perfiles <= 30
    assert interior.n_perfiles == exterior.n_perfiles + 2  # estaciones extra para dejar los extremos abiertos


def test_construir_ducto_rechaza_espesor_imposible(monkeypatch):
    _instalar_cadquery_falso(monkeypatch)
    pts = d.suavizar_ruta([[0, 0, 0], [100, 0, 0]], paso_mm=5.0)
    for espesor in (0, -1, 10.0, 15.0):  # alto=20 -> la pared debe ser < 10
        with pytest.raises(ValueError):
            d.construir_ducto(pts, 40.0, 20.0, espesor)


def test_generar_ducto_devuelve_reporte_y_exporta_step_y_stl(tmp_path, monkeypatch):
    registro = _instalar_cadquery_falso(monkeypatch)
    monkeypatch.setattr(d, "ruta_cad", tmp_path)
    params = {"r_min_mm": 30, "area_min_mm2": 310, "area_max_mm2": 900, "aspect_ratio_max": 3}

    rep = d.generar_ducto(
        [[0, 0, 0], [100, 0, 0], [100, 100, 0], [100, 100, 60]],
        ancho_mm=30.0, alto_mm=20.0, espesor_mm=2.0, parametros=params, nombre="ducto_prueba",
    )

    assert rep["nombre"] == "ducto_prueba"
    assert rep["archivos"] == {"step": "ducto_prueba.step", "stl": "ducto_prueba.stl"}
    assert (tmp_path / "ducto_prueba.step").exists() and (tmp_path / "ducto_prueba.stl").exists()
    assert rep["seccion"]["area_ok"] is True and rep["seccion"]["aspecto_ok"] is True
    assert rep["radio_curvatura"]["r_min_exigido_mm"] == 30.0
    assert rep["volumen_mm3"] == 1234.5 and rep["solido_valido"] is True
    assert rep["longitud_mm"] > 0
    assert len(registro["exportados"]) == 2


def test_generar_ducto_sin_parametros_no_aprueba_lo_que_no_verifico(tmp_path, monkeypatch):
    _instalar_cadquery_falso(monkeypatch)
    monkeypatch.setattr(d, "ruta_cad", tmp_path)
    rep = d.generar_ducto([[0, 0, 0], [100, 0, 0]], 30.0, 20.0, 2.0, parametros=None, nombre="sin_params")
    assert rep["radio_curvatura"]["cumple"] == "no_verificado"
    assert rep["seccion"]["area_ok"] == "no_verificado"


def test_generar_ducto_reporta_violacion_de_r_min_pero_igual_genera(tmp_path, monkeypatch):
    _instalar_cadquery_falso(monkeypatch)
    monkeypatch.setattr(d, "ruta_cad", tmp_path)
    # esquina de 90 grados con r_min enorme: tiene que violarse, y el ducto se genera igual
    rep = d.generar_ducto(
        [[0, 0, 0], [60, 0, 0], [60, 60, 0]], 30.0, 20.0, 2.0,
        parametros={"r_min_mm": 500}, nombre="esquina",
    )
    assert rep["radio_curvatura"]["cumple"] is False
    assert rep["radio_curvatura"]["num_violaciones"] > 0
    assert "archivos" in rep


def test_generar_ducto_escala_la_ruta_si_viene_en_celdas(tmp_path, monkeypatch):
    _instalar_cadquery_falso(monkeypatch)
    monkeypatch.setattr(d, "ruta_cad", tmp_path)
    en_mm = d.generar_ducto([[0, 0, 0], [100, 0, 0]], 30.0, 20.0, 2.0, nombre="a", exportar_archivos=False)
    en_celdas = d.generar_ducto([[0, 0, 0], [10, 0, 0]], 30.0, 20.0, 2.0, nombre="b",
                                mm_por_celda=10.0, exportar_archivos=False)
    assert en_celdas["longitud_mm"] == pytest.approx(en_mm["longitud_mm"])


# ---- transicion de seccion ----------------------------------------------------------------------------------
def _recta(largo=300.0, n=61):
    return np.column_stack([np.linspace(0, largo, n), np.zeros(n), np.zeros(n)])


def test_las_leyes_de_transicion_empiezan_en_0_y_terminan_en_1_y_son_monotonas():
    u = np.linspace(0, 1, 101)
    for ley in d.LEYES_TRANSICION:
        f = d.fraccion_de_transicion(u, ley)
        assert f[0] == pytest.approx(0.0) and f[-1] == pytest.approx(1.0)
        assert np.all(np.diff(f) >= -1e-12)
    with pytest.raises(ValueError):
        d.fraccion_de_transicion(u, "cubica")


def test_la_ley_suave_no_tiene_pendiente_en_los_puertos_pero_su_pico_es_1_5_veces_el_de_la_lineal():
    u = np.linspace(0, 1, 2001)
    suave, lineal = d.fraccion_de_transicion(u, "suave"), d.fraccion_de_transicion(u, "lineal")
    ps, pl = np.gradient(suave, u), np.gradient(lineal, u)
    assert ps[0] == pytest.approx(0.0, abs=1e-2) and ps[-1] == pytest.approx(0.0, abs=1e-2)
    assert ps.max() == pytest.approx(1.5 * pl.max(), rel=1e-3)  # la lineal es la de menor cambio maximo de area


def test_perfil_de_seccion_va_del_puerto_inicial_al_final_y_sin_fin_es_constante():
    pts = _recta()
    a, h = d.perfil_de_seccion(pts, 60.0, 40.0, 120.0, 50.0, "lineal")
    assert (a[0], h[0]) == (60.0, 40.0) and a[-1] == pytest.approx(120.0) and h[-1] == pytest.approx(50.0)
    assert a[len(a) // 2] == pytest.approx(90.0)
    a, h = d.perfil_de_seccion(pts, 60.0, 40.0)
    assert np.all(a == 60.0) and np.all(h == 40.0)


def test_seccion_final_incompleta_o_no_positiva_se_rechaza():
    with pytest.raises(ValueError):
        d.perfil_de_seccion(_recta(), 60.0, 40.0, ancho_fin_mm=80.0)  # falta el alto
    with pytest.raises(ValueError):
        d.perfil_de_seccion(_recta(), 60.0, 40.0, 0.0, 10.0)


def test_rectangulos_con_seccion_variable_cambian_de_tamano_y_la_extension_repite_el_extremo():
    pts = _recta(100.0, 11)
    t, n, b = d.marcos_transporte_paralelo(pts)
    anchos, altos = d.perfil_de_seccion(pts, 40.0, 20.0, 80.0, 30.0, "lineal")
    rects = d.estaciones_rectangulares(pts, t, n, b, anchos, altos, extension_mm=5.0)
    assert len(rects) == 13
    ancho_de = lambda r: np.linalg.norm(r[0] - r[1])
    assert ancho_de(rects[0]) == pytest.approx(40.0) and ancho_de(rects[1]) == pytest.approx(40.0)
    assert ancho_de(rects[-1]) == pytest.approx(80.0) and ancho_de(rects[-2]) == pytest.approx(80.0)
    assert ancho_de(rects[6]) == pytest.approx(60.0)  # estacion del medio de la ruta


def test_construir_ducto_con_transicion_hace_los_lofts_y_rechaza_pared_imposible_en_el_extremo_chico(monkeypatch):
    registro = _instalar_cadquery_falso(monkeypatch)
    pts = _recta(200.0, 41)
    d.construir_ducto(pts, 60.0, 40.0, 2.0, ancho_fin_mm=120.0, alto_fin_mm=50.0)
    assert len(registro["lofts"]) == 2
    with pytest.raises(ValueError):  # alto final 3 mm: una pared de 2 mm no cabe en ninguna parte del ducto
        d.construir_ducto(pts, 60.0, 40.0, 2.0, ancho_fin_mm=60.0, alto_fin_mm=3.0)


def test_cambio_max_de_area_por_50mm_de_una_rampa_lineal():
    s = np.linspace(0, 300, 301)
    area = 1000.0 + 2.0 * s  # sube 2 mm2 por mm: en 50 mm sube 100, el mayor cambio relativo es al inicio (10 %)
    assert d.cambio_max_relativo_por(area, s, 50.0) == pytest.approx(0.1, rel=1e-3)
    assert d.cambio_max_relativo_por(np.array([1000.0, 1500.0]), np.array([0.0, 30.0]), 50.0) == pytest.approx(0.5)


def test_validar_transicion_reporta_area_aspecto_y_semiangulo(monkeypatch):
    pts = _recta(300.0, 301)
    params = {"area_min_mm2": 1000, "area_max_mm2": 7000, "aspect_ratio_max": 3.0}
    r = d.validar_transicion(pts, 60.0, 40.0, 120.0, 50.0, "lineal", params)
    assert r["area_inicio_mm2"] == 2400.0 and r["area_fin_mm2"] == 6000.0
    assert r["area_min_mm2"] == 2400.0 and r["area_max_mm2"] == 6000.0
    assert r["area_ok"] is True and r["aspecto_ok"] is True
    assert r["semiangulo_difusor_max_grados"] > 0 and r["semiangulo_contraccion_max_grados"] == 0
    assert d.validar_transicion(pts, 60.0, 40.0, 120.0, 50.0, "lineal", {"area_max_mm2": 5000, "area_min_mm2": 1})["area_ok"] is False
    assert d.validar_transicion(pts, 60.0, 40.0, 120.0, 50.0, "lineal", {})["area_ok"] == "no_verificado"
    # la ley suave concentra el cambio en el centro: el cambio de area por 50 mm es mayor que con la lineal
    suave = d.validar_transicion(pts, 60.0, 40.0, 120.0, 50.0, "suave", params)
    assert suave["cambio_max_de_area_por_50mm"] > r["cambio_max_de_area_por_50mm"]
    assert suave["semiangulo_difusor_max_grados"] > r["semiangulo_difusor_max_grados"]


def test_generar_ducto_con_transicion_agrega_seccion_fin_y_transicion_y_sin_ella_no(tmp_path, monkeypatch):
    _instalar_cadquery_falso(monkeypatch)
    monkeypatch.setattr(d, "ruta_cad", tmp_path)
    ruta = [[0, 0, 0], [200, 0, 0], [200, 200, 0]]
    sin = d.generar_ducto(ruta, 60.0, 40.0, 2.0, nombre="constante", exportar_archivos=False)
    assert "transicion" not in sin and "seccion_fin" not in sin
    con = d.generar_ducto(ruta, 60.0, 40.0, 2.0, nombre="variable", exportar_archivos=False,
                          ancho_fin_mm=100.0, alto_fin_mm=50.0, ley_transicion="lineal",
                          parametros={"area_min_mm2": 2000, "area_max_mm2": 5100, "aspect_ratio_max": 3})
    # el area que se valida es la INTERIOR (por donde pasa el aire): (60-4)x(40-4) y (100-4)x(50-4)
    assert con["seccion"]["area_mm2"] == 2016.0 and con["seccion_fin"]["area_mm2"] == 4416.0
    assert con["seccion"]["medida"] == "interior" and con["transicion"]["medida"] == "interior"
    assert con["transicion"]["ley"] == "lineal" and con["transicion"]["area_ok"] is True
    with pytest.raises(ValueError):
        d.generar_ducto(ruta, 60.0, 40.0, 2.0, ancho_fin_mm=100.0, exportar_archivos=False)  # falta alto_fin
    with pytest.raises(ValueError):
        d.generar_ducto(ruta, 60.0, 40.0, 2.0, ancho_fin_mm=100.0, alto_fin_mm=50.0, ley_transicion="x",
                        exportar_archivos=False)


def test_el_area_de_paso_es_la_interior_una_seccion_puede_cumplir_por_fuera_y_fallar_por_dentro():
    params = {"area_min_mm2": 2300, "area_max_mm2": 5000}
    por_fuera = d.validar_seccion(60.0, 40.0, params)                    # 2400 mm2 exteriores: "cumple"
    por_dentro = d.validar_seccion(60.0, 40.0, params, espesor_mm=2.0)   # 56 x 36 = 2016 mm2 de paso: NO cumple
    assert por_fuera["area_ok"] is True and por_fuera["medida"] == "exterior"
    assert por_dentro["area_ok"] is False and por_dentro["area_mm2"] == 2016.0 and por_dentro["medida"] == "interior"
    with pytest.raises(ValueError):
        d.validar_seccion(10.0, 40.0, params, espesor_mm=5.0)  # la pared se come toda la seccion


# ---- malla numpy del ducto: comprobacion independiente de las estaciones (sin CadQuery) -------------------------
def test_la_malla_numpy_es_cerrada_y_la_ficha_recupera_largo_y_perfil_de_area_de_un_ducto_curvo_con_transicion():
    """Mismas estaciones que usa el loft de CadQuery -> malla -> ficha: si las estaciones (marcos de transporte
    paralelo, perfil de seccion, espesor) estuvieran mal, la ficha no recuperaria lo que se pidio."""
    from herramientas import ficha_ducto as fd
    from herramientas import ingesta_malla as im

    curva = d.suavizar_ruta([[0, 0, 0], [400, 0, 0], [400, 0, 300]], iteraciones=5, paso_mm=5.0)
    tri = d.malla_numpy(curva, 70.0, 40.0, 2.0, max_estaciones=80, ancho_fin_mm=120.0, alto_fin_mm=50.0)
    assert im.info_malla(tri)["cerrada"]
    f = fd.ficha_de_malla(tri, paso_mm=3.0, espesor_pared_mm=2.0)
    assert f["longitud_linea_central_mm"] == pytest.approx(d.longitud_mm(curva), rel=0.05)

    anchos, altos = d.perfil_de_seccion(curva, 70.0, 40.0, 120.0, 50.0, "suave")
    s_nom = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(curva, axis=0), axis=1))])
    area_int = (anchos - 4.0) * (altos - 4.0)
    cuerpo = [q for q in f["perfil"] if 60.0 <= q["s_mm"] <= f["longitud_linea_central_mm"] - 60.0]
    # el orden de los extremos de la ficha es arbitrario: se toma el sentido que empieza por el lado angosto
    angosto_primero = cuerpo[0]["ancho_int_mm"] < cuerpo[-1]["ancho_int_mm"]
    for q in cuerpo:
        s = q["s_mm"] if angosto_primero else s_nom[-1] - q["s_mm"]
        assert q["area_int_mm2"] == pytest.approx(float(np.interp(s, s_nom, area_int)), rel=0.2)


def test_la_malla_numpy_rechaza_la_pared_imposible_igual_que_el_cad():
    pts = _recta(200.0, 41)
    with pytest.raises(ValueError):
        d.malla_numpy(pts, 60.0, 40.0, 2.0, ancho_fin_mm=60.0, alto_fin_mm=3.0)


# ---- perfil por estaciones (varias secciones de control, interpolacion PCHIP) ---------------------------------
SECCIONES = [(0.0, 100.0, 60.0), (0.5, 80.0, 30.0), (1.0, 120.0, 70.0)]   # cuerpo mas plano que ambos puertos


def test_el_perfil_por_estaciones_pasa_por_cada_seccion_de_control():
    pts = _recta(300.0, 61)                       # u = 0, 0.5 y 1 caen exactamente en una estacion
    a, h = d.perfil_de_seccion(pts, None, None, None, None, "suave", SECCIONES)
    assert np.allclose([a[0], a[30], a[-1]], [100.0, 80.0, 120.0])
    assert np.allclose([h[0], h[30], h[-1]], [60.0, 30.0, 70.0])


def test_pchip_no_se_sale_de_las_secciones_dadas_y_es_monotona_entre_nodos_monotonos():
    u = np.linspace(0, 1, 501)
    # nodos con un minimo en el centro: no hay rebote por debajo del minimo ni por encima del maximo
    y = d.interpolar_pchip([0, 0.5, 1], [60, 30, 70], u)
    assert y.min() >= 30.0 - 1e-9 and y.max() <= 70.0 + 1e-9
    # nodos crecientes: la curva no baja nunca (un spline cubico normal si rebotaria con este escalon)
    x, v = [0, 0.4, 0.5, 1], [10, 10.5, 40, 41]
    m = d.interpolar_pchip(x, v, u)
    assert np.all(np.diff(m) >= -1e-9) and m.min() >= 10.0 - 1e-9 and m.max() <= 41.0 + 1e-9
    assert np.allclose(d.interpolar_pchip(x, v, np.array(x, dtype=float)), v)


def test_pchip_coincide_con_scipy_si_esta_instalado():
    interpolate = pytest.importorskip("scipy.interpolate")
    rng = np.random.default_rng(3)
    for _ in range(50):
        x = np.sort(np.concatenate([[0.0, 1.0], rng.uniform(0.05, 0.95, rng.integers(0, 5))]))
        if np.any(np.diff(x) < 1e-3):
            continue
        y = rng.uniform(5, 100, len(x))
        u = np.linspace(0, 1, 101)
        assert np.allclose(d.interpolar_pchip(x, y, u), interpolate.PchipInterpolator(x, y)(u), atol=1e-9)


def test_con_dos_secciones_pchip_es_una_recta_igual_que_la_ley_lineal():
    pts = _recta(200.0, 41)
    dos = [(0.0, 60.0, 40.0), (1.0, 120.0, 50.0)]
    a, h = d.perfil_de_seccion(pts, None, None, None, None, "suave", dos)
    al, hl = d.perfil_de_seccion(pts, 60.0, 40.0, 120.0, 50.0, "lineal")
    assert np.allclose(a, al) and np.allclose(h, hl)
    # la rampa suave 3u^2-2u^3 NO es lo mismo que PCHIP con 2 nodos: solo existe en el modo clasico
    a_s, _ = d.perfil_de_seccion(pts, 60.0, 40.0, 120.0, 50.0, "suave")
    assert not np.allclose(a_s, al)


def test_la_ley_lineal_con_estaciones_son_tramos_rectos():
    pts = _recta(300.0, 61)
    a, _ = d.perfil_de_seccion(pts, None, None, None, None, "lineal", SECCIONES)
    assert a[15] == pytest.approx(90.0) and a[45] == pytest.approx(100.0)       # mitad de cada tramo recto


def test_secciones_mal_formadas_se_rechazan():
    pts = _recta(100.0, 21)
    malas = [
        [(0.0, 60.0, 40.0)],                                            # una sola
        [(0.1, 60.0, 40.0), (1.0, 60.0, 40.0)],                         # no empieza en 0
        [(0.0, 60.0, 40.0), (0.9, 60.0, 40.0)],                         # no termina en 1
        [(0.0, 60.0, 40.0), (0.5, 60.0, 40.0), (0.5, 70.0, 40.0), (1.0, 60.0, 40.0)],   # posicion repetida
        [(0.0, 60.0, 40.0), (0.7, 60.0, 40.0), (0.3, 60.0, 40.0), (1.0, 60.0, 40.0)],   # desordenada
        [(0.0, 60.0, 40.0), (0.5, 0.0, 40.0), (1.0, 60.0, 40.0)],       # ancho cero
        [(0.0, 60.0, 40.0), (0.5, float("nan"), 40.0), (1.0, 60.0, 40.0)],
        [(0.0, 60.0), (1.0, 60.0)],                                     # faltan columnas
    ]
    for sec in malas:
        with pytest.raises(ValueError):
            d.perfil_de_seccion(pts, None, None, None, None, "suave", sec)
    with pytest.raises(ValueError):                                      # estaciones Y ancho/alto: ambiguo
        d.perfil_de_seccion(pts, 60.0, 40.0, None, None, "suave", SECCIONES)
    with pytest.raises(ValueError):                                      # ni ancho/alto ni estaciones
        d.perfil_de_seccion(pts, None, None)


def test_sin_secciones_el_perfil_clasico_no_cambia():
    pts = _recta(300.0, 61)
    a, h = d.perfil_de_seccion(pts, 60.0, 40.0, 120.0, 50.0, "suave")
    f = d.fraccion_de_transicion(np.linspace(0, 1, 61), "suave")
    assert np.allclose(a, 60.0 + 60.0 * f) and np.allclose(h, 40.0 + 10.0 * f)


def test_validar_transicion_con_estaciones_ve_el_cuerpo_angosto():
    pts = _recta(300.0, 61)
    r = d.validar_transicion(pts, None, None, None, None, "suave", {"area_min_mm2": 3000, "area_max_mm2": 9000},
                             espesor_mm=2.0, secciones=SECCIONES)
    assert r["n_secciones"] == 3 and r["interpolacion"] == "pchip" and r["medida"] == "interior"
    assert r["area_min_mm2"] == pytest.approx((80 - 4) * (30 - 4))       # 1976: el cuerpo, mas chico que los dos puertos
    assert r["area_inicio_mm2"] == pytest.approx(96 * 56) and r["area_fin_mm2"] == pytest.approx(116 * 66)
    assert r["area_ok"] is False                                          # el cuerpo no llega a 3000 mm2


def test_construir_ducto_y_malla_rechazan_pared_imposible_en_el_cuerpo_con_estaciones(monkeypatch):
    _instalar_cadquery_falso(monkeypatch)
    pts = _recta(200.0, 41)
    secciones = [(0.0, 60.0, 40.0), (0.5, 60.0, 5.0), (1.0, 60.0, 40.0)]   # el cuerpo es muy bajo para 3 mm de pared
    with pytest.raises(ValueError):
        d.construir_ducto(pts, None, None, 3.0, secciones=secciones)
    with pytest.raises(ValueError):
        d.malla_numpy(pts, None, None, 3.0, secciones=secciones)


def test_generar_ducto_con_estaciones_reporta_primera_y_ultima_y_n_secciones(tmp_path, monkeypatch):
    registro = _instalar_cadquery_falso(monkeypatch)
    monkeypatch.setattr(d, "ruta_cad", tmp_path)
    rep = d.generar_ducto([[0, 0, 0], [200, 0, 0], [200, 200, 0]], None, None, 2.0, nombre="estaciones",
                          exportar_archivos=False, secciones=SECCIONES)
    assert rep["seccion"]["area_mm2"] == pytest.approx(96 * 56) and rep["seccion_fin"]["area_mm2"] == pytest.approx(116 * 66)
    assert rep["transicion"]["n_secciones"] == 3 and len(registro["lofts"]) == 2
    constante = [(0.0, 60.0, 40.0), (1.0, 60.0, 40.0)]
    sin = d.generar_ducto([[0, 0, 0], [200, 0, 0]], None, None, 2.0, nombre="c", exportar_archivos=False,
                          secciones=constante)
    assert "transicion" not in sin                                        # todas iguales = no hay transicion
    with pytest.raises(ValueError):
        d.generar_ducto([[0, 0, 0], [200, 0, 0]], 60.0, 40.0, 2.0, nombre="x", exportar_archivos=False,
                        secciones=SECCIONES)


def test_la_malla_con_estaciones_es_cerrada_y_la_ficha_ve_el_cuerpo_plano():
    """Prueba extremo a extremo: un ducto con puertos grandes y cuerpo plano se mide con la ficha y recupera el area
    minima del cuerpo (que una interpolacion de solo 2 secciones nunca produciria)."""
    from herramientas import ficha_ducto as fd
    from herramientas import ingesta_malla as im

    curva = d.suavizar_ruta([[0, 0, 0], [500, 0, 0]], paso_mm=5.0)
    tri = d.malla_numpy(curva, None, None, 2.0, max_estaciones=80, secciones=SECCIONES)
    assert im.info_malla(tri)["cerrada"]
    f = fd.ficha_de_malla(tri, paso_mm=3.0, espesor_pared_mm=2.0)
    assert f["area_int_mm2"]["min"] == pytest.approx((80 - 4) * (30 - 4), rel=0.15)
    assert f["area_int_mm2"]["min"] < 0.7 * min(f["puertos"]["extremo_1"]["seccion_20_a_80mm"]["area_int_mm2"],
                                                f["puertos"]["extremo_2"]["seccion_20_a_80mm"]["area_int_mm2"])


# ---- tramos rectos: la seccion no cambia en la boca -------------------------------------------------------------------
def test_con_rectos_la_seccion_es_constante_en_los_extremos_y_el_cambio_ocurre_entre_ellos():
    pts = _recta(400.0, 81)                                           # 5 mm por punto
    a, h = d.perfil_de_seccion(pts, 60.0, 40.0, 120.0, 50.0, "lineal", rectos_mm=(80.0, 100.0))
    s = np.linspace(0, 400, 81)
    assert np.allclose(a[s <= 80.0], 60.0) and np.allclose(a[s >= 300.0], 120.0)
    assert np.allclose(h[s <= 80.0], 40.0) and np.allclose(h[s >= 300.0], 50.0)
    medio = (s > 80.0) & (s < 300.0)
    assert np.all(np.diff(a[medio]) > 0)
    assert a[s == 190.0][0] == pytest.approx(90.0)                    # mitad del tramo libre (80..300)
    # con estaciones, pos se mide entre los tramos rectos
    sec = [(0.0, 60.0, 40.0), (0.5, 90.0, 20.0), (1.0, 60.0, 40.0)]
    a2, h2 = d.perfil_de_seccion(pts, None, None, None, None, "lineal", sec, rectos_mm=(80.0, 100.0))
    assert a2[s == 190.0][0] == pytest.approx(90.0) and h2[s == 190.0][0] == pytest.approx(20.0)
    assert np.allclose(a2[s <= 80.0], 60.0) and np.allclose(a2[s >= 300.0], 60.0)


def test_rectos_none_o_cero_no_cambian_nada_y_los_invalidos_se_rechazan():
    pts = _recta(300.0, 61)
    base = d.perfil_de_seccion(pts, 60.0, 40.0, 120.0, 50.0, "suave")
    for r in (None, (0.0, 0.0)):
        otro = d.perfil_de_seccion(pts, 60.0, 40.0, 120.0, 50.0, "suave", rectos_mm=r)
        assert np.allclose(base[0], otro[0]) and np.allclose(base[1], otro[1])
    for malo in ((-1.0, 10.0), (float("nan"), 0.0), (1.0,), "ab"):
        with pytest.raises(ValueError):
            d.perfil_de_seccion(pts, 60.0, 40.0, 120.0, 50.0, "suave", rectos_mm=malo)
    with pytest.raises(ValueError):                                    # los tramos ocupan todo el ducto
        d.perfil_de_seccion(pts, 60.0, 40.0, 120.0, 50.0, "suave", rectos_mm=(150.0, 150.0))


def test_generar_ducto_con_rectos_lo_reporta_en_la_transicion(tmp_path, monkeypatch):
    _instalar_cadquery_falso(monkeypatch)
    monkeypatch.setattr(d, "ruta_cad", tmp_path)
    rep = d.generar_ducto([[0, 0, 0], [400, 0, 0]], 60.0, 40.0, 2.0, nombre="r", exportar_archivos=False,
                          ancho_fin_mm=120.0, alto_fin_mm=50.0, rectos_mm=(80.0, 100.0))
    assert rep["transicion"]["tramos_rectos_mm"] == [80.0, 100.0]
    sin = d.generar_ducto([[0, 0, 0], [400, 0, 0]], 60.0, 40.0, 2.0, nombre="s", exportar_archivos=False,
                          ancho_fin_mm=120.0, alto_fin_mm=50.0)
    assert "tramos_rectos_mm" not in sin["transicion"]
    # el cambio por 50 mm es mayor con tramos rectos: la misma diferencia de area ocurre en menos largo
    assert rep["transicion"]["cambio_max_de_area_por_50mm"] > sin["transicion"]["cambio_max_de_area_por_50mm"]


def test_generar_ducto_con_curva_dada_no_necesita_ruta(tmp_path, monkeypatch):
    """La evaluacion con la linea central de una referencia pasa la curva ya hecha y ruta=None."""
    _instalar_cadquery_falso(monkeypatch)
    monkeypatch.setattr(d, "ruta_cad", tmp_path)
    curva = d.suavizar_ruta([[0, 0, 0], [300, 0, 0], [300, 200, 0]], paso_mm=5.0)
    rep = d.generar_ducto(None, None, None, 2.0, nombre="curva_dada", curva=curva, exportar_archivos=False,
                          secciones=SECCIONES, rectos_mm=(40.0, 40.0))
    assert rep["longitud_mm"] == pytest.approx(d.longitud_mm(curva), abs=0.1)
    assert rep["transicion"]["n_secciones"] == 3 and rep["transicion"]["tramos_rectos_mm"] == [40.0, 40.0]
