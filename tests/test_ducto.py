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
