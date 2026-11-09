"""
Pruebas de la ficha tecnica de ductos (src/herramientas/ficha_ducto.py).

Se construyen ductos SINTETICOS de dimensiones conocidas (tubo rectangular hueco que gira 90 grados) y se comprueba
que el extractor recupera longitud, radio, secciones y puertos dentro de una tolerancia del orden del voxel.
Ejecutar desde la raiz del proyecto:  pytest tests/test_ficha_ducto.py -v
"""
import math

import numpy as np
import pytest

from herramientas import ficha_ducto as fd
from herramientas import ingesta_malla as im

PARED = 2.0


def _orientar(tri, deseada):
    """Voltea los triangulos cuya normal no apunta hacia 'deseada' (n x 3)."""
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    mal = np.einsum("ij,ij->i", n, deseada) < 0
    tri = tri.copy()
    tri[mal] = tri[mal][:, [0, 2, 1]]
    return tri


def tubo_rectangular(ancho=lambda u: 100.0, alto=lambda u: 50.0, recto1=300.0, radio=150.0, recto2=300.0,
                     pared=PARED, paso=4.0, por_lado=10):
    """Malla CERRADA de un tubo rectangular hueco: recto en +X, arco de 90 grados hacia +Z, recto en +Z.
    ancho(u), alto(u): dimensiones EXTERIORES segun la posicion u en [0, 1] a lo largo del eje."""
    L = recto1 + math.pi / 2 * radio + recto2
    s = np.arange(0.0, L + 1e-9, paso)
    if s[-1] < L:
        s = np.append(s, L)
    centros, tang = [], []
    for si in s:
        if si <= recto1:
            c, t = np.array([si, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
        elif si <= recto1 + math.pi / 2 * radio:
            th = (si - recto1) / radio
            c = np.array([recto1 + radio * math.sin(th), 0.0, radio * (1 - math.cos(th))])
            t = np.array([math.cos(th), 0.0, math.sin(th)])
        else:
            c = np.array([recto1 + radio, 0.0, radio + (si - recto1 - math.pi / 2 * radio)])
            t = np.array([0.0, 0.0, 1.0])
        centros.append(c)
        tang.append(t)
    Y = np.array([0.0, 1.0, 0.0])

    def anillo(i, mitad_w, mitad_h):
        n = np.cross(Y, tang[i])  # normal en el plano del giro
        lado = np.linspace(-1, 1, por_lado, endpoint=False)
        pts = ([(mitad_w, h * mitad_h) for h in lado] + [(-w * mitad_w, mitad_h) for w in lado]
               + [(-mitad_w, -h * mitad_h) for h in lado] + [(w * mitad_w, -mitad_h) for w in lado])
        pts = np.array(pts)
        return centros[i] + pts[:, 0:1] * Y + pts[:, 1:2] * n

    ext, intr = [], []
    for i, si in enumerate(s):
        u = si / L
        ext.append(anillo(i, ancho(u) / 2, alto(u) / 2))
        intr.append(anillo(i, ancho(u) / 2 - pared, alto(u) / 2 - pared))
    m = len(ext[0])
    tri = []
    for i in range(len(s) - 1):
        for j in range(m):
            k = (j + 1) % m
            for anillos, signo in ((ext, 1.0), (intr, -1.0)):
                a, b, c, d = anillos[i][j], anillos[i][k], anillos[i + 1][k], anillos[i + 1][j]
                tri += [(a, b, c), (a, c, d)]
    tri = np.array(tri)
    # orientacion: superficie exterior hacia afuera del eje, interior hacia el eje
    cen_tri = tri.mean(axis=1)
    # el tramo (cinta) de cada triangulo se reconstruye por orden de creacion
    por_cinta = 2 * m * 2
    cinta = np.repeat(np.arange(len(s) - 1), por_cinta)
    es_ext = (np.arange(len(tri)) % 4) < 2
    desde_eje = cen_tri - np.array(centros)[cinta]
    # componente perpendicular al eje
    tt = np.array(tang)[cinta]
    desde_eje -= np.einsum("ij,ij->i", desde_eje, tt)[:, None] * tt
    deseada = np.where(es_ext[:, None], desde_eje, -desde_eje)
    tri = _orientar(tri, deseada)
    # tapas anulares en los dos extremos
    tapas = []
    for i, signo in ((0, -1.0), (len(s) - 1, 1.0)):
        for j in range(m):
            k = (j + 1) % m
            tapas += [(ext[i][j], ext[i][k], intr[i][k]), (ext[i][j], intr[i][k], intr[i][j])]
    tapas = np.array(tapas)
    deseada_t = np.concatenate([np.tile(-tang[0], (2 * m, 1)), np.tile(tang[-1], (2 * m, 1))])
    tapas = _orientar(tapas, deseada_t)
    return np.concatenate([tri, tapas])


@pytest.fixture(scope="module")
def tubo():
    return tubo_rectangular()


@pytest.fixture(scope="module")
def ficha(tubo):
    return fd.ficha_de_malla(tubo, paso_mm=3.0, espesor_pared_mm=PARED)


# ---- el generador de pruebas es correcto ---------------------------------------------------------
def test_el_tubo_sintetico_es_cerrado_y_su_espesor_se_recupera(tubo):
    assert im.info_malla(tubo)["cerrada"]
    assert fd.espesor_de_pared(tubo) == pytest.approx(PARED, rel=0.15)


# ---- longitud, radio, secciones, puertos -------------------------------------------------------
def test_la_longitud_de_la_linea_central_coincide(ficha):
    esperada = 300 + math.pi / 2 * 150 + 300
    assert ficha["longitud_linea_central_mm"] == pytest.approx(esperada, rel=0.05)


def test_el_radio_minimo_se_parece_al_real_y_el_rango_lo_acota(ficha):
    assert ficha["radio_curvatura_minimo_mm"] == pytest.approx(150.0, rel=0.25)
    fino, grueso = ficha["radio_curvatura_minimo_rango_mm"]
    assert fino <= grueso and fino == pytest.approx(150.0, rel=0.35)


def test_la_seccion_del_tramo_recto_es_la_interior_del_tubo(ficha):
    medio = min(ficha["perfil"], key=lambda f: abs(f["s_mm"] - 150.0))
    assert medio["ancho_int_mm"] == pytest.approx(96.0, abs=8.0)   # 100 exterior - 2 x 2 de pared
    assert medio["alto_int_mm"] == pytest.approx(46.0, abs=8.0)
    assert medio["area_int_mm2"] == pytest.approx(96 * 46, rel=0.15)
    assert medio["diam_hidraulico_mm"] == pytest.approx(4 * 96 * 46 / (2 * (96 + 46)), rel=0.15)


def test_los_dos_puertos_estan_en_los_extremos_con_la_normal_hacia_afuera(ficha):
    p1, p2 = ficha["puertos"]["extremo_1"], ficha["puertos"]["extremo_2"]
    # cada puerto esta ~20 mm adentro del extremo; el orden de los extremos es arbitrario
    inicio, final = sorted([p1, p2], key=lambda p: p["punto_mm"][2])
    assert np.allclose(inicio["punto_mm"], [20, 0, 0], atol=6)
    assert np.allclose(final["punto_mm"], [450, 0, 430], atol=6)
    assert float(np.dot(inicio["normal_hacia_afuera"], [-1, 0, 0])) > 0.97
    assert float(np.dot(final["normal_hacia_afuera"], [0, 0, 1])) > 0.97
    for p in (p1, p2):
        assert p["seccion_20_a_80mm"]["ancho_int_mm"] == pytest.approx(96.0, abs=8.0)


def test_un_tubo_de_seccion_constante_tiene_area_casi_constante_y_aspecto_correcto(ficha):
    assert ficha["cambio_max_de_area_por_50mm"] < 0.15
    assert ficha["relacion_aspecto"]["mediana"] == pytest.approx(96 / 46, rel=0.2)


# ---- transicion de seccion --------------------------------------------------------------------
def test_una_transicion_de_ancho_plano_a_angosto_alto_se_ve_en_el_perfil():
    tubo_t = tubo_rectangular(ancho=lambda u: 120.0 - 60.0 * u, alto=lambda u: 40.0 + 40.0 * u)
    f = fd.ficha_de_malla(tubo_t, paso_mm=3.0, espesor_pared_mm=PARED)
    inicio, final = sorted(f["puertos"].values(), key=lambda p: p["punto_mm"][2])
    assert inicio["seccion_20_a_80mm"]["ancho_int_mm"] > final["seccion_20_a_80mm"]["ancho_int_mm"] + 20
    assert inicio["seccion_20_a_80mm"]["alto_int_mm"] < final["seccion_20_a_80mm"]["alto_int_mm"] - 15
    assert f["relacion_aspecto"]["max"] > 2.0


def test_la_ficha_sigue_el_perfil_de_area_nominal_de_una_transicion_suave():
    """Cruza la ficha (que mide una MALLA) con la ley de transicion de cad/ducto.py: si el ducto generado cambia de
    seccion como dice perfil_de_seccion, la ficha debe recuperar ese mismo perfil de area."""
    from cad import ducto as d

    a0, h0, a1, h1 = 60.0, 40.0, 120.0, 50.0

    def lado(u, x0, x1):
        return float(x0 + (x1 - x0) * d.fraccion_de_transicion(u, "suave"))

    malla = tubo_rectangular(ancho=lambda u: lado(u, a0, a1), alto=lambda u: lado(u, h0, h1))
    f = fd.ficha_de_malla(malla, paso_mm=3.0, espesor_pared_mm=PARED)
    largo = f["longitud_linea_central_mm"]
    perfil = [q for q in f["perfil"] if 60.0 <= q["s_mm"] <= largo - 60.0]
    # el orden de los extremos de la ficha es arbitrario: se toma el sentido que empieza por el lado angosto
    angosto_primero = perfil[0]["ancho_int_mm"] < perfil[-1]["ancho_int_mm"]
    for q in perfil:
        u = q["s_mm"] / largo if angosto_primero else 1.0 - q["s_mm"] / largo
        esperada = (lado(u, a0, a1) - 2 * PARED) * (lado(u, h0, h1) - 2 * PARED)
        assert q["area_int_mm2"] == pytest.approx(esperada, rel=0.2)


# ---- robustez -------------------------------------------------------------------------------
def test_una_pieza_suelta_se_ignora_y_se_avisa(tubo):
    clip = im.malla_caja((600, 100, 0), (620, 120, 20))  # lejos del ducto
    f = fd.ficha_de_malla(np.concatenate([tubo, clip]), paso_mm=3.0, espesor_pared_mm=PARED)
    assert any("suelta" in a for a in f["advertencias"])
    assert f["longitud_linea_central_mm"] == pytest.approx(300 + math.pi / 2 * 150 + 300, rel=0.05)


def test_la_ficha_no_importa_scipy():
    # el modulo debe funcionar solo con numpy (scipy no esta en requirements.txt)
    import inspect
    fuente = inspect.getsource(fd)
    assert "import scipy" not in fuente and "from scipy" not in fuente


def test_geometria_2d_area_y_perimetro_del_casco():
    cuadrado = np.array([[0, 0], [10, 0], [10, 4], [0, 4], [5, 2], [3, 1]], dtype=float)  # los interiores se descartan
    area, per = fd.area_perimetro(fd.casco_convexo(cuadrado))
    assert area == pytest.approx(40.0) and per == pytest.approx(28.0)


def test_radios_de_una_circunferencia_conocida():
    th = np.linspace(0, math.pi, 200)
    c = np.stack([100 * np.cos(th), 100 * np.sin(th), np.zeros_like(th)], axis=1)
    r = fd.radios_de_curvatura(c, separacion_mm=30.0)
    assert np.nanmedian(r) == pytest.approx(100.0, rel=0.02)
    assert np.isnan(r[0]) and np.isnan(r[-1])  # los extremos no tienen vecinos suficientes
    recta = np.stack([np.linspace(0, 300, 100), np.zeros(100), np.zeros(100)], axis=1)
    assert np.all(np.isinf(fd.radios_de_curvatura(recta, 30.0)[np.isfinite(fd.radios_de_curvatura(recta, 30.0))]))
