"""
Pruebas de src/cad/curva_ruta.py (ruta del A* -> curva con radio exacto + revision de colisiones).
Todo es numpy puro, sin CadQuery. Para ducto_desde_entorno() se usa un modulo 'ruteo_astar' falso
y un generar_ducto falso, porque lo que se prueba ahi es el pegamento, no el CAD.

Ejecutar desde la raiz del proyecto:  pytest tests -v
"""
import math
import sys
import types

import numpy as np
import pytest

from cad import curva_ruta as cr
from cad import ducto as d


def _bloque(nombre, centro, size, tipo="neutro", valor=0):
    return {"id": nombre, "nombre": nombre, "tipo": tipo, "valor": valor,
            "centro": dict(zip("xyz", centro)), "size": dict(zip("xyz", size))}


# ---- redondear_esquinas -------------------------------------------------------------------------
def test_esquina_de_90_grados_es_un_arco_de_radio_exacto():
    ruta = [[0, 0, 0], [300, 0, 0], [300, 300, 0]]
    curva, esquinas = cr.redondear_esquinas(ruta, radio_mm=50.0, paso_mm=5.0)

    assert esquinas == [{"vertice": 1, "angulo_giro_deg": 90.0, "radio_pedido_mm": 50.0,
                         "radio_logrado_mm": 50.0, "factible": True}]
    assert np.allclose(curva[0], [0, 0, 0]) and np.allclose(curva[-1], [300, 300, 0])
    # el arco empieza T = R*tan(45) = 50 mm antes del vertice y su centro es (250, 50, 0)
    arco = curva[(curva[:, 0] > 250.0 + 1e-6) & (curva[:, 1] < 50.0 - 1e-6)]
    assert len(arco) > 5
    assert np.allclose(np.linalg.norm(arco - [250, 50, 0], axis=1), 50.0, atol=1e-6)
    assert d.radios_de_curvatura(curva).min() == pytest.approx(50.0, abs=0.05)


def test_la_curva_es_tangente_a_las_rectas_sin_saltos():
    ruta = [[0, 0, 0], [300, 0, 0], [300, 300, 0]]
    curva, _ = cr.redondear_esquinas(ruta, 50.0, 5.0)
    pasos = np.linalg.norm(np.diff(curva, axis=0), axis=1)
    assert pasos.max() <= 5.0 + 1e-6      # ningun salto entre puntos consecutivos
    assert pasos.min() > 1e-6             # y ningun punto repetido


def test_esquina_en_3d_fuera_de_plano_tambien_da_el_radio_pedido():
    ruta = [[0, 0, 0], [200, 0, 0], [200, 0, 200]]    # gira en el plano XZ
    curva, esquinas = cr.redondear_esquinas(ruta, 40.0, 4.0)
    assert esquinas[0]["factible"]
    assert np.allclose(curva[:, 1], 0.0)
    assert d.radios_de_curvatura(curva).min() == pytest.approx(40.0, abs=0.05)


def test_esquina_que_no_cabe_se_reporta_y_da_el_radio_maximo_posible():
    # tramo del medio de 100 mm con dos giros de 90: cada vertice solo puede usar 50 mm (T), no 80
    ruta = [[0, 0, 0], [100, 0, 0], [100, 100, 0], [200, 100, 0]]
    curva, esquinas = cr.redondear_esquinas(ruta, 80.0, 5.0)

    assert [e["factible"] for e in esquinas] == [False, False]
    assert [e["radio_logrado_mm"] for e in esquinas] == [50.0, 50.0]
    assert d.radios_de_curvatura(curva).min() == pytest.approx(50.0, abs=0.05)


def test_vertice_colineal_no_genera_esquina():
    curva, esquinas = cr.redondear_esquinas([[0, 0, 0], [100, 0, 0], [200, 0, 0]], 50.0, 10.0)
    assert esquinas == []
    assert np.allclose(curva[:, 1:], 0.0) and np.allclose(curva[-1], [200, 0, 0])


def test_vuelta_en_u_no_revienta_y_se_marca_no_factible():
    curva, esquinas = cr.redondear_esquinas([[0, 0, 0], [100, 0, 0], [0, 0.0001, 0]], 20.0, 10.0)
    assert esquinas[0]["factible"] is False
    assert np.allclose(curva[-1], [0, 0.0001, 0])


# ---- distancia a cajas ---------------------------------------------------------------------------
CAJA = (0.0, 0.0, 0.0, 10.0, 10.0, 10.0)


def test_distancia_segmento_caja_exacta():
    # recta paralela al borde superior, 5 mm arriba; su punto mas cercano esta en medio del segmento
    assert cr._dist_min_segmento_caja([-20, 5, 15], [30, 5, 15], CAJA) == pytest.approx(5.0, abs=1e-6)
    # segmento que atraviesa la caja
    assert cr._dist_min_segmento_caja([-5, 5, 5], [15, 5, 5], CAJA) == pytest.approx(0.0, abs=1e-6)
    # el minimo esta en un extremo del segmento
    assert cr._dist_min_segmento_caja([13, 5, 5], [50, 5, 5], CAJA) == pytest.approx(3.0, abs=1e-6)


def test_cajas_desde_bloques_calcula_zona_de_seguridad():
    cajas = cr.cajas_desde_bloques([_bloque("M", [0, 0, 0], [10, 20, 30], "termico", 100),
                                    _bloque("N", [100, 0, 0], [10, 10, 10])])
    assert cajas[0]["caja"] == (-5.0, -10.0, -15.0, 5.0, 10.0, 15.0)
    assert cajas[0]["dseg"] == 100.0
    assert cajas[1]["dseg"] == 0.0


def test_cajas_desde_bloques_usa_la_funcion_de_seguridad_que_se_le_pase():
    cajas = cr.cajas_desde_bloques([_bloque("M", [0, 0, 0], [1, 1, 1], "termico", 10)],
                                   distancia_seguridad=lambda tipo, valor: valor * 3.0)
    assert cajas[0]["dseg"] == 30.0


# ---- simplificar_ruta ----------------------------------------------------------------------------
def test_sin_obstaculos_la_ruta_se_vuelve_recta():
    zigzag = [[0, 0, 0], [50, 40, 0], [100, 0, 0], [150, 40, 0], [200, 0, 0]]
    assert cr.simplificar_ruta(zigzag, [], radio_mm=10).tolist() == [[0, 0, 0], [200, 0, 0]]


def test_un_obstaculo_en_el_camino_obliga_a_conservar_vertices():
    cajas = cr.cajas_desde_bloques([_bloque("B", [100, 0, 0], [40, 40, 400])])
    ruta = [[0, 0, 0], [60, 80, 0], [140, 80, 0], [200, 0, 0]]   # rodea el bloque por y>0
    simple = cr.simplificar_ruta(ruta, cajas, radio_mm=10)
    assert len(simple) > 2
    for i in range(len(simple) - 1):
        assert cr._dist_min_segmento_caja(simple[i], simple[i + 1], cajas[0]["caja"]) >= 10 - 1e-6


def test_el_atajo_no_invade_la_zona_de_seguridad_que_la_ruta_original_respetaba():
    cajas = cr.cajas_desde_bloques([_bloque("Motor", [100, 0, 0], [20, 20, 20], "termico", 60)])
    ruta = [[0, 0, 0], [0, 150, 0], [200, 150, 0], [200, 0, 0]]   # pasa a ~140 mm del motor
    simple = cr.simplificar_ruta(ruta, cajas, radio_mm=10)
    for i in range(len(simple) - 1):
        # exigido: min(radio + zona = 70, holgura previa) -> nunca baja de 70 mm si la ruta ya cumplia
        assert cr._dist_min_segmento_caja(simple[i], simple[i + 1], cajas[0]["caja"]) >= 70 - 1e-6


# ---- holguras y revision -------------------------------------------------------------------------
def test_holguras_negativas_marcan_colision():
    cajas = cr.cajas_desde_bloques([_bloque("B", [0, 0, 0], [10, 10, 10])])  # caja de -5 a 5
    curva = np.array([[-50.0, 20.0, 0.0], [50.0, 20.0, 0.0]])               # pasa a 15 mm de la cara
    assert cr.holguras_mm(curva, cajas, radio_mm=10.0) == {"B": 5.0}
    assert cr.revisar_entorno(curva, cajas, 10.0)["colisiones"] == []
    assert cr.revisar_entorno(curva, cajas, 20.0)["colisiones"] == ["B"]


def test_revisar_entorno_detecta_zona_de_seguridad():
    cajas = cr.cajas_desde_bloques([_bloque("M", [0, 0, 0], [10, 10, 10], "termico", 50)])
    curva = np.array([[-50.0, 30.0, 0.0], [50.0, 30.0, 0.0]])   # a 25 mm de la cara, ducto de 5 -> holgura 20
    rev = cr.revisar_entorno(curva, cajas, 5.0)
    assert rev["colisiones"] == [] and rev["dentro_de_zona_de_seguridad"] == ["M"]


def test_varias_cajas_con_el_mismo_nombre_usan_la_holgura_minima_no_la_ultima():
    # un componente aproximado con 3 cajas (ver ingesta_malla): la curva pasa muy cerca de la PRIMERA
    bloques = [_bloque("Motor", [0, 0, 0], [10, 10, 10], "termico", 50),
               _bloque("Motor", [100, 0, 0], [10, 10, 10], "termico", 50),
               _bloque("Motor", [200, 0, 0], [10, 10, 10], "termico", 50)]
    cajas = cr.cajas_desde_bloques(bloques)
    curva = np.array([[-50.0, 12.0, 0.0], [50.0, 12.0, 0.0]])   # a 7 mm de la cara de la primera caja
    assert cr.holguras_mm(curva, cajas, radio_mm=0.0) == {"Motor": 7.0}   # antes: 'ganaba' la ultima caja (lejos)
    rev = cr.revisar_entorno(curva, cajas, 8.0)
    assert rev["colisiones"] == ["Motor"]            # una sola vez, no una por caja
    assert cr.revisar_entorno(curva, cajas, 0.0)["dentro_de_zona_de_seguridad"] == ["Motor"]


# ---- construir_curva -----------------------------------------------------------------------------
def test_construir_curva_cumple_r_min_y_reduce_vertices():
    zigzag = [[0, 0, 0], [40, 40, 0], [80, 0, 0], [120, 40, 0], [160, 0, 0], [200, 40, 0], [400, 40, 0]]
    curva, info = cr.construir_curva(zigzag, [], r_min_mm=45.0, radio_envolvente_mm=20.0, paso_mm=5.0)
    assert info["factible"] is True
    assert info["vertices_simplificada"] < info["vertices_ruta_original"]
    assert validar(curva, 45.0)


def validar(curva, r_min):
    return d.validar_radio_minimo(curva, r_min)["cumple"]


def test_construir_curva_marca_infactible_si_el_pasillo_no_deja_curvar():
    # dos muros desfasados obligan a un cambio de nivel de 140 mm en solo 20 mm de avance:
    # con R = 300 mm las esquinas no caben (y no hay atajo posible)
    bloques = [_bloque("A", [100, -100, 0], [60, 300, 400]), _bloque("B", [200, 100, 0], [60, 300, 400])]
    ruta = [[0, 70, 0], [140, 70, 0], [160, -70, 0], [300, -70, 0]]
    curva, info = cr.construir_curva(ruta, bloques, r_min_mm=300.0, radio_envolvente_mm=10.0, paso_mm=5.0)
    assert info["factible"] is False
    assert info["esquinas_no_factibles"] > 0 or info["colisiones"]


# ---- ducto_desde_entorno (pegamento) ------------------------------------------------------------
# Escena real (salida del A* de Fer): dos muros desfasados que obligan a una "S". R_min ~100 cabe; 600 no.
MUROS = [_bloque("P1", [500, 150, 0], [100, 600, 3000]), _bloque("P2", [1000, -150, 0], [100, 600, 3000])]
RUTA_S = [[0, 0, 0], [180, 0, 0], [405, -225, 0], [585, -225, 0], [900, 90, 0], [900, 180, 0], [945, 225, 0],
          [1080, 225, 0], [1170, 135, 0], [1215, 135, 0], [1260, 90, 0], [1305, 90, 0], [1350, 45, 0],
          [1395, 45, 0], [1440, 0, 0], [1500, 0, 0]]


def test_radio_maximo_factible_es_el_borde_entre_lo_que_cabe_y_lo_que_no():
    r = cr.radio_maximo_factible(RUTA_S, MUROS, r_max_mm=600.0, radio_envolvente_mm=23.6, paso_mm=6.25)
    assert 100 < r < 300
    assert cr.construir_curva(RUTA_S, MUROS, r, 23.6, 6.25)[1]["factible"] is True
    assert cr.construir_curva(RUTA_S, MUROS, r + 20, 23.6, 6.25)[1]["factible"] is False


def test_radio_maximo_factible_es_none_si_la_ruta_ya_choca():
    choca = [[0, 0, 0], [1500, 0, 0]]   # recta que atraviesa P1
    assert cr.radio_maximo_factible(choca, MUROS, 600.0, 23.6) is None


def _instalar_ruteo_falso(monkeypatch, estado="ok", ruta=None, bloques=None):
    modulo = types.ModuleType("ruteo_astar")
    modulo.llamadas = []

    def calculo_ruta(inicio, fin, costo_ruta=None):
        modulo.llamadas.append(costo_ruta)
        if estado != "ok":
            return {"ruta": [], "longitud_mm": 0.0, "estado": estado, "bloques_cercanos": [], "mensaje": "no hay camino"}
        return {"ruta": ruta or [[0, 0, 0], [100, 0, 0], [100, 100, 0], [300, 100, 0]], "longitud_mm": 400.0,
                "estado": "ok", "bloques_cercanos": ["B"], "mensaje": ""}

    modulo.calculo_ruta = calculo_ruta
    modulo.listar_bloques = lambda: bloques or []
    modulo.distancia_seguridad_mm = lambda tipo, valor: 0.0
    monkeypatch.setitem(sys.modules, "ruteo_astar", modulo)
    return modulo


def test_ducto_desde_entorno_pasa_el_radio_de_la_envolvente_al_astar_y_la_curva_al_cad(monkeypatch):
    ruteo = _instalar_ruteo_falso(monkeypatch)
    capturado = {}

    def generar_falso(ruta, ancho, alto, espesor, **kw):
        capturado.update(kw, ruta=ruta)
        return {"nombre": kw["nombre"], "solido_valido": True}

    monkeypatch.setattr(cr, "generar_ducto", generar_falso)

    rep = cr.ducto_desde_entorno([0, 0, 0], [300, 100, 0], 40.0, 30.0, 2.0,
                                 parametros={"r_min_mm": 45.0}, nombre="prueba")

    assert ruteo.llamadas[0]["radio_ducto_mm"] == pytest.approx(25.0)   # hypot(40, 30)/2
    assert capturado["curva"] is not None and len(capturado["curva"]) > 10
    assert rep["estado"] == "ok" and rep["curva"]["factible"] is True
    assert rep["ruta_astar"]["bloques_cercanos"] == ["B"]
    assert d.radios_de_curvatura(capturado["curva"]).min() >= 45.0


def test_ducto_desde_entorno_con_seccion_final_usa_la_mayor_para_esquivar_y_la_pasa_al_cad(monkeypatch):
    ruteo = _instalar_ruteo_falso(monkeypatch)
    capturado = {}
    monkeypatch.setattr(cr, "generar_ducto", lambda ruta, a, h, e, **kw: capturado.update(kw) or {"nombre": "x"})

    cr.ducto_desde_entorno([0, 0, 0], [300, 100, 0], 40.0, 30.0, 2.0, parametros={"r_min_mm": 45.0},
                           ancho_fin_mm=100.0, alto_fin_mm=60.0, ley_transicion="lineal")

    assert ruteo.llamadas[0]["radio_ducto_mm"] == pytest.approx(math.hypot(100, 60) / 2)  # la seccion MAYOR
    assert capturado["ancho_fin_mm"] == 100.0 and capturado["alto_fin_mm"] == 60.0 and capturado["ley_transicion"] == "lineal"
    with pytest.raises(ValueError):
        cr.ducto_desde_entorno([0, 0, 0], [300, 100, 0], 40.0, 30.0, 2.0, ancho_fin_mm=100.0)


def test_ducto_desde_entorno_con_estaciones_usa_la_mayor_de_todas_y_las_pasa_al_cad(monkeypatch):
    ruteo = _instalar_ruteo_falso(monkeypatch)
    capturado = {}
    monkeypatch.setattr(cr, "generar_ducto", lambda ruta, a, h, e, **kw: capturado.update(kw, ancho=a, alto=h) or {"nombre": "x"})
    secciones = [(0.0, 100.0, 60.0), (0.5, 160.0, 30.0), (1.0, 120.0, 70.0)]    # la del centro es la mas ancha

    cr.ducto_desde_entorno([0, 0, 0], [300, 100, 0], None, None, 2.0, parametros={"r_min_mm": 45.0},
                           secciones=secciones)

    assert ruteo.llamadas[0]["radio_ducto_mm"] == pytest.approx(math.hypot(160, 30) / 2)
    assert capturado["secciones"] == secciones and capturado["ancho"] is None and capturado["alto"] is None
    with pytest.raises(ValueError):   # estaciones Y ancho/alto: ambiguo
        cr.ducto_desde_entorno([0, 0, 0], [300, 100, 0], 40.0, 30.0, 2.0, secciones=secciones)
    with pytest.raises(ValueError):   # estaciones mal formadas
        cr.ducto_desde_entorno([0, 0, 0], [300, 100, 0], None, None, 2.0, secciones=[(0.0, 100.0, 60.0)])


def test_ducto_desde_entorno_sin_r_min_avisa_y_no_manda_curva(monkeypatch):
    _instalar_ruteo_falso(monkeypatch)
    capturado = {}
    monkeypatch.setattr(cr, "generar_ducto", lambda *a, **kw: capturado.update(kw) or {"nombre": "x"})
    rep = cr.ducto_desde_entorno([0, 0, 0], [300, 100, 0], 40.0, 30.0, 2.0, parametros={})
    assert capturado["curva"] is None
    assert "aviso" in rep["curva"]


def test_ducto_desde_entorno_devuelve_el_error_del_astar_sin_lanzar(monkeypatch):
    _instalar_ruteo_falso(monkeypatch, estado="sin_ruta")
    monkeypatch.setattr(cr, "generar_ducto", lambda *a, **kw: pytest.fail("no debe generar CAD sin ruta"))
    rep = cr.ducto_desde_entorno([0, 0, 0], [300, 100, 0], 40.0, 30.0, 2.0, parametros={"r_min_mm": 45.0})
    assert rep == {"estado": "sin_ruta", "mensaje": "no hay camino"}


def test_infactible_anuncia_propone_y_NO_genera_cad(monkeypatch):
    _instalar_ruteo_falso(monkeypatch, ruta=RUTA_S, bloques=MUROS)
    monkeypatch.setattr(cr, "generar_ducto", lambda *a, **kw: pytest.fail("no debe generar CAD si es infactible"))

    rep = cr.ducto_desde_entorno([0, 0, 0], [1500, 0, 0], 40.0, 25.0, 2.0, parametros={"r_min_mm": 600.0})

    assert rep["estado"] == "infactible"
    assert "600" in rep["mensaje"] and rep["pregunta"]
    ids = [p["id"] for p in rep["propuestas"]]
    assert ids == ["bajar_r_min", "mover_entorno", "generar_igual"]
    bajar = rep["propuestas"][0]
    assert bajar["requiere_aprobacion"] is True            # bajar R_min NO lo decide la maquina
    assert bajar["parametros"]["r_min_mm"] == rep["radio_maximo_factible_mm"] < 600
    assert rep["diagnostico"]["esquinas_no_factibles"]


def test_infactible_por_colision_no_ofrece_generar_igual(monkeypatch):
    _instalar_ruteo_falso(monkeypatch, ruta=[[0, 0, 0], [1500, 0, 0]], bloques=MUROS)
    monkeypatch.setattr(cr, "generar_ducto", lambda *a, **kw: pytest.fail("no debe generar CAD"))
    rep = cr.ducto_desde_entorno([0, 0, 0], [1500, 0, 0], 40.0, 25.0, 2.0, parametros={"r_min_mm": 100.0})
    assert rep["estado"] == "infactible"
    assert "generar_igual" not in [p["id"] for p in rep["propuestas"]]
    assert "bajar_r_min" not in [p["id"] for p in rep["propuestas"]]   # ningun radio arregla un choque


def test_generar_si_infactible_si_genera_el_ducto(monkeypatch):
    _instalar_ruteo_falso(monkeypatch, ruta=RUTA_S, bloques=MUROS)
    capturado = {}
    monkeypatch.setattr(cr, "generar_ducto", lambda *a, **kw: capturado.update(kw) or {"nombre": "x"})
    rep = cr.ducto_desde_entorno([0, 0, 0], [1500, 0, 0], 40.0, 25.0, 2.0,
                                 parametros={"r_min_mm": 600.0}, generar_si_infactible=True)
    assert rep["estado"] == "ok" and rep["curva"]["factible"] is False
    assert capturado["curva"] is not None


def test_ducto_desde_entorno_pasa_campo_costo_solo_si_se_da_uno(monkeypatch):
    ruteo = _instalar_ruteo_falso(monkeypatch)
    vistos = []
    original = ruteo.calculo_ruta

    def calculo(inicio, fin, costo_ruta=None, **kw):       # un A* VIEJO no aceptaria campo_costo: con None no se manda
        vistos.append(kw)
        return original(inicio, fin, costo_ruta)

    ruteo.calculo_ruta = calculo
    monkeypatch.setattr(cr, "generar_ducto", lambda *a, **kw: {"nombre": "x"})
    campo = lambda pts: [0.0] * len(pts)                   # noqa: E731

    cr.ducto_desde_entorno([0, 0, 0], [300, 100, 0], 40.0, 30.0, 2.0, parametros={"r_min_mm": 45.0})
    cr.ducto_desde_entorno([0, 0, 0], [300, 100, 0], 40.0, 30.0, 2.0, parametros={"r_min_mm": 45.0}, campo_costo=campo)

    assert vistos[0] == {} and vistos[1] == {"campo_costo": campo}


# ---- tramos rectos en los puertos ----------------------------------------------------------------------------------
def test_tramos_de_puertos_llevan_cada_extremo_hacia_adentro_por_la_normal():
    t = cr.tramos_de_puertos([0, 0, 0], [500, 200, 0],
                             {"inicio": {"normal": [-1, 0, 0], "tramo_mm": 100.0},
                              "fin": {"normal": [0, 2, 0], "tramo_mm": 80.0, "seccion_constante_mm": 50.0}})
    assert np.allclose(t["inicio_astar"], [100.0, 0.0, 0.0])          # la normal apunta hacia afuera: el ducto entra al reves
    assert np.allclose(t["fin_astar"], [500.0, 120.0, 0.0])           # normal no unitaria: se normaliza
    assert t["rectos_mm"] == (100.0, 50.0)                            # mm de SECCION constante, no de tramo recto
    sin = cr.tramos_de_puertos([0, 0, 0], [500, 0, 0], None)
    assert sin["inicio_astar"] == [0.0, 0.0, 0.0] and sin["rectos_mm"] == (0.0, 0.0)
    solo_uno = cr.tramos_de_puertos([0, 0, 0], [500, 0, 0], {"fin": {"normal": [1, 0, 0], "tramo_mm": 60.0}})
    assert solo_uno["inicio_astar"] == [0.0, 0.0, 0.0] and np.allclose(solo_uno["fin_astar"], [440.0, 0.0, 0.0])


def test_tramos_de_puertos_rechaza_datos_invalidos():
    for malo in ({"inicio": {"normal": [0, 0, 0], "tramo_mm": 50.0}},          # normal nula
                 {"inicio": {"tramo_mm": 50.0}},                                # sin normal
                 {"inicio": {"normal": [1, 0, 0], "tramo_mm": -5.0}},
                 {"inicio": {"normal": [1, 0, 0], "tramo_mm": float("nan")}},
                 {"inicio": {"normal": [1, 0], "tramo_mm": 50.0}},
                 {"inicio": {"normal": [1, 0, 0], "tramo_mm": 50.0, "seccion_constante_mm": -1}}):
        with pytest.raises(ValueError):
            cr.tramos_de_puertos([0, 0, 0], [500, 0, 0], malo)
    with pytest.raises(ValueError):   # los dos tramos terminan en el mismo punto
        cr.tramos_de_puertos([0, 0, 0], [100, 0, 0], {"inicio": {"normal": [-1, 0, 0], "tramo_mm": 50.0},
                                                      "fin": {"normal": [1, 0, 0], "tramo_mm": 50.0}})


def test_la_simplificacion_no_se_come_los_tramos_rectos_y_la_curva_sale_por_la_normal():
    ruta_astar = [[100, 0, 0], [100, 150, 0], [400, 200, 0]]               # del fin del tramo 1 al inicio del tramo 2
    curva, info = cr.construir_curva(ruta_astar, [], 45.0, 20.0, 5.0, prefijo=[[0, 0, 0]], sufijo=[[400, 300, 0]])
    # sin obstaculos la simplificacion deja solo los extremos, pero los tramos deben seguir ahi
    assert info["vertices_simplificada"] == 4 and info["factible"]
    assert np.allclose(curva[0], [0, 0, 0]) and np.allclose(curva[-1], [400, 300, 0])
    primera = curva[curva[:, 0] < 40.0]                                    # el arranque va por el eje X, sin desviarse
    assert np.allclose(primera[:, 1], 0.0, atol=1e-6)


def test_ducto_desde_entorno_con_puertos_corre_el_astar_entre_los_extremos_de_los_tramos_y_pasa_rectos_al_cad(monkeypatch):
    modulo = types.ModuleType("ruteo_astar")
    visto = {}

    def calculo_ruta(inicio, fin, cfg=None):
        visto.update(inicio=list(inicio), fin=list(fin))
        return {"ruta": [list(inicio), list(fin)], "longitud_mm": 1.0, "estado": "ok", "bloques_cercanos": [], "mensaje": ""}

    modulo.calculo_ruta, modulo.listar_bloques = calculo_ruta, (lambda: [])
    modulo.distancia_seguridad_mm = lambda tipo, valor: 0.0
    monkeypatch.setitem(sys.modules, "ruteo_astar", modulo)
    capturado = {}
    monkeypatch.setattr(cr, "generar_ducto", lambda ruta, a, h, e, **kw: capturado.update(kw, ruta=ruta) or {"nombre": "x"})
    puertos = {"inicio": {"normal": [-1, 0, 0], "tramo_mm": 110.0, "seccion_constante_mm": 80.0},
               "fin": {"normal": [0, 1, 0], "tramo_mm": 100.0, "seccion_constante_mm": 80.0}}

    rep = cr.ducto_desde_entorno([0, 0, 0], [400, 300, 0], 60.0, 40.0, 2.0, parametros={"r_min_mm": 45.0}, puertos=puertos)

    assert np.allclose(visto["inicio"], [110, 0, 0]) and np.allclose(visto["fin"], [400, 200, 0])
    assert capturado["rectos_mm"] == (80.0, 80.0) and rep["tramos_rectos_mm"] == [110.0, 100.0]
    curva = capturado["curva"]
    assert np.allclose(curva[0], [0, 0, 0]) and np.allclose(curva[-1], [400, 300, 0])
    # el ducto llega al puerto final por su normal (eje Y): el ultimo tramo es vertical
    ultimos = curva[curva[:, 1] > 255.0]
    assert np.allclose(ultimos[:, 0], 400.0, atol=1e-6)
    # sin puertos todo es como antes
    capturado.clear()
    rep2 = cr.ducto_desde_entorno([0, 0, 0], [400, 300, 0], 60.0, 40.0, 2.0, parametros={"r_min_mm": 45.0})
    assert capturado["rectos_mm"] is None and "tramos_rectos_mm" not in rep2 and visto["inicio"] == [0, 0, 0]


def test_construir_curva_sin_simplificar_respeta_los_vertices_y_simplificando_sin_bloques_da_una_recta():
    ruta = [[0, 0, 0], [200, 0, 0], [200, 150, 0], [400, 150, 0]]
    _, simple = cr.construir_curva(ruta, [], 45.0, 20.0, 5.0)
    _, tal_cual = cr.construir_curva(ruta, [], 45.0, 20.0, 5.0, simplificar=False)
    assert simple["vertices_simplificada"] == 2 and tal_cual["vertices_simplificada"] == 4


def test_ducto_desde_entorno_con_ruta_dada_no_corre_el_astar_y_usa_esa_ruta(monkeypatch):
    modulo = types.ModuleType("ruteo_astar")
    modulo.calculo_ruta = lambda *a, **k: (_ for _ in ()).throw(AssertionError("el A* no debia correr"))
    modulo.listar_bloques, modulo.distancia_seguridad_mm = (lambda: []), (lambda tipo, valor: 0.0)
    monkeypatch.setitem(sys.modules, "ruteo_astar", modulo)
    capturado = {}
    monkeypatch.setattr(cr, "generar_ducto", lambda ruta, a, h, e, **kw: capturado.update(kw, ruta=ruta) or {"nombre": "x"})
    ruta = [[0, 0, 0], [200, 0, 0], [200, 150, 0], [400, 150, 0]]

    rep = cr.ducto_desde_entorno([0, 0, 0], [400, 150, 0], 60.0, 40.0, 2.0, parametros={"r_min_mm": 45.0}, ruta_dada=ruta)

    assert rep["estado"] == "ok" and rep["ruta_astar"]["longitud_mm"] == pytest.approx(200 + 150 + 200)
    curva = capturado["curva"]
    assert np.allclose(curva[0], [0, 0, 0]) and np.allclose(curva[-1], [400, 150, 0])
    assert d.radios_de_curvatura(curva).min() >= 45.0
    # la curva SIGUE el rodeo de la ruta dada (sin bloques, la simplificacion normal la volveria una recta)
    recta = np.array([400.0, 150.0, 0.0]) / np.linalg.norm([400.0, 150.0, 0.0])
    desvio = np.linalg.norm(curva - np.outer(curva @ recta, recta), axis=1)
    assert desvio.max() > 40.0
    assert rep["curva"]["vertices_simplificada"] == 4
    with pytest.raises(ValueError):
        cr.ducto_desde_entorno([0, 0, 0], [400, 150, 0], 60.0, 40.0, 2.0, ruta_dada=[[0, 0, 0]])
