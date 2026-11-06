"""
Pruebas de la ingesta de mallas (src/herramientas/ingesta_malla.py): STL/STEP por componente + manifiesto -> cajas.

Usan mallas sinteticas creadas en memoria (cajas, cilindro, una L), asi que no necesitan archivos de GM ni trimesh.
Ejecutar desde la raiz del proyecto:  pytest tests/test_ingesta_malla.py -v
"""
import json
import time

import numpy as np
import pytest

from herramientas import ingesta_malla as im
from herramientas import ruteo_astar


# ---- ayudas ---------------------------------------------------------------------------------------
def _L(grosor=100.0, largo=300.0):
    """Dos cajas pegadas que forman una L (cada una cerrada)."""
    return np.concatenate([im.malla_caja((0, 0, 0), (largo, grosor, grosor)),
                           im.malla_caja((0, grosor, 0), (grosor, largo, grosor))])


def _cubierto(cajas, puntos):
    """True por punto si cae dentro (o en el borde) de alguna caja."""
    p = np.asarray(puntos, dtype=float)
    dentro = np.zeros(len(p), dtype=bool)
    for x0, y0, z0, x1, y1, z1 in cajas:
        dentro |= np.all((p >= np.array([x0, y0, z0]) - 1e-9) & (p <= np.array([x1, y1, z1]) + 1e-9), axis=1)
    return dentro


def _escena(tmp_path, componentes, mallas, unidades="mm", puertos=None):
    for nombre, tri in mallas.items():
        im.escribir_stl(tmp_path / nombre, tri)
    manifiesto = {"version": 1, "nombre": "prueba", "unidades": unidades, "componentes": componentes}
    if puertos:
        manifiesto["puertos"] = puertos
    (tmp_path / "escena.json").write_text(json.dumps(manifiesto), encoding="utf-8")
    return tmp_path


COMPONENTES = [
    {"id": "motor", "nombre": "Motor", "archivo": "motor.stl", "tipo": "termico", "valor": 100},
    {"id": "bateria", "nombre": "Bateria", "archivo": "bateria.stl", "tipo": "electrico", "valor": 60},
    {"id": "tablero", "nombre": "Tablero", "archivo": "tablero.stl", "tipo": "neutro", "valor": 55},
]


def _mallas_basicas():
    return {
        "motor.stl": im.malla_caja((1000, -250, 250), (1600, 250, 750)),
        "bateria.stl": im.malla_cilindro((600, 300, 200), 100, 200, 24),
        "tablero.stl": im.malla_caja((1850, 0, 450), (2150, 400, 750)),
    }


# ---- STL: lectura y escritura ---------------------------------------------------------------------
def test_stl_binario_ida_y_vuelta(tmp_path):
    tri = im.malla_cilindro((0, 0, 0), 10, 20, 12)
    im.escribir_stl(tmp_path / "c.stl", tri)
    assert np.allclose(im.leer_stl(tmp_path / "c.stl"), tri, atol=1e-4)


def test_stl_ascii_se_lee(tmp_path):
    (tmp_path / "a.stl").write_text(
        "solid t\n facet normal 0 0 1\n  outer loop\n   vertex 0 0 0\n   vertex 1 0 0\n   vertex 0 1 0\n"
        "  endloop\n endfacet\nendsolid t\n", encoding="utf-8")
    tri = im.leer_stl(tmp_path / "a.stl")
    assert tri.shape == (1, 3, 3) and tri[0, 1].tolist() == [1.0, 0.0, 0.0]


def test_stl_corrupto_o_vacio_da_error_claro(tmp_path):
    (tmp_path / "x.stl").write_bytes(b"esto no es un stl")
    (tmp_path / "v.stl").write_bytes(b"")
    for nombre in ("x.stl", "v.stl"):
        with pytest.raises(im.IngestaError):
            im.leer_stl(tmp_path / nombre)


def test_stl_con_nan_se_rechaza(tmp_path):
    tri = im.malla_caja((0, 0, 0), (1, 1, 1))
    tri[0, 0, 0] = np.nan
    im.escribir_stl(tmp_path / "n.stl", tri)
    with pytest.raises(im.IngestaError):
        im.leer_stl(tmp_path / "n.stl")


# ---- propiedades de la malla ------------------------------------------------------------------------
def test_caja_es_cerrada_y_su_volumen_es_el_esperado():
    info = im.info_malla(im.malla_caja((0, 0, 0), (100, 60, 40)))
    assert info["cerrada"] and info["n_triangulos"] == 12
    assert info["volumen_mm3"] == pytest.approx(240000.0)
    assert info["bbox_min"] == [0.0, 0.0, 0.0] and info["bbox_max"] == [100.0, 60.0, 40.0]


def test_cilindro_es_cerrado_y_su_volumen_se_parece_al_real():
    info = im.info_malla(im.malla_cilindro((0, 0, 0), 50, 100, 96))
    assert info["cerrada"]
    assert info["volumen_mm3"] == pytest.approx(np.pi * 50 ** 2 * 100, rel=0.01)


def test_quitar_un_triangulo_la_hace_abierta_y_sin_volumen():
    info = im.info_malla(im.malla_caja((0, 0, 0), (10, 10, 10))[:-1])
    assert not info["cerrada"] and info["volumen_mm3"] is None


# ---- voxelizacion y cajas -------------------------------------------------------------------------
def test_una_caja_se_funde_en_una_sola_caja_igual_a_la_original():
    cajas, meta = im.malla_a_cajas(im.malla_caja((0, 0, 0), (100, 60, 40)), 10.0)
    assert cajas == [(0.0, 0.0, 0.0, 100.0, 60.0, 40.0)] and meta["metodo"] == "voxel"


def test_las_cajas_cubren_la_pieza_es_conservador():
    rng = np.random.default_rng(0)
    cil = im.malla_cilindro((0, 0, 0), 100, 200, 48)
    cajas, _ = im.malla_a_cajas(cil, 12.0)
    # puntos dentro del cilindro (con un pelo de margen contra los facetados)
    r = 95 * np.sqrt(rng.random(2000))
    t = rng.random(2000) * 2 * np.pi
    pts = np.stack([r * np.cos(t), r * np.sin(t), rng.random(2000) * 200], axis=1)
    assert _cubierto(cajas, pts).all()


def test_las_cajas_no_se_salen_de_la_caja_envolvente():
    cil = im.malla_cilindro((10, 20, 30), 40, 60, 24)
    cajas, _ = im.malla_a_cajas(cil, 7.0)
    lo, hi = cil.reshape(-1, 3).min(axis=0), cil.reshape(-1, 3).max(axis=0)
    for c in cajas:
        assert np.all(np.array(c[:3]) >= lo - 1e-9) and np.all(np.array(c[3:]) <= hi + 1e-9)


def test_una_L_deja_libre_el_hueco_que_una_caja_envolvente_taparia():
    cajas, _ = im.malla_a_cajas(_L(), 10.0)
    hueco = [[250.0, 250.0, 50.0]]       # dentro del rectangulo envolvente 300x300, pero fuera de la L
    pieza = [[50.0, 50.0, 50.0], [250.0, 50.0, 50.0], [50.0, 250.0, 50.0]]
    assert not _cubierto(cajas, hueco).any()
    assert _cubierto(cajas, pieza).all()


def test_un_tubo_hueco_cerrado_se_rellena_igual_que_un_solido_porque_su_interior_no_se_alcanza():
    exterior = im.malla_caja((0, 0, 0), (100, 100, 100))
    interior = im.malla_caja((30, 30, 30), (70, 70, 70))[:, ::-1]  # cavidad interna (normales invertidas)
    cajas, meta = im.malla_a_cajas(np.concatenate([exterior, interior]), 10.0)
    assert meta["metodo"] == "voxel" and cajas == [(0.0, 0.0, 0.0, 100.0, 100.0, 100.0)]


def test_malla_abierta_usa_la_caja_envolvente_y_avisa():
    abierta = im.malla_caja((0, 0, 0), (100, 100, 100))[:-2]  # le falta una cara
    cajas, meta = im.malla_a_cajas(abierta, 10.0)
    assert meta["metodo"] == "aabb" and cajas == [(0.0, 0.0, 0.0, 100.0, 100.0, 100.0)]
    assert any("ABIERTA" in a for a in meta["advertencias"])


def test_una_pieza_mas_delgada_que_el_paso_no_desaparece():
    lamina = im.malla_caja((0, 0, 0), (200, 200, 1))
    cajas, _ = im.malla_a_cajas(lamina, 20.0)
    assert cajas and _cubierto(cajas, [[100.0, 100.0, 0.5]]).all()


def test_paso_absurdamente_chico_da_error_en_vez_de_agotar_la_memoria():
    with pytest.raises(im.IngestaError):
        im.malla_a_cajas(im.malla_caja((0, 0, 0), (10000, 10000, 10000)), 1.0)


# ---- manifiesto -----------------------------------------------------------------------------------
def test_escena_basica_produce_bloques_en_el_formato_de_fer(tmp_path):
    carpeta = _escena(tmp_path, COMPONENTES, _mallas_basicas())
    bloques, informe = im.escena_a_bloques(carpeta)
    assert bloques and informe["n_cajas_total"] == len(bloques)
    for b in bloques:
        ruteo_astar._normalizar_bloque(b)  # el validador de Fer los acepta tal cual
    assert {b["nombre"] for b in bloques} == {"Motor", "Bateria", "Tablero"}
    assert len({b["id"] for b in bloques}) == len(bloques)  # ids unicos
    assert all(b["valor"] == 0.0 for b in bloques if b["tipo"] == "neutro")  # neutro siempre 0, aunque el manifiesto diga 55


def test_el_informe_cuantifica_la_sobreestimacion_y_los_triangulos(tmp_path):
    carpeta = _escena(tmp_path, COMPONENTES, _mallas_basicas())
    _, informe = im.escena_a_bloques(carpeta)
    motor = next(c for c in informe["componentes"] if c["id"] == "motor")
    assert motor["n_triangulos"] == 12 and motor["malla_cerrada"] and motor["metodo"] == "voxel"
    assert motor["sobreestimacion"] == pytest.approx(1.0, abs=1e-6)  # una caja se recorta a su envolvente
    bateria = next(c for c in informe["componentes"] if c["id"] == "bateria")
    assert bateria["sobreestimacion"] >= 1.0  # un cilindro siempre se sobreestima, nunca se subestima


def test_las_cajas_de_una_escena_cubren_cada_pieza(tmp_path):
    mallas = _mallas_basicas()
    carpeta = _escena(tmp_path, COMPONENTES, mallas)
    bloques, _ = im.escena_a_bloques(carpeta)
    cajas = [(b["centro"]["x"] - b["size"]["x"] / 2, b["centro"]["y"] - b["size"]["y"] / 2, b["centro"]["z"] - b["size"]["z"] / 2,
              b["centro"]["x"] + b["size"]["x"] / 2, b["centro"]["y"] + b["size"]["y"] / 2, b["centro"]["z"] + b["size"]["z"] / 2)
             for b in bloques]
    for tri in mallas.values():
        vertices = tri.reshape(-1, 3)
        assert _cubierto(cajas, vertices).all()


def test_max_cajas_se_respeta_aumentando_el_paso(tmp_path):
    carpeta = _escena(tmp_path, COMPONENTES, {**_mallas_basicas(), "motor.stl": _L(100, 600) + np.array([1000, 0, 250])})
    _, fino = im.escena_a_bloques(carpeta, paso_mm=10, max_cajas=10_000)
    _, grueso = im.escena_a_bloques(carpeta, paso_mm=10, max_cajas=3)
    assert grueso["n_cajas_total"] <= max(3, 3 + 0) and grueso["paso_mm"] > fino["paso_mm"]


def test_unidades_en_metros_se_escalan(tmp_path):
    chico = {n: t / 1000.0 for n, t in _mallas_basicas().items()}  # la misma escena, dibujada en metros
    carpeta = _escena(tmp_path, COMPONENTES, chico, unidades="m")
    _, informe = im.escena_a_bloques(carpeta)
    motor = next(c for c in informe["componentes"] if c["id"] == "motor")
    assert motor["bbox_min_mm"] == pytest.approx([1000, -250, 250], abs=0.1)
    assert not informe["advertencias"]


def test_si_no_declaras_unidades_y_la_escena_parece_un_juguete_se_avisa(tmp_path):
    chico = {n: t / 1000.0 for n, t in _mallas_basicas().items()}
    carpeta = _escena(tmp_path, COMPONENTES, chico, unidades="mm")  # olvido: estaban en metros
    _, informe = im.escena_a_bloques(carpeta)
    assert any("unidades" in a for a in informe["advertencias"])


def test_puerto_dentro_de_un_componente_se_avisa(tmp_path):
    puertos = {"inicio": {"punto": [1300, 0, 500]}, "fin": {"punto": [2400, 0, 500]}}  # el inicio esta dentro del motor
    carpeta = _escena(tmp_path, COMPONENTES, _mallas_basicas(), puertos=puertos)
    _, informe = im.escena_a_bloques(carpeta)
    assert any("inicio" in a and "DENTRO" in a for a in informe["advertencias"])
    assert not any("'fin'" in a for a in informe["advertencias"])


@pytest.mark.parametrize("cambio", [
    {"tipo": "radiactivo"},
    {"archivo": "no_existe.stl"},
    {"archivo": "../motor.stl"},
    {"archivo": "C:\\motor.stl"},
    {"archivo": "motor.obj"},
    {"valor": -5},
    {"valor": "mucho"},
    {"id": ""},
])
def test_manifiesto_invalido_da_error_claro(tmp_path, cambio):
    comp = [dict(COMPONENTES[0], **cambio)]
    carpeta = _escena(tmp_path, comp, {"motor.stl": im.malla_caja((0, 0, 0), (10, 10, 10))})
    with pytest.raises(im.IngestaError):
        im.cargar_manifiesto(carpeta)


def test_ids_repetidos_o_sin_manifiesto_o_json_roto_dan_error(tmp_path):
    comp = [COMPONENTES[0], dict(COMPONENTES[1], id="motor")]
    carpeta = _escena(tmp_path, comp, {"motor.stl": im.malla_caja((0, 0, 0), (10, 10, 10)),
                                       "bateria.stl": im.malla_caja((20, 0, 0), (30, 10, 10))})
    with pytest.raises(im.IngestaError):
        im.cargar_manifiesto(carpeta)
    vacia = tmp_path / "vacia"
    vacia.mkdir()
    with pytest.raises(im.IngestaError):
        im.cargar_manifiesto(vacia)
    (vacia / "escena.json").write_text("{no es json", encoding="utf-8")
    with pytest.raises(im.IngestaError):
        im.cargar_manifiesto(vacia)


def test_tipo_con_acento_y_mayusculas_se_acepta(tmp_path):
    comp = [dict(COMPONENTES[0], tipo="Térmico")]
    carpeta = _escena(tmp_path, comp, {"motor.stl": im.malla_caja((0, 0, 0), (10, 10, 10))})
    assert im.cargar_manifiesto(carpeta)["componentes"][0]["tipo"] == "termico"


# ---- de punta a punta con el A* de Fer ----------------------------------------------------------------
def test_guardar_entorno_y_que_el_astar_de_fer_lo_use(tmp_path, monkeypatch):
    carpeta = _escena(tmp_path, COMPONENTES, _mallas_basicas())
    bloques, _ = im.escena_a_bloques(carpeta)
    destino = im.guardar_entorno(bloques, tmp_path / "salida" / "entorno.json")
    assert json.loads(destino.read_text(encoding="utf-8"))["version"] == 1

    monkeypatch.setattr(ruteo_astar, "RUTA_ENTORNO", str(destino))
    assert len(ruteo_astar.listar_bloques()) == len(bloques)  # Fer los carga todos

    res = ruteo_astar.calculo_ruta([300, 0, 500], [2400, 0, 500])
    assert res["estado"] == "ok" and len(res["ruta"]) >= 2
    assert "Motor" in res["bloques_cercanos"]  # el motor esta en el camino directo y la ruta lo bordea


def test_la_ruta_pasa_por_el_hueco_de_la_L_en_vez_de_rodear_la_caja_envolvente(tmp_path, monkeypatch):
    # una L grande entre inicio y fin: con una sola caja envolvente la ruta tendria que rodearla toda
    comp = [dict(COMPONENTES[0], tipo="neutro")]
    carpeta = _escena(tmp_path, comp, {"motor.stl": _L(100, 1000)})
    bloques, _ = im.escena_a_bloques(carpeta, paso_mm=20)
    destino = im.guardar_entorno(bloques, tmp_path / "e.json")
    monkeypatch.setattr(ruteo_astar, "RUTA_ENTORNO", str(destino))
    res = ruteo_astar.calculo_ruta([800, 800, 50], [200, 200, 50])  # ambos puntos en el "hueco" de la L
    assert res["estado"] == "ok"
    assert res["longitud_mm"] < 1.2 * np.linalg.norm([600, 600])  # casi recta: no rodea nada


# ---- tiempo ---------------------------------------------------------------------------------------
def test_una_malla_de_decenas_de_miles_de_triangulos_se_procesa_en_segundos():
    cil = im.malla_cilindro((0, 0, 0), 200, 600, 6000)  # 24 000 triangulos
    t0 = time.time()
    cajas, _ = im.malla_a_cajas(cil, 10.0)
    assert cajas and time.time() - t0 < 10.0
