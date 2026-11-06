"""
Pruebas de src/costos/campo_costo.py (features, maestro sintetico, dataset, entrenamiento, campo por lotes).
Numpy puro: el modelo es un doble de prueba, asi que NO hace falta TensorFlow.
Ejecutar desde la raiz del proyecto:  pytest tests -v
"""
import inspect
import sys

import numpy as np
import pytest

from costos import campo_costo as cc
from costos.motor_costos import FEATURES


def _bloque(nombre, centro, size, tipo="neutro", valor=0.0):
    return {"id": nombre, "nombre": nombre, "tipo": tipo, "valor": valor,
            "centro": dict(zip("xyz", centro)), "size": dict(zip("xyz", size))}


BLOQUES = [_bloque("Motor", [0, 0, 0], [100, 100, 100], "termico", 50.0),        # caja -50..50, zona: 50 mm
           _bloque("Bateria", [500, 0, 0], [100, 100, 100], "electrico", 30.0),   # caja 450..550, zona: 30 mm
           _bloque("Viga", [250, 400, 0], [100, 100, 100])]
ESCENA = cc.construir_escena(BLOQUES)


# ---- escena y features ---------------------------------------------------------------------------
def test_escena_separa_tipos_y_zonas_y_agrega_margen():
    assert len(ESCENA["termicos"]) == 1 and ESCENA["termicos"][0][1] == 50.0
    assert len(ESCENA["electricos"]) == 1 and ESCENA["electricos"][0][1] == 30.0
    assert len(ESCENA["todas"]) == 3
    assert ESCENA["caja"][:3] == (-350.0, -350.0, -350.0)        # -50 menos el margen de 300
    assert ESCENA["caja"][3:] == (850.0, 750.0, 350.0)


def test_caracteristicas_tienen_el_orden_y_las_distancias_correctas():
    raw = cc.caracteristicas([[100, 0, 0], [500, 0, 0], [0, 0, 0]], ESCENA)
    assert raw.shape == (3, len(FEATURES)) == (3, 7)
    # punto a 50 mm de la cara del motor: justo en el borde de su zona de seguridad (50 mm) -> NO esta "dentro"
    assert raw[0, 3] == pytest.approx(50.0) and raw[0, 6] == 0.0
    # punto dentro de la bateria: distancia 0 y dentro de su zona (zona electrica = 2)
    assert raw[1, 4] == 0.0 and raw[1, 6] == 2.0
    # punto dentro del motor: accesibilidad 0 y zona termica (= 1)
    assert raw[2, 5] == 0.0 and raw[2, 6] == 1.0


def test_distancias_se_topan_en_d_ref_y_sin_bloques_no_revienta():
    lejos = cc.caracteristicas([[5000, 5000, 5000]], ESCENA)
    assert lejos[0, 3] == lejos[0, 4] == lejos[0, 5] == cc.D_REF
    vacia = cc.construir_escena([])
    assert cc.caracteristicas([[1, 2, 3]], vacia)[0, 3:].tolist() == [cc.D_REF, cc.D_REF, cc.D_REF, 0.0]


def test_normalizar_deja_todo_en_cero_uno():
    lo, hi = np.array(ESCENA["caja"][:3]), np.array(ESCENA["caja"][3:])
    raw = cc.caracteristicas(np.random.default_rng(0).uniform(lo, hi, (500, 3)), ESCENA)
    X = cc.normalizar(raw, ESCENA)
    assert X.dtype == np.float32 and X.min() >= -1e-6 and X.max() <= 1.0 + 1e-6


def test_normalizar_no_extrapola_fuera_de_la_caja_de_entrenamiento():
    X = cc.normalizar(cc.caracteristicas([[99999, -99999, 5]], ESCENA), ESCENA)
    assert X[0, 0] == 1.0 and X[0, 1] == 0.0 and 0.0 <= X[0, 2] <= 1.0


# ---- maestro sintetico ---------------------------------------------------------------------------
def test_el_maestro_esta_en_cero_uno_y_baja_al_alejarse_del_motor():
    x = np.linspace(60, 450, 30)
    raw = cc.caracteristicas(np.column_stack([x, np.zeros(30), np.zeros(30)]), ESCENA)
    c = cc.costo_maestro(raw)
    assert c.min() >= 0.0 and c.max() <= 1.0
    assert c[0] > c[10]                    # mas cerca de lo termico -> mas caro (entre el motor y la bateria)


def test_el_maestro_cobra_mas_dentro_de_la_zona_de_seguridad_que_justo_afuera():
    dentro = cc.caracteristicas([[60, 0, 0]], ESCENA)     # a 10 mm del motor: zona termica
    fuera = cc.caracteristicas([[110, 0, 0]], ESCENA)     # a 60 mm: fuera de la zona
    assert dentro[0, 6] == 1.0 and fuera[0, 6] == 0.0
    assert cc.costo_maestro(dentro)[0] > cc.costo_maestro(fuera)[0] + cc.PESOS_MAESTRO["zona_termica"] * 0.9


def test_pesos_del_maestro_suman_uno():
    assert sum(cc.PESOS_MAESTRO.values()) == pytest.approx(1.0)


# ---- dataset -------------------------------------------------------------------------------------
def test_dataset_tiene_forma_rango_y_es_reproducible():
    X, y, raw = cc.generar_dataset(ESCENA, n=400, seed=3)
    assert X.shape == (400, 7) and y.shape == (400,) and raw.shape == (400, 7)
    assert y.min() >= 0 and y.max() <= 1
    X2, y2, _ = cc.generar_dataset(ESCENA, n=400, seed=3)
    assert np.array_equal(X, X2) and np.array_equal(y, y2)
    assert not np.array_equal(X, cc.generar_dataset(ESCENA, n=400, seed=4)[0])


def test_el_muestreo_pone_puntos_cerca_de_los_bloques():
    p = cc.muestrear(ESCENA, 1000, np.random.default_rng(0), fraccion_cerca=0.5)
    raw = cc.caracteristicas(p, ESCENA)
    assert (raw[:, 5] < 250).mean() > 0.4        # buena parte a menos de 250 mm de algun bloque


# ---- entrenar y metricas -------------------------------------------------------------------------
class ModeloPerfecto:
    """Doble de prueba: 'aprende' el maestro exacto (predice con la formula) y anota como lo entrenaron."""
    def __init__(self, escena):
        self.escena, self.fit_args = escena, None

    def fit(self, X, y, **kw):
        self.fit_args = (X.shape, y.shape, kw)

    def predict(self, X, batch_size=None, verbose=None):
        lo, hi = np.array(self.escena["caja"][:3]), np.array(self.escena["caja"][3:])
        raw = np.empty_like(X, dtype=float)
        raw[:, :3] = X[:, :3] * (hi - lo) + lo
        raw[:, 3:6] = X[:, 3:6] * cc.D_REF
        raw[:, 6] = X[:, 6] * 3.0
        return cc.costo_maestro(raw).reshape(-1, 1)          # (N,1) como Keras


def test_metricas_de_una_prediccion_perfecta_y_de_una_constante():
    y = np.array([0.0, 0.5, 1.0])
    assert cc.metricas(y, y) == {"mae": 0.0, "r2": 1.0}
    m = cc.metricas(y, np.full(3, y.mean()))
    assert m["r2"] == pytest.approx(0.0, abs=1e-9)


def test_entrenar_llama_fit_con_los_argumentos_de_keras_y_evalua_en_puntos_nuevos():
    modelo = ModeloPerfecto(ESCENA)
    res = cc.entrenar(modelo, ESCENA, n_train=300, n_test=200, epocas=7)
    forma_x, forma_y, kw = modelo.fit_args
    assert forma_x == (300, 7) and forma_y == (300,)
    assert kw["epochs"] == 7 and kw["batch_size"] == 256 and kw["validation_split"] == 0.1 and kw["verbose"] == 0
    assert res["r2"] == pytest.approx(1.0, abs=1e-3) and res["mae"] < 1e-3 and res["n_test"] == 200


# ---- el campo para el A* -------------------------------------------------------------------------
def test_campo_nn_devuelve_una_lista_de_costos_por_punto_en_cero_uno():
    campo = cc.CampoCostoNN(ModeloPerfecto(ESCENA), ESCENA, lote=64)      # lote chico: obliga a partir en pedazos
    pts = np.random.default_rng(1).uniform(-300, 800, (200, 3)).tolist()
    costos = campo(pts)
    assert isinstance(costos, list) and len(costos) == 200
    assert min(costos) >= 0 and max(costos) <= 1
    assert costos == pytest.approx(cc.CampoMaestro(ESCENA)(pts), abs=1e-6)
    assert campo([]) == []


def test_campo_nn_usa_un_solo_forward_por_lote_no_uno_por_punto():
    llamadas = []

    class Contador(ModeloPerfecto):
        def predict(self, X, batch_size=None, verbose=None):
            llamadas.append(len(X))
            return super().predict(X)

    cc.CampoCostoNN(Contador(ESCENA), ESCENA, lote=1000)(np.zeros((2500, 3)).tolist())
    assert llamadas == [1000, 1000, 500]


def test_campo_acepta_modelos_con_predict_simple_y_modelos_llamables():
    class SoloPredict:
        def predict(self, X):                 # sin batch_size/verbose (p. ej. scikit-learn)
            return np.full(len(X), 0.25)

    class Llamable:
        def __call__(self, X, training=False):
            return np.full((len(X), 1), 0.75)

    assert cc.CampoCostoNN(SoloPredict(), ESCENA)([[0, 0, 0]]) == [0.25]
    assert cc.CampoCostoNN(Llamable(), ESCENA)([[0, 0, 0]]) == [0.75]


def test_hash_de_bloques_cambia_si_se_mueve_un_bloque_y_no_con_el_orden():
    h = cc.hash_de_bloques(BLOQUES)
    assert h == cc.hash_de_bloques(list(reversed(BLOQUES)))
    movido = [dict(BLOQUES[0], centro={"x": 1.0, "y": 0.0, "z": 0.0})] + BLOQUES[1:]
    assert cc.hash_de_bloques(movido) != h


# ---- el gancho en el A* de Fer (se omite si todavia no tiene campo_costo) -------------------------
def _ruteo_con_gancho():
    try:
        import ruteo_astar as r
    except ImportError:
        pytest.skip("no hay ruteo_astar en el path")
    if "campo_costo" not in inspect.signature(r.calculo_ruta).parameters:
        pytest.skip("ruteo_astar.py todavia sin el parche campo_costo")
    return r


def test_astar_con_campo_evita_la_zona_cara_y_el_campo_se_llama_una_sola_vez(tmp_path):
    r = _ruteo_con_gancho()
    r.configurar_ruta_entorno(tmp_path / "e.json")
    r.limpiar_entorno()
    r.agregar_bloque("Muro", [500, 0, 0], [20, 200, 2000], "neutro")      # obliga a rodear por y>100 o y<-100
    llamadas = []

    def campo(puntos):
        llamadas.append(len(puntos))
        return [1.0 if p[1] > 0 else 0.0 for p in puntos]                 # el lado y>0 es "carisimo"

    sin = r.calculo_ruta([0, 0, 0], [1000, 0, 0], {"radio_ducto_mm": 10})
    con = r.calculo_ruta([0, 0, 0], [1000, 0, 0], {"radio_ducto_mm": 10, "lambda_campo": 20}, campo_costo=campo)

    assert con["estado"] == "ok" and len(llamadas) == 1
    assert max(p[1] for p in con["ruta"]) <= 1e-6 or min(p[1] for p in con["ruta"]) < -100   # rodeo por el lado barato
    assert con["longitud_mm"] >= 1000 - 1e-6 and sin["estado"] == "ok"   # sigue siendo >= la recta (heuristica valida)


def test_astar_con_campo_que_devuelve_mal_el_largo_da_error_claro_sin_lanzar(tmp_path):
    r = _ruteo_con_gancho()
    r.configurar_ruta_entorno(tmp_path / "e.json")
    r.limpiar_entorno()
    r.agregar_bloque("Muro", [500, 0, 0], [20, 200, 2000], "neutro")
    res = r.calculo_ruta([0, 0, 0], [1000, 0, 0], {"radio_ducto_mm": 10}, campo_costo=lambda pts: [0.0])
    assert res["estado"] == "error" and "valores" in res["mensaje"]
