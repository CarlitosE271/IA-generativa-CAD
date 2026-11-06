"""
Pruebas de src/costos/motor_costos.py. NO usan TensorFlow real: este entorno de trabajo no
tuvo acceso a PyPI para instalarlo, asi que se simula con un modulo 'tensorflow' falso que
respeta la API documentada de Keras (Sequential, Input, Dense, compile). predecir_costo() se
prueba aparte con un modelo falso simplemente CALLABLE, porque su logica (armar el vector en
el orden correcto, darle la forma (1,7), sacar el escalar) no depende de que el modelo sea de
Keras de verdad.

Ejecutar desde la raiz del proyecto:  pytest tests -v
"""
import sys
import types

import numpy as np
import pytest

from costos import motor_costos as mc


# ---- construir_modelo(): arma la arquitectura del plan, con Keras simulado ----------------------
class _CapaFalsa:
    def __init__(self, tipo, **kwargs):
        self.tipo = tipo
        self.kwargs = kwargs


class _ModeloFalso:
    def __init__(self, capas):
        self.capas = capas
        self.compilado = None

    def compile(self, optimizer=None, loss=None):
        self.compilado = {"optimizer": optimizer, "loss": loss}


def _instalar_tensorflow_falso(monkeypatch):
    """Registra un modulo 'tensorflow' falso en sys.modules, con solo lo que motor_costos.py usa."""

    def Input(shape):
        return _CapaFalsa("Input", shape=shape)

    def Dense(units, activation=None):
        return _CapaFalsa("Dense", units=units, activation=activation)

    def Sequential(capas):
        return _ModeloFalso(capas)

    keras = types.SimpleNamespace(
        Sequential=Sequential,
        layers=types.SimpleNamespace(Input=Input, Dense=Dense),
    )
    modulo = types.ModuleType("tensorflow")
    modulo.keras = keras
    monkeypatch.setitem(sys.modules, "tensorflow", modulo)


def test_construir_modelo_arma_la_arquitectura_del_plan(monkeypatch):
    _instalar_tensorflow_falso(monkeypatch)

    modelo = mc.construir_modelo()

    tipos = [c.tipo for c in modelo.capas]
    assert tipos == ["Input", "Dense", "Dense", "Dense"]
    assert modelo.capas[0].kwargs == {"shape": (len(mc.FEATURES),)}
    assert modelo.capas[1].kwargs == {"units": 64, "activation": "relu"}
    assert modelo.capas[2].kwargs == {"units": 32, "activation": "relu"}
    assert modelo.capas[3].kwargs == {"units": 1, "activation": "sigmoid"}
    assert modelo.compilado == {"optimizer": "adam", "loss": "mse"}  # regresion, no clasificacion


def test_construir_modelo_usa_siete_features_de_entrada(monkeypatch):
    # El shape de entrada tiene que seguir a FEATURES, no un numero suelto hardcodeado.
    _instalar_tensorflow_falso(monkeypatch)
    modelo = mc.construir_modelo()
    assert modelo.capas[0].kwargs["shape"] == (7,)
    assert len(mc.FEATURES) == 7


# ---- _vector_de_entrada(): logica pura, sin TensorFlow --------------------------------------------
def test_vector_de_entrada_respeta_el_orden_de_features():
    vector = mc._vector_de_entrada(
        x=10, y=20, z=30, dist_thermal=1, dist_EMC=2, accesibilidad=3, zona=4
    )
    assert vector == [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0]
    assert len(vector) == len(mc.FEATURES)


def test_vector_de_entrada_convierte_a_float():
    # Por si algun feature llega como int, str numerico, etc.
    vector = mc._vector_de_entrada(
        x="1", y=2, z=3.5, dist_thermal=4, dist_EMC=5, accesibilidad=6, zona=7
    )
    assert vector == [1.0, 2.0, 3.5, 4.0, 5.0, 6.0, 7.0]
    assert all(isinstance(v, float) for v in vector)


# ---- predecir_costo(): con un modelo falso simplemente callable ----------------------------------
class _ModeloPrediceFalso:
    """Simula un modelo Keras YA CONSTRUIDO: solo necesita ser llamable como modelo(entrada)."""

    def __init__(self, valor):
        self.valor = valor
        self.ultima_entrada = None
        self.ultimo_training = None

    def __call__(self, entrada, training=False):
        self.ultima_entrada = entrada
        self.ultimo_training = training
        return np.array([[self.valor]], dtype="float32")


def test_predecir_costo_arma_el_vector_en_el_orden_correcto_y_la_forma_1x7():
    modelo = _ModeloPrediceFalso(0.42)

    costo = mc.predecir_costo(
        modelo, x=1, y=2, z=3, dist_thermal=4, dist_EMC=5, accesibilidad=6, zona=7
    )

    assert costo == pytest.approx(0.42)
    assert modelo.ultima_entrada.shape == (1, 7)
    assert list(modelo.ultima_entrada[0]) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert modelo.ultimo_training is False  # nunca se debe entrenar al evaluar un punto


def test_predecir_costo_devuelve_float_python_no_numpy():
    modelo = _ModeloPrediceFalso(0.7)
    costo = mc.predecir_costo(
        modelo, x=0, y=0, z=0, dist_thermal=0, dist_EMC=0, accesibilidad=0, zona=0
    )
    assert type(costo) is float
