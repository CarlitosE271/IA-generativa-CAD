"""
Motor de costos: Un coord-MLP en TensorFlow/Keras que, dado un punto (x, y, z) del entorno mas unos features de
ingenieria, predice un costo en [0,1] de pasar el ducto por ahi. Ese costo es lo que el
solver de ruta (A* o RRT*) va a sumar dentro de:

    f(n) = g(n) + h(n) + lambda * C_NN(n)

Solo la arquitectura del modelo y una funcion que lo evalua en un punto, para probar que el cableado es 
correcto (orden de los 7 features,forma del tensor de entrada, como se extrae el escalar de salida). 

Junto con el solver A* se valida el costo viendo la ruta resultante,
no a ciegas).

Los 7 features de entrada:
  x, y, z            -> posicion del punto en el entorno
  dist_thermal       -> distancia (SDF) a la fuente de calor mas cercana (ej. el motor)
  dist_EMC           -> distancia (SDF) al componente electrico/EMC mas cercano (ej. arnes)
  accesibilidad      -> que tan accesible es la zona para mantenimiento (a definir)
  zona               -> que zona del vehiculo es (a definir, probablemente categorica)
Las distancias SDF se calculan con trimesh.proximity.signed_distance contra la malla (esa
parte es preprocesamiento geometrico, no le toca a este modulo).

"""

from __future__ import annotations

import numpy as np

# El orden fijo de los features lo comparte construir_modelo() (para el shape de entrada) y
# predecir_costo()/_vector_de_entrada() (para armar el vector).
FEATURES = ("x", "y", "z", "dist_thermal", "dist_EMC", "accesibilidad", "zona")


def construir_modelo():
    """Arma el coord-MLP con una salida logica (sigmoid),
    compilado para regresion (adam / mse). Devuelve el modelo sin entrenar (pesos aleatorios)."""
    import tensorflow as tf  # import perezoso: es pesada y solo se usa aqui

    modelo = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(len(FEATURES),)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),  
        ]
    )
    modelo.compile(optimizer="adam", loss="mse")  # regresion, no clasificacion
    return modelo


def _vector_de_entrada(x, y, z, dist_thermal, dist_EMC, accesibilidad, zona) -> list[float]:
    """Arma el vector de 7 valores en el mismo orden que espera el modelo.
    Separada de predecir_costo para poder probar el orden sin necesitar TensorFlow real."""
    return [float(v) for v in (x, y, z, dist_thermal, dist_EMC, accesibilidad, zona)]


def predecir_costo(modelo, x, y, z, dist_thermal, dist_EMC, accesibilidad, zona) -> float:
    """Evalua C_NN en un punto. Se usa __call__ del modelo (modelo(entrada)) en vez de
    modelo.predict(entrada): predict() trae overhead pensado para lotes grandes (logging,
    callbacks), y el A* va a llamar esto una vez por nodo explorado del grid -- miles de
    llamadas por ruta. Cuando el volumen de puntos a evaluar sea grande, 
    conviene una version por lotes en lugar de esta."""
    vector = _vector_de_entrada(x, y, z, dist_thermal, dist_EMC, accesibilidad, zona)
    entrada = np.array([vector], dtype="float32")  # forma (1, 7): un lote de un solo punto
    salida = modelo(entrada, training=False)
    return float(np.asarray(salida).reshape(-1)[0])
