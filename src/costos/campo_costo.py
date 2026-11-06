"""
Campo de costo C_NN(x, y, z) para el A*: de los bloques del entorno a un costo en [0, 1] por punto.

Piezas (todas numpy puro salvo el modelo, que se le pasa ya construido):
  construir_escena      bloques -> cajas termicas / electricas / todas + caja de la escena
  caracteristicas       puntos (N,3) -> los 7 features crudos (mm) del plan: x, y, z, dist_thermal, dist_EMC,
                        accesibilidad, zona
  normalizar            features crudos -> entrada de la red (todo en ~[0, 1])
  costo_maestro         el "maestro" SINTETICO: de los features crudos a un costo en [0, 1]
  generar_dataset       puntos aleatorios + features + etiqueta del maestro
  entrenar              fit + metricas en puntos NUEVOS (MAE, R2)
  CampoCostoNN          llamable por LOTES: lista de puntos -> lista de costos (lo que usa el A*)
  obtener_campo         entrena una vez por escena y lo guarda (cache por hash de los bloques)

Honestidad intelectual (esto es lo que hay que decir ante el panel):
  - SIN datos de GM no hay etiquetas reales. El maestro (costo_maestro) es una formula que ESCRIBIMOS, con
    pesos arbitrarios. Entrenar la red sobre ella demuestra que la TUBERIA funciona (features -> red -> campo
    -> A* -> ruta), NO que el costo sea "correcto" fisicamente.
  - El valor de C_NN llega cuando las etiquetas sean otras: perdida de presion de un CFD, ruido medido, o
    puntaje de un ingeniero. Cambiar de maestro = cambiar UNA funcion (costo_maestro) o pasar otras etiquetas a
    entrenar(); el resto no se toca.
  - Se evalua por LOTES (un solo forward para toda la malla del A*), no punto por punto: Keras tarda ~1 ms por
    llamada suelta, y el A* puede tocar cientos de miles de celdas.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from costos.motor_costos import FEATURES

D_REF = 500.0  # mm: mas alla de esta distancia a un bloque, la distancia ya no cambia el costo

# Pesos del maestro SINTETICO (suman 1.0). Son un supuesto, no una norma: ver "Honestidad intelectual".
PESOS_MAESTRO = {
    "termico": 0.35,       # cercania a una fuente de calor
    "emc": 0.20,           # cercania a algo electrico (interferencia)
    "acceso": 0.15,        # poca holgura libre alrededor (mantenimiento/montaje)
    "zona_termica": 0.20,  # dentro de la zona de seguridad de un bloque termico
    "zona_emc": 0.10,      # dentro de la zona de seguridad de un bloque electrico
}
ESCALA_TERMICA_MM = 120.0  # que tan rapido cae el costo al alejarse de lo termico
ESCALA_EMC_MM = 80.0


# --------------------------------------------------------------------------
# Escena y features
# --------------------------------------------------------------------------
def _dseg_defecto(tipo, valor) -> float:
    # Debe coincidir con ruteo_astar.DISTANCIA_MM_POR_UNIDAD (1 mm por punto de valor, neutro = 0)
    return max(0.0, float(valor)) * {"termico": 1.0, "electrico": 1.0}.get(tipo, 0.0)


def _caja(b) -> tuple:
    c, s = b["centro"], b["size"]
    return (c["x"] - s["x"] / 2.0, c["y"] - s["y"] / 2.0, c["z"] - s["z"] / 2.0,
            c["x"] + s["x"] / 2.0, c["y"] + s["y"] / 2.0, c["z"] + s["z"] / 2.0)


def _dist_a_caja(puntos: np.ndarray, caja) -> np.ndarray:
    lo, hi = np.array(caja[:3]), np.array(caja[3:])
    d = np.maximum(np.maximum(lo - puntos, 0.0), puntos - hi)
    return np.linalg.norm(d, axis=1)


def construir_escena(bloques, margen_mm: float = 300.0, distancia_seguridad=None) -> dict:
    """bloques (formato de ruteo_astar) -> escena serializable:
    {"termicos": [(caja, dseg)], "electricos": [(caja, dseg)], "todas": [caja], "caja": (x0,y0,z0,x1,y1,z1)}"""
    dseg = distancia_seguridad or _dseg_defecto
    termicos, electricos, todas = [], [], []
    for b in bloques:
        caja = _caja(b)
        todas.append(caja)
        d = float(dseg(b["tipo"], b.get("valor", 0)))
        if b["tipo"] == "termico":
            termicos.append((caja, d))
        elif b["tipo"] == "electrico":
            electricos.append((caja, d))
    if todas:
        lo = np.min([c[:3] for c in todas], axis=0) - margen_mm
        hi = np.max([c[3:] for c in todas], axis=0) + margen_mm
    else:
        lo, hi = np.zeros(3), np.full(3, D_REF)
    return {"termicos": termicos, "electricos": electricos, "todas": todas,
            "caja": tuple(float(v) for v in (*lo, *hi))}


def _dist_minima(puntos: np.ndarray, cajas) -> np.ndarray:
    if not cajas:
        return np.full(len(puntos), D_REF)
    return np.minimum.reduce([_dist_a_caja(puntos, c) for c in cajas]).clip(max=D_REF)


def caracteristicas(puntos, escena: dict) -> np.ndarray:
    """(N,3) en mm -> (N,7) crudo, en el orden de motor_costos.FEATURES:
      x, y, z (mm) | dist_thermal, dist_EMC (mm, tope D_REF) | accesibilidad = distancia al bloque mas
      cercano de cualquier tipo (mm, tope D_REF) | zona = 1*(dentro de zona termica) + 2*(dentro de zona electrica)."""
    p = np.asarray(puntos, dtype=float).reshape(-1, 3)
    d_th = _dist_minima(p, [c for c, _ in escena["termicos"]])
    d_em = _dist_minima(p, [c for c, _ in escena["electricos"]])
    acceso = _dist_minima(p, escena["todas"])
    en_th = np.zeros(len(p), bool)
    for c, d in escena["termicos"]:
        en_th |= _dist_a_caja(p, c) < d
    en_em = np.zeros(len(p), bool)
    for c, d in escena["electricos"]:
        en_em |= _dist_a_caja(p, c) < d
    zona = en_th.astype(float) + 2.0 * en_em
    return np.column_stack([p, d_th, d_em, acceso, zona])


def normalizar(raw: np.ndarray, escena: dict) -> np.ndarray:
    """Features crudos -> entrada de la red en ~[0, 1] (las redes aprenden mal con mm sueltos)."""
    lo, hi = np.array(escena["caja"][:3]), np.array(escena["caja"][3:])
    X = np.empty_like(raw, dtype=np.float32)
    # clip: fuera de la caja con la que se entreno la red, el costo se queda constante en vez de extrapolar a ciegas
    X[:, :3] = np.clip((raw[:, :3] - lo) / np.maximum(hi - lo, 1e-9), 0.0, 1.0)
    X[:, 3:6] = np.clip(raw[:, 3:6] / D_REF, 0.0, 1.0)
    X[:, 6] = raw[:, 6] / 3.0
    return X


# --------------------------------------------------------------------------
# El maestro sintetico y el dataset
# --------------------------------------------------------------------------
def costo_maestro(raw: np.ndarray) -> np.ndarray:
    """Costo sintetico en [0, 1] a partir de los features crudos (ver PESOS_MAESTRO y el aviso del modulo)."""
    w = PESOS_MAESTRO
    d_th, d_em, acceso, zona = raw[:, 3], raw[:, 4], raw[:, 5], raw[:, 6]
    en_th = (zona % 2) >= 1
    en_em = zona >= 2
    c = (w["termico"] * np.exp(-d_th / ESCALA_TERMICA_MM)
         + w["emc"] * np.exp(-d_em / ESCALA_EMC_MM)
         + w["acceso"] * (1.0 - np.clip(acceso / D_REF, 0.0, 1.0))
         + w["zona_termica"] * en_th
         + w["zona_emc"] * en_em)
    return np.clip(c, 0.0, 1.0)


def muestrear(escena: dict, n: int, rng: np.random.Generator, fraccion_cerca: float = 0.5) -> np.ndarray:
    """n puntos: una parte uniforme en toda la escena y otra cerca de los bloques (donde el costo cambia)."""
    lo, hi = np.array(escena["caja"][:3]), np.array(escena["caja"][3:])
    n_cerca = int(n * fraccion_cerca) if escena["todas"] else 0
    puntos = [rng.uniform(lo, hi, size=(n - n_cerca, 3))]
    if n_cerca:
        cajas = np.array(escena["todas"])
        elegidas = cajas[rng.integers(0, len(cajas), n_cerca)]
        puntos.append(rng.uniform(elegidas[:, :3] - 250.0, elegidas[:, 3:] + 250.0))
    return np.vstack(puntos)


def generar_dataset(escena: dict, n: int = 20000, seed: int = 0):
    """-> (X normalizada (n,7), y (n,), raw (n,7))."""
    rng = np.random.default_rng(seed)
    raw = caracteristicas(muestrear(escena, n, rng), escena)
    return normalizar(raw, escena), costo_maestro(raw).astype(np.float32), raw


# --------------------------------------------------------------------------
# Entrenamiento y evaluacion
# --------------------------------------------------------------------------
def _predecir(modelo, X: np.ndarray) -> np.ndarray:
    """Un forward por lotes. Acepta un modelo Keras (predict / __call__) o cualquiera con predict(X)."""
    if hasattr(modelo, "predict"):
        try:
            y = modelo.predict(X, batch_size=8192, verbose=0)
        except TypeError:  # modelos con predict(X) a secas
            y = modelo.predict(X)
    else:
        y = modelo(X, training=False)
    return np.asarray(y, dtype=float).reshape(-1)


def metricas(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_real
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_real - y_real.mean()) ** 2))
    return {"mae": round(float(np.mean(np.abs(err))), 4),
            "r2": round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else float("nan")}


def entrenar(modelo, escena: dict, n_train: int = 20000, n_test: int = 5000, epocas: int = 40,
             seed: int = 0) -> dict:
    """Entrena y evalua en puntos NUEVOS (otra semilla): el R2 de entrenamiento no dice nada de generalizacion."""
    X, y, _ = generar_dataset(escena, n_train, seed)
    modelo.fit(X, y, epochs=epocas, batch_size=256, validation_split=0.1, shuffle=True, verbose=0)
    Xt, yt, _ = generar_dataset(escena, n_test, seed + 1)
    return {"n_train": n_train, "n_test": n_test, "epocas": epocas, **metricas(yt, _predecir(modelo, Xt))}


class CampoCostoNN:
    """Lo que recibe el A*: llamable con una LISTA de puntos [x, y, z] (mm) y devuelve una lista de costos
    en [0, 1], uno por punto, con un solo forward de la red."""

    def __init__(self, modelo, escena: dict, lote: int = 50000):
        self.modelo, self.escena, self.lote = modelo, escena, lote

    def __call__(self, puntos) -> list[float]:
        p = np.asarray(puntos, dtype=float).reshape(-1, 3)
        salida = []
        for i in range(0, len(p), self.lote):
            X = normalizar(caracteristicas(p[i:i + self.lote], self.escena), self.escena)
            salida.append(np.clip(_predecir(self.modelo, X), 0.0, 1.0))
        return np.concatenate(salida).tolist() if salida else []


class CampoMaestro:
    """El maestro sintetico como campo (sin red): sirve de referencia para medir que tan bien lo aproxima la red."""

    def __init__(self, escena: dict):
        self.escena = escena

    def __call__(self, puntos) -> list[float]:
        p = np.asarray(puntos, dtype=float).reshape(-1, 3)
        return costo_maestro(caracteristicas(p, self.escena)).tolist() if len(p) else []


# --------------------------------------------------------------------------
# Cache por escena (la red depende de las posiciones de los bloques: si se mueven, hay que reentrenar)
# --------------------------------------------------------------------------
def hash_de_bloques(bloques) -> str:
    base = sorted(({k: b[k] for k in ("nombre", "centro", "size", "tipo", "valor")} for b in bloques),
                  key=lambda d: json.dumps(d, sort_keys=True))
    return hashlib.sha1(json.dumps(base, sort_keys=True).encode()).hexdigest()[:10]


def obtener_campo(bloques, carpeta, epocas: int = 40, n_train: int = 20000, forzar: bool = False,
                  distancia_seguridad=None):
    """(campo, info). Si ya hay una red entrenada para ESTOS bloques en `carpeta`, la carga; si no, entrena una
    (Keras, import perezoso) y la guarda. info = {"origen": "cargado"|"entrenado", "metricas": {...}|None}."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    clave = hash_de_bloques(bloques)
    ruta = carpeta / f"c_nn_{clave}.keras"
    escena = construir_escena(bloques, distancia_seguridad=distancia_seguridad)

    import tensorflow as tf  # noqa: PLC0415 - import perezoso
    from costos.motor_costos import construir_modelo  # noqa: PLC0415

    if ruta.exists() and not forzar:
        return CampoCostoNN(tf.keras.models.load_model(str(ruta)), escena), {"origen": "cargado", "metricas": None, "archivo": ruta.name}
    modelo = construir_modelo()
    mets = entrenar(modelo, escena, n_train=n_train, epocas=epocas)
    modelo.save(str(ruta))
    return CampoCostoNN(modelo, escena), {"origen": "entrenado", "metricas": mets, "archivo": ruta.name}
