"""
Entrena C_NN (Keras REAL) sobre el entorno de bloques y compara rutas del A* con y sin el campo de costo.

Uso, desde la raiz del proyecto:
  python scripts/entrenar_c_nn.py              
  python scripts/entrenar_c_nn.py 0 0 500  2300 0 500      # o los puntos x0 y0 z0 x1 y1 z1

Requiere el A* con el parche campo_costo (parches/ruteo_astar_con_campo_costo.py)
Genera data/modelos/c_nn_<hash>.keras (se reusa mientras no se muevan los bloques) y data/salidas/comparacion_c_nn.png.
"""
import inspect
import math
import sys
from pathlib import Path

import numpy as np

raiz = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(raiz / "src"))
sys.path.insert(0, str(raiz / "src" / "herramientas"))

import ruteo_astar as r  # noqa: E402
from costos import campo_costo as cc  # noqa: E402

r.configurar_ruta_entorno(raiz / "data" / "entorno" / "entorno.json")
bloques = r.listar_bloques()
if not bloques:
    sys.exit("El entorno esta vacio (data/entorno/entorno.json): coloca bloques primero.")
if "campo_costo" not in inspect.signature(r.calculo_ruta).parameters:
    sys.exit("Tu ruteo_astar.py todavia no tiene el parametro campo_costo.\n"
             "Aplica el parche: parches/ruteo_astar_con_campo_costo.py (o parche_campo_costo.diff) y vuelve a correr.")

argumentos = [float(a) for a in sys.argv[1:]]
INICIO, FIN = (argumentos[:3], argumentos[3:6]) if len(argumentos) == 6 else ([0, 0, 500], [2300, 0, 500])
RADIO = math.hypot(40.0, 25.0) / 2.0 

print(f"Entorno: {[b['nombre'] for b in bloques]}")
print("Entrenando C_NN con Keras (la primera vez tarda; despues se reusa)...")
campo_nn, info = cc.obtener_campo(bloques, raiz / "data" / "modelos", forzar=True,
                                  distancia_seguridad=r.distancia_seguridad_mm)
print(f"  {info['origen']} -> {info['archivo']}   R2={info['metricas']['r2']}  MAE={info['metricas']['mae']}  "
      f"(en {info['metricas']['n_test']} puntos nuevos)")
escena = campo_nn.escena
maestro = cc.CampoMaestro(escena)


def medir(res):
    p = np.array(res["ruta"], dtype=float)
    denso = [p[0]]
    for a, b in zip(p, p[1:]):
        k = max(1, int(np.linalg.norm(b - a) / 10))
        denso += [a + (b - a) * i / k for i in range(1, k + 1)]
    raw = cc.caracteristicas(np.array(denso), escena)
    return {"largo": res["longitud_mm"], "vertices": len(p), "term": raw[:, 3].min(), "elec": raw[:, 4].min(),
            "costo": float(cc.costo_maestro(raw).mean())}


VARIANTES = [("sin campo", None, 0.0), ("maestro  l=2", maestro, 2.0), ("C_NN     l=2", campo_nn, 2.0),
             ("C_NN     l=8", campo_nn, 8.0)]
resultados = {}
print(f"\nRuta {INICIO} -> {FIN}, radio del ducto {RADIO:.1f} mm")
print(f"{'variante':14} {'largo mm':>9} {'vert':>5} {'dist min termico':>17} {'dist min electrico':>19} {'costo medio (maestro)':>22}")
for nombre, campo, lam in VARIANTES:
    cfg = {"radio_ducto_mm": RADIO, **({"lambda_campo": lam} if campo else {})}
    res = r.calculo_ruta(INICIO, FIN, cfg, campo_costo=campo) if campo else r.calculo_ruta(INICIO, FIN, cfg)
    if res["estado"] != "ok":
        print(f"{nombre:14} {res['estado']}: {res['mensaje']}")
        continue
    m = medir(res)
    resultados[nombre] = res["ruta"]
    print(f"{nombre:14} {m['largo']:9.0f} {m['vertices']:5d} {m['term']:17.0f} {m['elec']:19.0f} {m['costo']:22.3f}")

try:  # figura vista desde arriba (x-y)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    colores = {"termico": "#e4572e", "electrico": "#f2b705", "neutro": "#8d99ae"}
    for b in bloques:
        c, s = b["centro"], b["size"]
        ax.add_patch(plt.Rectangle((c["x"] - s["x"] / 2, c["y"] - s["y"] / 2), s["x"], s["y"], alpha=0.5,
                                   color=colores[b["tipo"]]))
        ax.text(c["x"], c["y"], b["nombre"], ha="center", va="center", fontsize=8)
    for (nombre, ruta), estilo in zip(resultados.items(), ["k--", "g-", "b-", "r-"]):
        pts = np.array(ruta)
        ax.plot(pts[:, 0], pts[:, 1], estilo, lw=2, label=nombre)
    ax.set_aspect("equal"); ax.autoscale(); ax.legend(); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title("Rutas del A*: sin campo vs con C_NN (vista desde arriba)")
    salida = raiz / "data" / "salidas" / "comparacion_c_nn.png"
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=130, bbox_inches="tight")
    print("\nFigura:", salida)
except ImportError:
    print("\n(sin matplotlib: se omite la figura)")
