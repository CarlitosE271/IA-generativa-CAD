"""
Dibuja en 3D los bloques y la ruta (usa matplotlib).

Para verlo en una ventana:
    res = r.calculo_ruta([0, 0, 500], [2300, 0, 500], {"radio_ducto_mm": 25})
    visualizar(res)

Para el agente (sin ventana, guarda un png y regresa un dict):
    out = visualizar_ruta([0, 0, 500], [2300, 0, 500])
    out["imagen"]   # ruta del png

Colores: termico rojo, electrico amarillo, neutro gris.
Las cajas punteadas son la distancia de seguridad.
"""

import base64
import os
import tempfile
import time

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import ruteo_astar as r

COLORES = {"termico": "#e4572e", "electrico": "#f2b705", "neutro": "#8d99ae"}
ETIQUETAS = {"termico": "Térmico", "electrico": "Eléctrico", "neutro": "Neutro"}
CARPETA = os.path.dirname(os.path.abspath(__file__))


def _esquinas(b, extra=0.0):
    c, s = b["centro"], b["size"]
    return (
        c["x"] - s["x"] / 2 - extra, c["y"] - s["y"] / 2 - extra, c["z"] - s["z"] / 2 - extra,
        c["x"] + s["x"] / 2 + extra, c["y"] + s["y"] / 2 + extra, c["z"] + s["z"] / 2 + extra,
    )


def _vertices(e):
    x0, y0, z0, x1, y1, z1 = e
    return [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]


# indices de los vertices que forman cada cara y cada arista
_CARAS = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]
_ARISTAS = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)]


def _dibujar(fig, resultado, bloques, inicio, fin, zonas, titulo):
    ruta = resultado["ruta"] if resultado else []
    if ruta:
        inicio = inicio if inicio is not None else ruta[0]
        fin = fin if fin is not None else ruta[-1]

    ax = fig.add_subplot(111, projection="3d")
    todos = []   # todos los puntos, para sacar los limites de los ejes

    for b in bloques:
        e = _esquinas(b)
        v = _vertices(e)
        todos.extend(v)
        color = COLORES.get(b["tipo"], "#8d99ae")

        caras = Poly3DCollection([[v[i] for i in cara] for cara in _CARAS],
                                 facecolor=color, edgecolor="black",
                                 linewidths=0.6, alpha=0.45)
        ax.add_collection3d(caras)
        ax.text((e[0] + e[3]) / 2, (e[1] + e[4]) / 2, e[5], b["nombre"],
                fontsize=8, ha="center", va="bottom")

        # caja punteada con la distancia de seguridad
        d = r.distancia_seguridad_mm(b["tipo"], b["valor"])
        if zonas and d > 0:
            vz = _vertices(_esquinas(b, extra=d))
            for i, j in _ARISTAS:
                ax.plot([vz[i][0], vz[j][0]], [vz[i][1], vz[j][1]],
                        [vz[i][2], vz[j][2]], color=color, linestyle=":",
                        linewidth=0.9, alpha=0.8)

    if ruta:
        xs, ys, zs = zip(*ruta)
        ax.plot(xs, ys, zs, color="#1d3557", linewidth=2.6, marker="o",
                markersize=3.5, zorder=10)
        todos.extend(ruta)
    if inicio is not None:
        ax.scatter(*inicio, color="green", s=70, depthshade=False, zorder=11)
        todos.append(tuple(inicio))
    if fin is not None:
        ax.scatter(*fin, color="blue", s=70, depthshade=False, zorder=11)
        todos.append(tuple(fin))

    if todos:
        rangos = []
        for k, nombre in enumerate("xyz"):
            lo = min(p[k] for p in todos)
            hi = max(p[k] for p in todos)
            pad = max((hi - lo) * 0.05, 1.0)
            getattr(ax, "set_%slim" % nombre)(lo - pad, hi + pad)
            rangos.append(hi - lo + 2 * pad)
        ax.set_box_aspect(rangos)   # para que no se deformen las proporciones

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    subtitulo = ""
    if resultado:
        subtitulo = "  |  %s, %.0f mm" % (resultado["estado"], resultado["longitud_mm"])
    ax.set_title(titulo + subtitulo)

    leyenda = [Patch(facecolor=COLORES[t], edgecolor="black", alpha=0.6, label=ETIQUETAS[t])
               for t in COLORES if any(b["tipo"] == t for b in bloques)]
    if ruta:
        leyenda.append(Line2D([0], [0], color="#1d3557", lw=2.6, label="Ruta"))
    if inicio is not None:
        leyenda.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="green",
                              markersize=9, label="Inicio"))
    if fin is not None:
        leyenda.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="blue",
                              markersize=9, label="Fin"))
    ax.legend(handles=leyenda, loc="upper left")
    fig.tight_layout()


def visualizar(resultado=None, bloques=None, inicio=None, fin=None, zonas=True,
               titulo="Ruteo A* en 3D", guardar_en=os.path.join(CARPETA, "ruteo_3d.png"),
               mostrar=True):
    # abre la ventana y guarda el png. Si bloques es None los lee del entorno.json
    if bloques is None:
        bloques = r.listar_bloques()
    fig = plt.figure(figsize=(11, 7.5))
    _dibujar(fig, resultado, bloques, inicio, fin, zonas, titulo)
    if guardar_en:
        fig.savefig(guardar_en, dpi=150)
        print("Imagen guardada en", guardar_en)
    if mostrar:
        plt.show()
    return fig


def _punto_o_none(p):
    try:
        return r._a_xyz(p, "punto")
    except ValueError:
        return None


def visualizar_ruta(inicio, fin, costo_ruta=None, ruta_png=None, zonas=True,
                    incluir_base64=False):
    """
    Calcula la ruta y guarda el dibujo en un png, sin abrir ventanas.
    Regresa el dict de calculo_ruta mas:
      "imagen": ruta del png (None si fallo el dibujo)
      "imagen_base64": el png en base64 (solo si incluir_base64=True)
      "error_imagen": el motivo, si fallo el dibujo
    Si ruta_png es None crea un nombre nuevo en data/salidas/.
    No lanza excepciones.
    """
    res = r.calculo_ruta(inicio, fin, costo_ruta)
    res["imagen"] = None
    try:
        bloques = r.listar_bloques()
        if ruta_png is None:
            ahora = time.time()
            nombre = "ruta_3d_%s_%03d.png" % (time.strftime("%Y%m%d_%H%M%S"),
                                              int(ahora * 1000) % 1000)
            ruta_png = os.path.join(CARPETA, "data", "salidas", nombre)
        ruta_png = os.path.abspath(ruta_png)
        os.makedirs(os.path.dirname(ruta_png), exist_ok=True)

        # Figure directo, sin pyplot, para no abrir ventanas ni acumular figuras
        fig = Figure(figsize=(11, 7.5))
        _dibujar(fig, res, bloques, _punto_o_none(inicio), _punto_o_none(fin),
                 zonas, "Ruteo A* en 3D")
        fig.savefig(ruta_png, dpi=150)
        res["imagen"] = ruta_png

        if incluir_base64:
            with open(ruta_png, "rb") as f:
                res["imagen_base64"] = base64.b64encode(f.read()).decode("ascii")
    except Exception as e:
        res["error_imagen"] = "%s: %s" % (type(e).__name__, e)
    return res


if __name__ == "__main__":
    # ejemplo con un entorno temporal, no toca tu entorno.json
    r.configurar_ruta_entorno(os.path.join(tempfile.mkdtemp(), "entorno.json"))
    r.agregar_bloque("Motor_principal", {"x": 1000, "y": 0, "z": 500},
                     {"x": 600, "y": 500, "z": 500}, "termico", 100)
    r.agregar_bloque("Bateria", {"x": 1800, "y": 300, "z": 200},
                     {"x": 400, "y": 300, "z": 200}, "electrico", 60)
    r.agregar_bloque("Chasis_viga", {"x": 1400, "y": -300, "z": 150},
                     {"x": 1200, "y": 60, "z": 300}, "neutro")
    r.agregar_bloque("Tablero", {"x": 300, "y": 0, "z": 600},
                     {"x": 250, "y": 900, "z": 500}, "neutro")

    res = r.calculo_ruta([0, 0, 500], [2300, 0, 500], {"radio_ducto_mm": 25})
    print("estado:", res["estado"], "| puntos:", len(res["ruta"]),
          "| longitud_mm:", res["longitud_mm"], "| cercanos:", res["bloques_cercanos"])
    visualizar(res)
