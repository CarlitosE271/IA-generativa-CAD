"""
Ficha tecnica de un ducto de referencia en formato STL, utilizando los valores de longitud, radios de curvatura, perfil de seccion y puertos.

Uso:
    python scripts/ficha_ducto.py ruta/primera_pieza.STL
    python scripts/ficha_ducto.py a.STL b.STL c.STL --paso 3 --espesor 1.6
    python scripts/ficha_ducto.py a.STL --salida data/referencia

Por cada STL escribe en <salida>/<nombre>/ :
    ficha.json   todas las cifras (lo que se compara contra el ducto que genere la IA)
    ficha.png    vistas con la linea central encontrada + perfil de seccion a lo largo del ducto

Sirve para comparar ductos entre si, no para dimensionar. Se asume que utiliza mm
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from herramientas import ficha_ducto as fd  # noqa: E402
from herramientas import ingesta_malla as im  # noqa: E402


def _a_json(ficha: dict) -> dict:
    return {k: v for k, v in ficha.items() if k != "_interno"}


def _figura(tri: np.ndarray, ficha: dict, ruta: Path, titulo: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = ficha["_interno"]["centros"]
    pts = tri.mean(axis=1)[:: max(1, len(tri) // 6000)]
    ext = np.array(ficha["extremos_mm"])
    fig = plt.figure(figsize=(15, 9))
    vistas = [(0, 1, "Planta (X-Y)"), (0, 2, "Perfil (X-Z)")]
    for k, (i, j, nombre) in enumerate(vistas):
        ax = fig.add_subplot(2, 2, k + 1)
        ax.scatter(pts[:, i], pts[:, j], s=0.4, c="0.65")
        ax.plot(c[:, i], c[:, j], "r-", lw=1.6, label="linea central")
        ax.scatter(ext[:, i], ext[:, j], c="b", s=30, zorder=5, label="extremos del voxel")
        ax.set_aspect("equal")
        ax.set_title(nombre)
        ax.set_xlabel("mm")
        if k == 0:
            ax.legend(loc="upper right", fontsize=8)
    perfil = ficha["perfil"]
    s = [f["s_mm"] for f in perfil]
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(s, [f["ancho_int_mm"] for f in perfil], label="ancho interior")
    ax.plot(s, [f["alto_int_mm"] for f in perfil], label="alto interior")
    ax.set_xlabel("distancia a lo largo de la linea central (mm)")
    ax.set_ylabel("mm")
    ax.legend(fontsize=8)
    ax.set_title("Dimensiones de la seccion")
    ax.grid(alpha=0.3)
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(s, [f["area_int_mm2"] for f in perfil], color="tab:green", label="area interior")
    ax.set_xlabel("distancia a lo largo de la linea central (mm)")
    ax.set_ylabel("mm2")
    ax2 = ax.twinx()
    r = ficha["_interno"]["radios"]
    sa = ficha["_interno"]["s_arco"]
    ok = np.isfinite(r) & (r < 600)
    ax2.plot(sa[ok], r[ok], color="tab:orange", lw=1, label="radio de curvatura")
    ax2.set_ylabel("radio de curvatura (mm, <600)")
    ax.set_title("Area y radio de curvatura")
    ax.grid(alpha=0.3)
    fig.suptitle(f"{titulo}   largo {ficha['longitud_linea_central_mm']} mm   "
                 f"R min ~{ficha['radio_curvatura_minimo_mm']} mm   (voxel {ficha['paso_mm']} mm)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def _resumen(nombre: str, f: dict) -> None:
    print(f"\n=== {nombre} ===")
    print(f"  largo de la linea central : {f['longitud_linea_central_mm']} mm")
    print(f"  radio de curvatura minimo : {f['radio_curvatura_minimo_mm']} mm  "
          f"(rango segun suavizado {f['radio_curvatura_minimo_rango_mm']}; p10 {f['radio_curvatura_p10_mm']})")
    a = f["area_int_mm2"]
    print(f"  area interior del cuerpo  : min {a['min']:.0f} / mediana {a['mediana']:.0f} / max {a['max']:.0f} mm2")
    print(f"  relacion ancho/alto       : mediana {f['relacion_aspecto']['mediana']}  max {f['relacion_aspecto']['max']}")
    print(f"  cambio max de area / 50 mm: {f['cambio_max_de_area_por_50mm']}")
    for lado, p in f["puertos"].items():
        s = p["seccion_20_a_80mm"]
        print(f"  puerto {lado}: punto {p['punto_mm']}  normal {p['normal_hacia_afuera']}  "
              f"seccion {s['ancho_int_mm']} x {s['alto_int_mm']} mm  (area {s['area_int_mm2']:.0f}, D_h {s['diam_hidraulico_mm']})")
    for a_ in f["advertencias"]:
        print(f"  ATENCION: {a_}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("stl", nargs="+", help="uno o varios archivos .stl")
    ap.add_argument("--paso", type=float, default=3.0, help="voxel en mm (por defecto 3)")
    ap.add_argument("--espesor", type=float, default=None, help="espesor de pared en mm (por defecto: se estima o 1.6)")
    ap.add_argument("--salida", default=str(RAIZ / "data" / "referencia"), help="carpeta de salida")
    args = ap.parse_args()

    codigo = 0
    for ruta in args.stl:
        ruta = Path(ruta)
        try:
            tri = im.leer_stl(ruta)
            ficha = fd.ficha_de_malla(tri, paso_mm=args.paso, espesor_pared_mm=args.espesor)
        except (im.IngestaError, FileNotFoundError, OSError) as e:
            print(f"ERROR en {ruta.name}: {e}")
            codigo = 1
            continue
        carpeta = Path(args.salida) / ruta.stem
        carpeta.mkdir(parents=True, exist_ok=True)
        (carpeta / "ficha.json").write_text(json.dumps(_a_json(ficha), indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            _figura(tri, ficha, carpeta / "ficha.png", ruta.stem)
        except ImportError:
            print("(matplotlib no esta instalado: se omite la figura)")
        _resumen(ruta.stem, ficha)
        print(f"  escrito: {carpeta / 'ficha.json'}  y  {carpeta / 'ficha.png'}")
    return codigo


if __name__ == "__main__":
    sys.exit(main())
