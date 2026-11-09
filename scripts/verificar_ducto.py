"""
Mide el ducto que generó la IA (STL exportado) y los compara

Uso:
    python scripts/verificar_ducto.py data/cad/ducto_ab12cd.stl
    python scripts/verificar_ducto.py data/cad/ducto_ab12cd.stl --referencia data/referencia/segunda_pieza/ficha.json
    python scripts/verificar_ducto.py ducto.stl --r-min 60 --area-min 3000 --area-max 8000 --aspecto-max 3.5        # Sin --r-min/--area-min/... se usan los parametros aprobados (data/parametros_aprobados.json).


Escribe en data/verificacion/<nombre>/ :  verificacion.json  y  ficha.png 

Se mide la malla exportada, no lo que se pidio. 
Precision: voxel de 3 mm
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from herramientas import ficha_ducto as fd  # noqa: E402
from herramientas import ingesta_malla as im  # noqa: E402
from herramientas import metricas_ducto as mt  # noqa: E402


def _parametros(args) -> dict:
    ruta = RAIZ / "data" / "parametros_aprobados.json"
    p = {}
    if ruta.exists():
        try:
            p = json.loads(ruta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            print(f"AVISO: no se pudo leer {ruta}; se usan solo los valores dados por linea de comandos.")
    for clave, valor in (("r_min_mm", args.r_min), ("area_min_mm2", args.area_min), ("area_max_mm2", args.area_max),
                         ("aspect_ratio_max", args.aspecto_max)):
        if valor is not None:
            p[clave] = valor
    return p


def _figura(tri, ficha, destino: Path, titulo: str) -> None:
    spec = importlib.util.spec_from_file_location("ficha_cli", RAIZ / "scripts" / "ficha_ducto.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    modulo._figura(tri, ficha, destino, titulo)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("stl", help="STL del ducto generado")
    ap.add_argument("--referencia", help="ficha.json de un ducto de referencia (de scripts/ficha_ducto.py)")
    ap.add_argument("--paso", type=float, default=3.0, help="voxel en mm (por defecto 3)")
    ap.add_argument("--espesor", type=float, default=None, help="espesor de pared en mm (por defecto se estima)")
    ap.add_argument("--r-min", type=float, default=None)
    ap.add_argument("--area-min", type=float, default=None)
    ap.add_argument("--area-max", type=float, default=None)
    ap.add_argument("--aspecto-max", type=float, default=None)
    ap.add_argument("--salida", default=str(RAIZ / "data" / "verificacion"))
    args = ap.parse_args()

    ruta = Path(args.stl)
    try:
        tri = im.leer_stl(ruta)
        ficha = fd.ficha_de_malla(tri, paso_mm=args.paso, espesor_pared_mm=args.espesor)
    except (im.IngestaError, FileNotFoundError, OSError) as e:
        print(f"ERROR en {ruta.name}: {e}")
        return 1

    params = _parametros(args)
    cumple = mt.cumplimiento(ficha, params)
    salida = {"ducto": ruta.name, "parametros_usados": params, "resumen": mt.resumen(ficha), "cumplimiento": cumple,
              "advertencias": ficha["advertencias"]}
    print(f"\n=== {ruta.name} ===  largo {ficha['longitud_linea_central_mm']} mm, R min ~{ficha['radio_curvatura_minimo_mm']} mm")
    print(f"Parametros usados: {params if params else 'ninguno (todo saldra no_verificado)'}\n")
    for nombre, c in cumple.items():
        if nombre == "nota":
            continue
        valor = next((v for k, v in c.items() if k.startswith("medido")), None)
        limite = next((v for k, v in c.items() if k.startswith("limite")), None)
        print(f"  {nombre:<18} medido {valor}   limite {limite}   ->  {c['cumple']}")
    print(f"\n  {cumple['nota']}")
    for a in ficha["advertencias"]:
        print(f"  ATENCION: {a}")

    if args.referencia:
        try:
            ref = json.loads(Path(args.referencia).read_text(encoding="utf-8"))
            comp = mt.comparar(ficha, ref)
        except (OSError, ValueError, KeyError) as e:
            print(f"ERROR leyendo la referencia: {e}")
            return 1
        salida["comparacion"] = comp
        print(f"\nComparacion contra {Path(args.referencia).parent.name}  (razon = generada / referencia):")
        for k, v in comp.items():
            print(f"  {k:<34} ref {v['referencia']!s:>9}   gen {v['generada']!s:>9}   razon {v['razon']}")

    carpeta = Path(args.salida) / ruta.stem
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "verificacion.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        _figura(tri, ficha, carpeta / "ficha.png", ruta.stem)
    except ImportError:
        print("(matplotlib no esta instalado: se omite la figura)")
    print(f"\nEscrito: {carpeta / 'verificacion.json'}  y  {carpeta / 'ficha.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
