"""
Convierte una escena de mallas (STL/STEP + escena.json) en bloques para el A*

Usos:
    python scripts/cargar_escena.py ejemplo                      # solo convierte y muestra el informe
    python scripts/cargar_escena.py ejemplo --paso 15            # voxel de 15 mm
    python scripts/cargar_escena.py ejemplo --max-cajas 300      # permite mas cajas
    python scripts/cargar_escena.py ejemplo --activar            # ademas deja la escena como entorno activo

'nombre' es una carpeta de data/escenas/.
Escribe en la carpeta de la escena en entorno.json (los bloques) e informe.json (lo que se vera en pantalla).
--activar copia entorno.json a data/entorno/entorno.json, guardando antes una copia del entorno actual
(data/entorno/entorno_<fecha>_respaldo.json) para no perder el que habia.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from herramientas import ingesta_malla as im  # noqa: E402


def _carpeta(nombre: str) -> Path:
    directa = Path(nombre)
    if directa.is_dir():
        return directa
    return RAIZ / "data" / "escenas" / nombre


def _mostrar(informe: dict) -> None:
    print(f"\nEscena '{informe['escena']}'  unidades declaradas: {informe['unidades_declaradas']}  "
          f"extension: {informe['extension_mm']} mm")
    print(f"Voxel: {informe['paso_mm']} mm (pedido {informe['paso_inicial_mm']} mm)   "
          f"cajas: {informe['n_cajas_total']} (maximo {informe['max_cajas']})\n")
    print(f"{'componente':<14}{'tipo':<10}{'valor':>6}{'triang.':>9}{'cerrada':>9}{'metodo':>8}{'cajas':>7}{'sobreest.':>11}")
    for c in informe["componentes"]:
        sobre = f"{c['sobreestimacion']:.2f}x" if c["sobreestimacion"] else "n/d"
        print(f"{c['nombre'][:13]:<14}{c['tipo']:<10}{c['valor']:>6.0f}{c['n_triangulos']:>9}"
              f"{('si' if c['malla_cerrada'] else 'NO'):>9}{c['metodo']:>8}{c['n_cajas']:>7}{sobre:>11}")
        for a in c["advertencias"]:
            print(f"    ! {a}")
    if informe["puertos"]:
        print()
        for lado, p in informe["puertos"].items():
            extra = {k: v for k, v in p.items() if k != "punto"}
            print(f"Puerto {lado}: {p['punto']}  {extra if extra else ''}")
    if informe["advertencias"]:
        print()
        for a in informe["advertencias"]:
            print(f"ATENCION: {a}")
    print("\n'sobreest.' = volumen de las cajas / volumen de la malla (1.00x = exacto; las cajas SIEMPRE cubren la pieza).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("nombre", help="carpeta dentro de data/escenas/ (o ruta a una carpeta con escena.json)")
    ap.add_argument("--paso", type=float, default=None, help="tamano del voxel en mm (por defecto: extension/80)")
    ap.add_argument("--max-cajas", type=int, default=150, help="tope del total de cajas (por defecto 150)")
    ap.add_argument("--activar", action="store_true", help="deja la escena como entorno activo (con respaldo)")
    args = ap.parse_args()

    carpeta = _carpeta(args.nombre)
    try:
        bloques, informe = im.escena_a_bloques(carpeta, paso_mm=args.paso, max_cajas=args.max_cajas)
    except (im.IngestaError, FileNotFoundError) as e:
        print(f"ERROR: {e}")
        return 1

    im.guardar_entorno(bloques, carpeta / "entorno.json")
    (carpeta / "informe.json").write_text(json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8")
    _mostrar(informe)
    print(f"\nEscrito: {carpeta / 'entorno.json'}  y  {carpeta / 'informe.json'}")

    if not args.activar:
        print("Para usarla en el chat / en los scripts:  agrega --activar")
        return 0

    activo = RAIZ / "data" / "entorno" / "entorno.json"
    activo.parent.mkdir(parents=True, exist_ok=True)
    if activo.exists():
        respaldo = activo.parent / f"entorno_{datetime.now():%Y%m%d_%H%M%S}_respaldo.json"
        shutil.copy2(activo, respaldo)
        print(f"Entorno anterior guardado en {respaldo}")
    shutil.copy2(carpeta / "entorno.json", activo)
    print(f"Escena ACTIVADA: {activo}")
    p = informe["puertos"]
    if "inicio" in p and "fin" in p:
        a, b = p["inicio"], p["fin"]
        print("\nPara generar el ducto en el chat (el entorno se lee del archivo en cada llamada, no hace falta reiniciar):")
        print(f"  Genera un ducto de {a['punto']} a {b['punto']} con ancho {a.get('ancho_mm', '?')} mm, "
              f"alto {a.get('alto_mm', '?')} mm y espesor 2 mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
