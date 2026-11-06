"""
Crea una escena de EJEMPLO en data/escenas/ejemplo/ (mallas STL sinteticas + escena.json).

Sirve para dos cosas:
  1. Probar la ingesta de mallas sin esperar la geometria de Maikol:
        python scripts/crear_escena_ejemplo.py
        python scripts/cargar_escena.py ejemplo
  2. Ser el MODELO del formato que Maikol debe entregar (un archivo por componente + escena.json).

Uso:  python scripts/crear_escena_ejemplo.py [--forzar]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from herramientas import ingesta_malla as im  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--forzar", action="store_true", help="sobrescribe la escena de ejemplo si ya existe")
    args = ap.parse_args()

    carpeta = RAIZ / "data" / "escenas" / "ejemplo"
    if (carpeta / "escena.json").exists() and not args.forzar:
        print(f"Ya existe {carpeta}. Usa --forzar para recrearla.")
        return 1
    carpeta.mkdir(parents=True, exist_ok=True)

    mallas = {
        "motor.stl": im.malla_cilindro((1300, 0, 250), 260, 500, 48),              # cilindro: una caja lo sobreestimaria
        "bateria.stl": im.malla_cilindro((600, 300, 150), 110, 250, 32),
        "soporte.stl": np.concatenate([                                            # una L: el hueco debe quedar libre
            im.malla_caja((300, -400, 0), (900, -300, 300)),
            im.malla_caja((300, -400, 0), (400, 100, 300)),
        ]),
        "tablero.stl": im.malla_caja((1850, 0, 450), (2150, 400, 750)),
    }
    for nombre, tri in mallas.items():
        im.escribir_stl(carpeta / nombre, tri)

    manifiesto = {
        "version": 1,
        "nombre": "ejemplo",
        "unidades": "mm",
        "componentes": [
            {"id": "motor", "nombre": "Motor", "archivo": "motor.stl", "tipo": "termico", "valor": 100},
            {"id": "bateria", "nombre": "Bateria", "archivo": "bateria.stl", "tipo": "electrico", "valor": 60},
            {"id": "soporte", "nombre": "Soporte", "archivo": "soporte.stl", "tipo": "neutro", "valor": 0},
            {"id": "tablero", "nombre": "Tablero", "archivo": "tablero.stl", "tipo": "neutro", "valor": 0},
        ],
        "puertos": {
            "inicio": {"punto": [300, 0, 500], "ancho_mm": 50, "alto_mm": 20},
            "fin": {"punto": [2300, 0, 500], "ancho_mm": 50, "alto_mm": 20},
        },
    }
    (carpeta / "escena.json").write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Escena de ejemplo creada en {carpeta}")
    print("Siguiente paso:  python scripts/cargar_escena.py ejemplo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
