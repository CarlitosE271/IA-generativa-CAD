"""
Prueba manual de src/cad/ducto.py con CadQuery REAL (las pruebas de pytest usan un CadQuery falso,
asi que esta es la primera vez que el loft corre de verdad).

Uso, desde la raiz del proyecto:   python scripts/probar_ducto.py
Genera data/cad/ducto_prueba.step y .stl: abre el .stl (Visor 3D de Windows, FreeCAD, etc.) y revisa:
  1) que sea hueco y los dos extremos esten abiertos,
  2) que el rectangulo no se tuerza a lo largo de las curvas,
  3) que el espesor de pared se vea parejo.
"""
import json
import sys
from pathlib import Path

raiz = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(raiz / "src"))

from cad.ducto import generar_ducto  # noqa: E402

# Ruta de ejemplo en mm, con una vuelta horizontal y una subida (parecida a lo que sacaria el A*).
RUTA = [[0, 0, 0], [150, 0, 0], [150, 120, 0], [150, 120, 100], [300, 120, 100]]

try:  # si ya aprobaste parametros en el agente, se usan; si no, se valida solo lo que haya
    parametros = json.loads((raiz / "data" / "parametros_aprobados.json").read_text(encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError):
    parametros = {}

reporte = generar_ducto(
    RUTA, ancho_mm=40.0, alto_mm=25.0, espesor_mm=2.0,
    parametros=parametros, nombre="ducto_prueba",
)
print(json.dumps(reporte, indent=2, ensure_ascii=False))
