"""
Prueba manual de src/cad/ducto.py con CadQuery.

Uso, desde la raiz del proyecto:   
    python scripts/probar_ducto.py

Genera data/cad/ducto_prueba.step y .stl
"""
import json
import sys
from pathlib import Path

raiz = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(raiz / "src"))

from cad.ducto import generar_ducto  # noqa: E402

# Ruta de ejemplo en mm, con una vuelta horizontal y una subida 
# Se parece a lo que sacaria el A*
RUTA = [[0, 0, 0], [150, 0, 0], [150, 120, 0], [150, 120, 100], [300, 120, 100]]

try:  # si el usuario ya aprobo los parametros en el agente, se usan
    parametros = json.loads((raiz / "data" / "parametros_aprobados.json").read_text(encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError):
    parametros = {}

reporte = generar_ducto(
    RUTA, ancho_mm=40.0, alto_mm=25.0, espesor_mm=2.0,
    parametros=parametros, nombre="ducto_prueba",
)
print(json.dumps(reporte, indent=2, ensure_ascii=False))
