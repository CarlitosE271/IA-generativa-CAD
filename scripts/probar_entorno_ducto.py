"""
Prueba de punta a punta con CadQuery REAL: bloques (entorno de Fer) -> A* -> curva con R_min exacto -> ducto CAD.

Uso, desde la raiz del proyecto:   python scripts/probar_entorno_ducto.py
Requiere ruteo_astar.py (y visualizar_ruteo.py para el PNG) en src/herramientas/.
Genera data/cad/ducto_entorno.step/.stl y data/salidas/ruta_ducto.png.
"""
import json
import sys
from pathlib import Path

raiz = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(raiz / "src"))
sys.path.insert(0, str(raiz / "src" / "herramientas"))  # para que 'import ruteo_astar' funcione tal cual

import ruteo_astar as r  # noqa: E402
from cad.curva_ruta import ducto_desde_entorno  # noqa: E402

# El entorno se guarda en data/ de TU proyecto (no junto al archivo de Fer)
r.configurar_ruta_entorno(raiz / "data" / "entorno" / "entorno.json")
r.limpiar_entorno()
r.agregar_bloque("Motor principal", [1200, 0, 500], [600, 500, 500], "termico", 100)
r.agregar_bloque("Bateria", [600, 300, 300], [300, 200, 200], "electrico", 60)
r.agregar_bloque("Chasis viga", [1800, -250, 200], [800, 100, 100], "neutro")
r.agregar_bloque("Tablero", [2000, 200, 600], [300, 400, 300], "neutro")

try:
    parametros = json.loads((raiz / "data" / "parametros_aprobados.json").read_text(encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError):
    parametros = {}
parametros.setdefault("r_min_mm", 45.0)  # si no hay aprobados, usa 45 mm para la prueba

INICIO, FIN = [0, 0, 500], [2300, 0, 500]
reporte = ducto_desde_entorno(INICIO, FIN, ancho_mm=40.0, alto_mm=25.0, espesor_mm=2.0,
                              parametros=parametros, nombre="ducto_entorno")
print(json.dumps(reporte, indent=2, ensure_ascii=False))

try:  # dibujo de la ruta cruda del A* (el PNG de Fer); matplotlib es opcional
    import visualizar_ruteo as v
    out = v.visualizar_ruta(INICIO, FIN, {"radio_ducto_mm": 23.6}, ruta_png=str(raiz / "data" / "salidas" / "ruta_ducto.png"))
    print("PNG:", out.get("imagen") or out.get("error_imagen"))
except ImportError:
    print("(sin visualizar_ruteo/matplotlib: se omite el PNG)")
