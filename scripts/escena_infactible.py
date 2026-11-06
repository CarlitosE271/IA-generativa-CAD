"""
Carga una escena de prueba en la que un R_min grande es INFACTIBLE, para probar en el chat el flujo
"el agente anuncia el problema, propone y tu decides". Guarda un respaldo de tu entorno para no perderlo.

Uso, desde la raiz del proyecto:
  python scripts/escena_infactible.py              # respalda tu entorno y carga la escena de dos muros
  python scripts/escena_infactible.py --restaurar  # devuelve tu entorno original

La escena: dos muros altos desfasados (P1 y P2) entre el inicio [0,0,0] y el fin [1500,0,0], que obligan a
una "S". Con la seccion 40x25 mm: R_min <= ~178 mm si cabe; R_min = 600 mm NO cabe.
"""
import shutil
import sys
from pathlib import Path

raiz = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(raiz / "src" / "herramientas"))

import ruteo_astar as r  # noqa: E402

entorno = raiz / "data" / "entorno" / "entorno.json"
respaldo = entorno.with_name("entorno_respaldo.json")
r.configurar_ruta_entorno(entorno)

if "--restaurar" in sys.argv:
    if not respaldo.exists():
        sys.exit("No hay respaldo (data/entorno/entorno_respaldo.json): no se cambio nada.")
    shutil.copy2(respaldo, entorno)
    respaldo.unlink()
    print("Entorno original restaurado:", [b["nombre"] for b in r.listar_bloques()])
    sys.exit(0)

if respaldo.exists():
    print("Ya hay un respaldo guardado: NO se sobreescribe (asi no pierdes tu entorno original).")
elif entorno.exists():
    shutil.copy2(entorno, respaldo)
    print("Respaldo de tu entorno en", respaldo)

r.limpiar_entorno()
r.agregar_bloque("Muro P1", [500, 150, 0], [100, 600, 3000], "neutro")
r.agregar_bloque("Muro P2", [1000, -150, 0], [100, 600, 3000], "neutro")
print("Escena cargada:", [b["nombre"] for b in r.listar_bloques()])
print("""
Prueba en el chat (Streamlit), en este orden:
  1) "Cambia r_min_mm a 600"                       -> aprueba el formulario
  2) "Genera el ducto de [0,0,0] a [1500,0,0], seccion 40 por 25, espesor 2"   -> aprueba
     Esperado: el agente anuncia INFACTIBLE, da el radio maximo (~178 mm) y te pregunta que propuesta eliges.
  3) "Baja el R_min a lo que propones"             -> aprueba el formulario de r_min
  4) Repite el ducto del paso 2                    -> ahora si debe generarse
  Al terminar:  python scripts/escena_infactible.py --restaurar   y   "Cambia r_min_mm a 45"
""")
