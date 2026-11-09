"""
Encontrar el prompt más largo que llama3.1 aguanta, agregando reglas de
una en una, y probar una alternativa de ingenieria, que las herramientas de lectura devuelvan
su texto directo al usuario, sin que el modelo lo vuelva a redactar (return_direct).

Uso, desde la raiz del proyecto (se necesita que Ollama este  encendido):   
    python scripts/diagnostico_llm.py

Lo que se sabe de rondas de prueba anteriores:
  - prompt actual (largo) + nota: falla. prompt_corto + nota: 3/3 ok.
  - prompt nuevo (mediano) con nota larga O corta: sigue fallando en "que bloques hay".

  Variantes:
  base         prompt_corto                                   (control: debe estar ok)
  +params      base + reglas de registrar/actualizar/consultar parametros
  +honestidad  +params + "valores vigentes" y "nunca digas que hiciste algo sin ejecutarlo"
  +entorno     +honestidad + reglas de entorno/ruta/ducto/infactible (= prompt "nuevo")
  +entorno_sn  igual que +entorno pero SIN nota                -> ¿la nota importa con el prompt largo?
  directo      +entorno, y consultar_entorno/calcular_ruta con return_direct (el modelo no re-redacta)

  Veredictos: ok (trae datos) | VACIA (suena bien pero sin datos) | INUTIL (disculpa/vacio)
"""
import json
import re
import sys
import time
from pathlib import Path

raiz = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(raiz / "src"))

from langchain.agents import create_agent  # noqa: E402

from LLM import agente as ag  # noqa: E402
from cad import informe_ducto as ic  # noqa: E402

BASE = (
    "Eres un asistente de ingenieria para ductos de aire acondicionado automotrices. Responde en espanol, "
    "breve. Usa las herramientas para obtener los datos y responde con lo que devuelven; nunca inventes "
    "numeros ni valores de normas. Si un resultado trae marcas entre corchetes, no las repitas. "
)
REGLA_PARAMS = (
    "Parametros de diseno (radio minimo mm, area minima mm2, area maxima mm2, relacion de aspecto maxima, "
    "angulo de desmoldeo grados): registralos con registrar_parametros solo cuando el usuario te dicte los cinco "
    "(si falta alguno, preguntalo); para cambiar uno usa actualizar_parametro; para verlos usa consultar_parametros. "
)
REGLA_HONESTIDAD = (
    "Los valores vigentes son los de la nota [Contexto interno] y los de la ultima herramienta: el usuario puede "
    "corregirlos al aprobar. Nunca digas que registraste, actualizaste, guardaste o generaste algo si no ejecutaste "
    "la herramienta. Con buscar_en_documentos cita siempre archivo y seccion. "
)
REGLA_ENTORNO = (
    "El entorno de bloques se consulta con consultar_entorno; la ruta cruda con calcular_ruta; el ducto con "
    "generar_ducto (necesita parametros aprobados y que el usuario te de inicio, fin, ancho, alto y espesor: "
    "si falta alguno, preguntalo). Si generar_ducto devuelve INFACTIBLE, anuncialo, resume las propuestas y "
    "pregunta cual elige el usuario; nunca cambies r_min por tu cuenta."
)
P_PARAMS = BASE + REGLA_PARAMS
P_HONESTIDAD = P_PARAMS + REGLA_HONESTIDAD
P_ENTORNO = P_HONESTIDAD + REGLA_ENTORNO


def nota() -> str:
    estado = json.dumps(ag.PARAMETROS) if ag.PARAMETROS else "ninguno todavia"
    return f"\n\n[Contexto interno: parametros aprobados: {estado}]"


def sin_nota() -> str:
    return ""


def herramientas_directas():
    """Mismas herramientas, pero las de LECTURA con return_direct=True: su texto ES la respuesta."""
    directas = {"consultar_entorno", "calcular_ruta"}
    return [t.model_copy(update={"return_direct": True}) if t.name in directas else t for t in ag.HERRAMIENTAS]


PREGUNTAS = [
    ("Qué bloques hay en el entorno?", lambda t: "motor" in t.lower()),
    ("Qué hay en el entorno y de qué tipo es cada bloque?", lambda t: "motor" in t.lower()),
    ("Calcula la ruta de [0,0,500] a [2300,0,500]", lambda t: bool(re.search(r"\d{2,}", t))),
    ("Cuáles son los parámetros aprobados?", lambda t: bool(re.search(r"\d", t)) or "no hay" in t.lower() or "todavia" in t.lower()),
]

# nombre: (system_prompt, nota, herramientas)
VARIANTES = {
    "base": (BASE, nota, ag.HERRAMIENTAS),
    "+params": (P_PARAMS, nota, ag.HERRAMIENTAS),
    "+honestidad": (P_HONESTIDAD, nota, ag.HERRAMIENTAS),
    "+entorno": (P_ENTORNO, nota, ag.HERRAMIENTAS),
    "+entorno_sin_nota": (P_ENTORNO, sin_nota, ag.HERRAMIENTAS),
    "directo": (P_ENTORNO, nota, None),  # herramientas_directas() se arma abajo
}

tabla = []
for nombre, (prompt, nota_fn, herramientas) in VARIANTES.items():
    herramientas = herramientas or herramientas_directas()
    agente = create_agent(model=ag.modelo, tools=herramientas, system_prompt=prompt)
    for pregunta, trae_datos in PREGUNTAS:
        t0 = time.time()
        try:
            res = agente.invoke({"messages": [{"role": "user", "content": pregunta + nota_fn()}]})
            mensajes = ag._mensajes(res)
        except Exception as e:  # noqa: BLE001
            tabla.append((nombre, pregunta, "ERROR", f"{type(e).__name__}: {e}", time.time() - t0))
            continue
        llamadas = [p["herramienta"] for p in ic.traza_herramientas(mensajes) if p["tipo"] == "llamada"]
        final = ag._limpiar_respuesta(str(mensajes[-1].content))
        if ic.respuesta_inutil(final):
            veredicto = "INUTIL"
        else:
            veredicto = "ok" if trae_datos(final) else "VACIA"
        tabla.append((nombre, pregunta, veredicto, f"llamo {llamadas or 'NADA'} -> {final[:90]!r}", time.time() - t0))

print("\n" + "=" * 100)
for nombre, pregunta, veredicto, detalle, seg in tabla:
    print(f"[{veredicto:6}] {nombre:17} | {pregunta[:28]:28} | {seg:3.0f}s | {detalle}")
print("\nResumen (ok / total):")
for nombre in VARIANTES:
    filas = [f for f in tabla if f[0] == nombre]
    print(f"  {nombre:17} {sum(f[2] == 'ok' for f in filas)}/{len(filas)}")
