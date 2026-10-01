"""
Agente de ingesta 3D (Fase 1) - LangChain + Ollama, 100% local, la elección de este modelo es para no usar API Keys

Componentes:
  1. Modelo:         ChatOllama (chat + tool calling)
  2. Herramientas:   registrar_parametros y actualizar_parametro (escriben, con aprobacion humana) y
                     consultar_parametros (lee). Se pueden modificar segun los requerimientos
  3. Memoria:        InMemorySaver (checkpointer): recuerda la conversacion por thread_id
  4. HITL:           interrupt() se colocó dentro de la herramienta: el agente se pausa y el usuario decide si aprobar, editar o rechazar
  5. Persistencia:   los parametros aprobados se guardan en un archivo JSON (data/parametros_aprobados.json)

Flujo del human-in-the-loop:
  Input del usuario
  -> LLM propone registrar_parametros
  -> la herramienta llama interrupt() para saber si el usuario quiere aprobar, editar o rechazar
  -> el grafo se pausa y guarda su estado en el checkpointer
  -> chat() detecta la pausa, muestra la propuesta y pide la decision
  -> chat() reanuda con Command(resume=decision) y la herramienta continua
"""

import json
import re
import sys
from pathlib import Path
from typing import Literal

from langchain.agents import create_agent
from langchain_core.tools import tool  # Convierte una funcion Python en una herramienta que el LLM puede pedir ejecutar
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver  # Checkpointer en RAM: guarda el estado del grafo (mensajes, llamada pendiente, punto de pausa) por thread_id
from langgraph.types import Command, interrupt

# Permite ejecutar `python src/LLM/agente.py` desde cualquier carpeta: agrega src/ al path
# para poder importar el paquete hermano `herramientas` (funciones puras, sin LangChain).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from herramientas import documentos  # noqa: E402
from herramientas import geometria  # noqa: E402

DEBUG = False  # True para imprimir el resultado crudo de cada invoke()


# ----
# MODELO
# ----


modelo = ChatOllama( # Superior a OllamaLLM, pues permite manejar tool calling y mensajes
    model="llama3.1", # Si quieres otro modelo solo coloca ollama pull -modelo-
    temperature=0,  # Genera respuestas reales, cuando temperature es igual a 0, menos creativas son las respuestas
    num_ctx=8192,   # si es pequeno, Ollama recorta el texto sin avisar
)


# ----
# PARAMETROS APROBADOS
# ----


# parents[2] sube de LLM -> src -> raiz del proyecto
RUTA_PARAMETROS = Path(__file__).resolve().parents[2] / "data" / "parametros_aprobados.json"

# Funciones auxiliares
def _cargar_parametros() -> dict:
    """Lee los parametros aprobados del disco. Si el archivo no existe o esta danado, empieza vacio."""
    try:
        datos = json.loads(RUTA_PARAMETROS.read_text(encoding="utf-8"))
        return datos if isinstance(datos, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _guardar_parametros(parametros: dict) -> None:
    RUTA_PARAMETROS.parent.mkdir(parents=True, exist_ok=True)
    RUTA_PARAMETROS.write_text(
        json.dumps(parametros, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# PARAMETROS es una variable normal de Python (RAM del proceso).
# Para que sobreviva al cerrar el programa se escribe en un JSON y se vuelve a leer al iniciar.
PARAMETROS: dict = _cargar_parametros()  # solo contiene los parametros validados por el usuario


# ----
# HERRAMIENTAS
# ----


def _aprobar_y_guardar(propuesta: dict) -> str:
    """Pausa para que el usuario decida si aprobar, editar o rechazar y, solo y solo si aprueba o edita,
    guarda los valores. Devuelve el texto que el LLM leera como resultado de la herramienta.
    La comparten registrar_parametros y actualizar_parametro."""
    # Nota: al reanudar, LangGraph vuelve a ejecutar esta funcion desde el principio
    # e interrupt() devuelve esta vez la decision lo que el usuario metió.
    # Por eso antes de interrupt() nada se guarda todavia.
    decision = interrupt( # Esto crea la pausa
        {
            "tipo": "aprobar_parametros",
            "propuesta": propuesta,
            "actuales": dict(PARAMETROS),  # para que el humano vea que valores sobreescritos
            "mensaje": "El agente capturó estos parametros.",
        }
    )

    tipo = decision.get("decision") # Lee la clave "decision" del diccionario que armo pedir_decision

    # En caso de editar se sobreescriben los valores
    if tipo == "approve":
        finales = propuesta
    elif tipo == "edit":
        # El usuario solo manda los valores que cambio y el resto se conserva
        finales = {**propuesta, **decision.get("valores", {})}
    else:  # "reject" (o cualquier otra cosa por si las dudas, no se guarda nada)
        motivo = decision.get("motivo") or "sin motivo" # Esto es por si el usuario quiere dar algún motivo de su cancelacion
        return (
            f"El usuario RECHAZO los parametros porque: ({motivo}). "
            "No se registro nada. Preguntar al usuario por los valores correctos."
        )

    # finales es el diccionario que se guarda
    PARAMETROS.update(finales)  # El efecto ocurre despues de la aprobacion
    _guardar_parametros(PARAMETROS)  # y se escribe en disco

    if tipo == "edit":
        # Este texto es lo que el LLM lee. Sin esto, el LLM "recuerda" los valores originales
        # (los que aparecen en el mensaje del usuario y en su propia llamada) y no los corregidos.
        return (
            "El usuario corrigió algunos valores antes de aprobar. "
            f"Valores finales registrados: {json.dumps(PARAMETROS)}. "
            "Usa estos valores de ahora en adelante, no los de mensajes anteriores."
        )
    return f"Parametros registrados: {json.dumps(PARAMETROS)}"  # Convierte el diccionario PARAMETROS en un json


@tool
def registrar_parametros( # Ejemplo de parámetros, se pueden modificar
    r_min_mm: float,
    area_min_mm2: float,
    area_max_mm2: float,
    aspect_ratio_max: float,
    draft_angle_deg: float,
) -> str:
    """Registra parametros de diseno nuevos del ducto de A/C que el usuario acaba de dictar.
    Solo llamala cuando el usuario ya te haya dado los cinco valores.
    NO la uses para consultar o mostrar los valores actuales: para eso existe consultar_parametros.

    Args:
        r_min_mm: radio de curvatura minimo, en milimetros.
        area_min_mm2: area minima de la seccion transversal, en mm^2.
        area_max_mm2: area maxima de la seccion transversal, en mm^2.
        aspect_ratio_max: relacion de aspecto maxima de la seccion.
        draft_angle_deg: angulo de desmoldeo, en grados.
    """
    # Cuando creas una tool con langchain y/o langgraph, debes colocar un docstring para que el LLM sepa qué es lo que debe hacer.
    # Por eso tiene expecíficamente: "Solo llamala cuando el usuario ya te haya dado los cinco valores."

    propuesta = {   # Diccionario para mostrar al usuario
        "r_min_mm": r_min_mm,
        "area_min_mm2": area_min_mm2,
        "area_max_mm2": area_max_mm2,
        "aspect_ratio_max": aspect_ratio_max,
        "draft_angle_deg": draft_angle_deg,
    }

    return _aprobar_y_guardar(propuesta)


@tool
def consultar_parametros() -> str:
    """Devuelve los parametros de diseño ya aprobados y registrados.
    Usala cuando el usuario pregunte por los valores actuales o pida verlos.
    Solo lee: no registra ni modifica nada."""
    if not PARAMETROS:
        return "Todavia no hay parametros aprobados."
    return f"Parametros aprobados actualmente: {json.dumps(PARAMETROS)}"


@tool
def actualizar_parametro(
    campo: Literal["r_min_mm", "area_min_mm2", "area_max_mm2", "aspect_ratio_max", "draft_angle_deg"], # Literal indica que solo se pueden usar estas 5 variables
    valor: float,
) -> str:
    """Cambia un solo parametro ya aprobado (por ejemplo area_max_mm2 a 1000), sin repetir los demas.
    Usala cuando el usuario quiera corregir o cambiar un valor que ya esta guardado.
    Si todavia no hay parametros aprobados, usa registrar_parametros.

    Args:
        campo: nombre del parametro a cambiar.
        valor: nuevo valor numerico.
    """
    if not PARAMETROS: # Si todavia no hay parametros guardados
        return "Todavia no hay parametros aprobados. Usa registrar_parametros con los cinco valores."
    # La propuesta son los valores vigentes con solo ese campo cambiado que pasa por la misma aprobacion
    # ** Es un operador de desempaque para un diccionario
    return _aprobar_y_guardar({**PARAMETROS, campo: valor})


@tool
def procesar_documentos() -> str:
    """Convierte a Markdown los PDF nuevos de la carpeta de documentos, para poder buscar en ellos.
    Usala cuando el usuario diga que agrego un PDF o pida procesar los documentos. No recibe argumentos."""
    resultados = documentos.convertir_pendientes() # La descripción se encuentra en src/herramientas/documentos
    if not resultados: # Si no hay archivos pendientes por transformar
        return "No hay PDF en data/docs/pdf. Pidele al usuario que copie ahi sus archivos."
    lineas = [] 
    for r in resultados:
        # Si el resultado tiene la clave 'caracteres' arma un texto del tamaño definido de no encontrar nada regresa un ""
        detalle = f" ({r['caracteres']} caracteres, ~{r['tokens_aprox']} tokens)" if "caracteres" in r else ""
        # Si hay alguna anomalia como imagenes, o algo asi, el programa devuelve la advertencia. De no encontrar nada devuelve ""
        aviso = f" ADVERTENCIA: {r['advertencia']}" if "advertencia" in r else ""
        # Junta todo en formato de lista .md y la agrega a la lista lineas
        lineas.append(f"- {r['pdf']}: {r['estado']}{detalle}{aviso}")
    return "Documentos en data/docs/pdf:\n" + "\n".join(lineas) # Se guardan los valores en el vector lineas


@tool
def buscar_en_documentos(consulta: str) -> str:
    """Busca en los documentos ya procesados (normas, guias) y devuelve los 3 fragmentos mas relevantes,
    cada uno con su archivo y seccion. Usala para responder preguntas sobre normas o para obtener valores
    de diseno. Al responder, cita siempre el archivo y la seccion. Nunca des valores de una norma de memoria.

    Args:
        consulta: palabras clave de lo que se busca, por ejemplo 'radio minimo de curvatura'.
    """
    # Al LLM solo llegan 3 fragmentos (900 tokens), no el documento entero para que quepa en su ventana de contexto
    resultados = documentos.buscar(consulta, top_k=3) # La descripción se encuentra en src/herramientas/documentos
    if not resultados:
        return (
            "No encontre nada relevante. Puede que falte ejecutar procesar_documentos, "
            "o que las palabras de la consulta no aparezcan en los documentos."
        )
    partes = [f"[{i}] {r['archivo']} > {r['seccion']}\n{r['texto']}" for i, r in enumerate(resultados, 1)] # devuelve las 3 partes más relevantes para la resolución
    return "\n\n".join(partes)


@tool
def procesar_geometria() -> str:
    """Convierte a malla (.stl) los archivos STEP nuevos de la carpeta de geometria, y reporta
    un resumen (bounding box, unidades, piezas) de cada uno. No recibe argumentos.
    Usala cuando el usuario diga que agrego un STEP o pida procesar la geometria.

    Por ahora esta herramienta SOLO ingiere la malla del entorno: no detecta puertos de conexion
    (inicio/fin) automaticamente. Esos los debe dar el usuario aparte (posicion y seccion transversal)."""
    resultados = geometria.convertir_pendientes() # La descripción se encuentra en src/herramientas/geometria
    if not resultados:
        return "No hay archivos STEP (.step/.stp) en data/geometria/step. Pidele al usuario que copie ahi el archivo."
    lineas = []
    for r in resultados: 
        if "bounding_box_mm" in r: # Para conocer si el archivo estaba corrupto o falló la conversion entonces no habrá datos
            bb = r["bounding_box_mm"] #  Es el recueadro que delimita la pieza 3D
            detalle = (
                f" (bounding box mm: x{bb['x']} y{bb['y']} z{bb['z']}, " # Posición en X, Y, Z de la caja delimitadora
                # Indica el numero de cuarpos solidos independientes
                # Num de triangulos indica como esta formada la malla STL o sea que tan pesada es
                f"{r['num_piezas']} pieza(s), {r['num_triangulos']} triangulos)"
            )
        else:
            detalle = "" # Si la geometria esta corrupta entonces se envia un ""
        lineas.append(f"- {r['step']}: {r['estado']}{detalle}") 
    return "Geometria en data/geometria/step:\n" + "\n".join(lineas) # Manda la informacion de las geometrias procesadas


# Estas herramientas seran utilizadas posteriormente
HERRAMIENTAS = [
    registrar_parametros,
    consultar_parametros,
    actualizar_parametro,
    procesar_documentos,
    buscar_en_documentos,
    procesar_geometria,
]


# ----
# MEMORIA
# ----


# El checkpointer necesario para pausar y reanudar, esto permite guardar el estado
# Sin él no hay pausa ni memoria
CHECKPOINTER = InMemorySaver()

SYSTEM_PROMPT = (
    "Eres un asistente de ingenieria para ductos de aire acondicionado automotrices. "
    "Extrae parametros de diseno de lo que el usuario te indique y registralos con la "
    "herramienta registrar_parametros. Los cinco valores son: radio minimo (mm), area "
    "minima (mm2), area maxima (mm2), relacion de aspecto maxima y angulo de desmoldeo "
    "(grados). NUNCA inventes ni asumas valores: si falta alguno, preguntalo al usuario "
    "y no llames la herramienta hasta tener los cinco. "
    "Para mostrar o consultar los valores actuales usa SIEMPRE consultar_parametros; "
    "llama registrar_parametros unicamente cuando el usuario te dicte los cinco valores nuevos; "
    "si solo quiere cambiar UNO de los ya guardados, usa actualizar_parametro. "
    "El usuario puede corregir valores al aprobarlos: los valores vigentes son los que "
    "devuelven consultar_parametros o el ultimo resultado de una herramienta, "
    "no los de mensajes anteriores. "
    "Cada mensaje del usuario termina con una nota [Contexto interno ...] con los parametros "
    "aprobados: es la fuente de verdad, usala para responder consultas, pero NUNCA la repitas "
    "ni la menciones. "
    "Los valores que propongas deben salir de lo que el usuario escribio, nunca de tu memoria. "
    "Nunca digas que registraste, actualizaste o guardaste algo si no ejecutaste la herramienta. "
    "Para preguntas sobre normas, o para obtener valores de diseno de un documento, usa "
    "buscar_en_documentos y cita archivo y seccion; nunca des valores de una norma de memoria. "
    "Si el usuario dice que agrego un PDF, usa procesar_documentos. "
    "Si el usuario dice que agrego un archivo STEP (.step/.stp) o pide procesar la geometria, "
    "usa procesar_geometria; esa herramienta solo da un resumen (bounding box, unidades, "
    "piezas), nunca inventes esos numeros tampoco."
)

agente = create_agent( # conecta todo lo que hemos creado
    model=modelo,
    tools=HERRAMIENTAS,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=CHECKPOINTER,
)


# ----
# INTERFAZ DE TERMINAL PARA EL USUARIO (HITL)
# ----

# El guion bajo inicial es una convencion de Python indica que la funcion es de uso interno del modulo
# Las dos funciones _pausas y _mensajes sirven por si la función de invoke cambia según la versión
def _pausas(resultado) -> list:
    """Devuelve las pausas pendientes. Soporta el formato dict (clave '__interrupt__')
    y el formato nuevo con atributo .interrupts, segun la version de langgraph."""
    # En versiones antiguas las pausas se guardaban en la clave __interrupt__
    # Ahora es un atributo llamado .interrupts
    if isinstance(resultado, dict): # Para que sin importar la version pueda obtener la clave __interrupt__
        return list(resultado.get("__interrupt__") or []) 
    # Se asume un formato nuevo
    # getattr(resultado, "interrupts", None) lee el nuevo atributo .interrupts, de no existir devuelve None
    return list(getattr(resultado, "interrupts", None) or [])


def _mensajes(resultado) -> list:
    # Extrae el historial de la conversacion
    if isinstance(resultado, dict): # Antes los mensajes se guardaban en "messages", ahora dentro del atributo .value
        return resultado["messages"]
    return resultado.value["messages"]


def _limpiar_respuesta(texto: str) -> str:
    """Quita el eco de la nota de contexto interno, por si el LLM la copia en su respuesta.
    Separada de chat() para poder reutilizarla tambien desde la interfaz web (src/interfaz/app.py)."""
    # Busca un patron en el texto y lo sustituye
    # Busca un corchete lo que se encunetra en el estado actual
    return re.sub(r"\[(?:Contexto interno|ESTADO ACTUAL)[^\]]*\]", "", texto).strip()


def _numeros(texto: str) -> set[float]:
    """Todos los numeros que aparecen en un texto (acepta coma o punto decimal)."""
    return {float(n.replace(",", ".")) for n in re.findall(r"\d+(?:[.,]\d+)?", texto)} # Puro regex


def _numeros_de_busquedas(resultado) -> set[float]:
    """Numeros que aparecen en los fragmentos que devolvio buscar_en_documentos.
    Un valor tomado de un documento tiene fuente, aunque el usuario no lo haya escrito."""
    # Los numeros deben ser te tipo flotante. set() crea el conjunto vacio en la memoria
    numeros: set[float] = set()
    for m in _mensajes(resultado):
        if getattr(m, "name", None) == "buscar_en_documentos": # Busca los mensajes de la herramienta _mensajes
            for linea in str(m.content).splitlines(): # Toma el contenido del mensaje y lo divide linea por linea
                # se omiten las lineas de fuente "[1] archivo > 2. Seccion": el indice y el numero
                # de seccion no son valores de diseño y taparian valores inventados como 1 o 2
                if not re.match(r"^\[\d+\] ", linea):
                    numeros |= _numeros(linea) # si la linea no es un indice, se usa la herramienta _numeros
    return numeros


def _fuentes_de_busquedas(mensajes: list) -> list[str]:
    """Contenido de las llamadas a buscar_en_documentos dentro de `mensajes`. Se muestra tal cual al humano
    para que compare la respuesta del LLM contra el texto real, sin pausar el flujo (buscar_en_documentos
    es de solo lectura, a diferencia de registrar_parametros/actualizar_parametro)."""
    # Extrae todo el texto disponible en el contexto
    return [
        str(m.content) # Cada mensaje que pasa para el filtro extrae su contenido
        for m in mensajes 
        if getattr(m, "name", None) == "buscar_en_documentos"  # Filtra los mensajes que se encunetran en la herramienta buscar_en_documentos
        and not str(m.content).startswith("No encontre nada relevante") # En caso de que no se haya encontrado nada útil
    ]


def _contexto() -> str:
    """Estado real (el JSON) que se le recuerda al LLM en cada turno.
    El LLM no es fuente de verdad, sin esto, en una sesion nueva no sabe que ya hay valores guardados.
    Va al final del mensaje, caso contrario (al principio), el LLM la copiaba como parte de su respuesta."""
    estado = json.dumps(PARAMETROS) if PARAMETROS else "ninguno todavia" # Es la memoria a largo plazo del agente
    return f"\n\n[Contexto interno, no lo repitas ni lo menciones. Parametros aprobados: {estado}]"


def _confirmar_sospechosos(pendientes: set) -> bool:
    """Si quedan valores que el usuario no dijo, exige una confirmacion explicita."""
    if not pendientes: # Si todos lo numero prouestos por la IA son confiables
        return True
    # Si salen numeros sospechosos los imprime la terminal
    print(f"  Cuidado: estos valores no los escribiste tu ni salen de los documentos consultados y podrian estar inventados: {', '.join(sorted(pendientes))}")
    return input("  Aprobar de todos modos? [s/n]: ").strip().lower() in {"s", "si", "sí"}


def pedir_decision(
    propuesta: dict, actuales: dict | None = None, dichos: set[float] | None = None
) -> dict:
    """Muestra la propuesta y devuelve la decision del usuario como dict.

    dichos: numeros que el usuario escribio en la conversacion. Un valor propuesto que no
    este ahi pudo ser inventado por el LLM: se marca y se pide confirmacion antes de aprobarlo.
    """
    # Propuesta son los valores que el agente quiere guardad
    # Actuales son los vvalores aprobados en el sistema. Son opcionales porque la primera vez no existen
    # Dichos es el conjunto de valores que el usuario escribió explicitamente
    actuales = actuales or {} # Si el diccionario esta vacio, evita errores
    if dichos is None: # Si el usuario no dijo nada, entonces el agente no puede saber que es sospechoso y que no, entonces se crea un conjunto vacio
        sospechosos: set = set()  # sin informacion no se puede juzgar
    else:
        sospechosos = {
            campo for campo, valor in propuesta.items() # Recorre cada campo y valor en la propuesta para ver si no hay alucinaciones
            if actuales.get(campo) != valor  # un valor ya aprobado antes no es sospechoso. Pero si fue aprobado y aparece un valor nuevo, entonces es sospechoso.
            and not any(abs(valor - n) < 1e-9 for n in dichos) # Revisa si el valor coincide con alguno que el usuario escribió
        }

    print("\n--- El agente propone registrar estos valores ---")
    for campo, valor in propuesta.items(): # Ciclo para imprimir todos los valores registrados
        anterior = actuales.get(campo)
        # Si ya habia un valor aprobado distinto, se muestra para que el usuario vea que va a sobreescribir
        aviso = f"   <- CAMBIA (antes: {anterior})" if anterior is not None and anterior != valor else ""
        if campo in sospechosos:
            aviso += "   <- OJO: no aparece en lo que escribiste ni en los documentos consultados" # Se muestran los cambios en valores anteriores y sospechosos
        print(f"  {campo}: {valor}{aviso}")

    # Comienzo del ciclo
    while True:
        opcion = input("[a]probar / [e]ditar / [r]echazar: ").strip().lower() # Es para normalizar los valores en la pausa

        if opcion in {"a", "aprobar"}:
            if _confirmar_sospechosos(sospechosos): # llama a _confirmar_sospechosos. Si no los hay, manda Approve
                return {"decision": "approve"}
            print("  Elige editar o rechazar.")
            continue # Reinicia el bucle 

        if opcion in {"e", "editar"}:
            valores: dict = {} # Crea un diccionario vacio donde se cuardan las ediciones del usuario 
            print("Escribe campo=valor (e.g. r_min_mm=60). Enter vacio para terminar.") # Estructura
            while True:
                linea = input("  editar> ").strip()  # Linea input para cambiar el valor
                if not linea:
                    break # El enter vacio
                campo, _, valor = linea.partition("=") # separa los valores en campo y valor con el =
                campo = campo.strip() # Elimina espacios 
                if campo not in propuesta:
                    print(f"  Campo desconocido. Validos: {', '.join(propuesta)}") # Solo reconoce lo que se encunetra en el diccionario original
                    continue
                try:
                    valores[campo] = float(valor.replace(",", "."))
                except ValueError:
                    print("  El valor debe ser numerico.")
            if valores: # En caso de que no haya cambiado nada
                # Los valores sospechosos que no se corrigieron se aprobarian tal cual pero igual hay que confirmarlo
                if _confirmar_sospechosos(sospechosos - set(valores)): # Resta de los sospechosos a los que el usuario corrigió manualmente
                    return {"decision": "edit", "valores": valores}
                print("  Vuelve a elegir.")
                continue
            print("No editaste ningun valor.")
            continue

        if opcion in {"r", "rechazar"}: # Rechaza la decision
            motivo = input("Motivo (opcional): ").strip() # Pide un motivo
            return {"decision": "reject", "motivo": motivo}

        print("Opcion no valida.")


# ----
# BUCLE DE CONVERSACION
# ----


def chat() -> None:
    # Mismo thread_id = misma memoria. Tambien es lo que permite reanudar la pausa.
    config = {"configurable": {"thread_id": "sesion-1"}}
    dichos: set[float] = set()  # numeros que el usuario escribio, o que salieron de los documentos consultados
    documentos.asegurar_carpetas()  # crea data/docs/pdf y data/docs/md si no existen
    geometria.asegurar_carpetas()  # crea data/geometria/step y data/geometria/malla si no existen
    mensajes_vistos = 0  # cuantos mensajes tenia el historial antes de este turno (para no repetir fuentes viejas)

    while True:
        texto = input("Tu: ").strip() # Esto manda el mensaje al agente
        if texto.lower() in {"salir", "exit"}: # Para terminar la conversacion
            break
        if not texto:
            continue

        dichos |= _numeros(texto)  # se calcula con el texto puro, sin la nota que le agrega _contexto()

        # Es el resultado del texto con la configuracion dada y el texto de input
        resultado = agente.invoke(
            {"messages": [{"role": "user", "content": texto + _contexto()}]}, config  # type: ignore
        )

        # Mientras el agente siga pausado se pide al usuario que tome una decision y se reanuda con el mismo config
        # Se asume una sola llamada a herramienta a la vez

        # Asigna y evalua a la vez
        while pausas := _pausas(resultado):
            dichos |= _numeros_de_busquedas(resultado)  # valores con fuente en un documento no son sospechosos
            contenido = pausas[0].value  # el diccionario que la herramienta le paso a interrupt()
            decision = pedir_decision(contenido["propuesta"], contenido.get("actuales"), dichos)
            # Segun la respuesta en el interrupt() y se devuelve con la siguiente linea
            resultado = agente.invoke(Command(resume=decision), config)  #type: ignore

        if DEBUG:
            print(resultado)

        # Respuesta final del agente
        respuesta = _limpiar_respuesta(_mensajes(resultado)[-1].content)
        print("Agente:", respuesta or "(sin texto)")

        # Solo los mensajes agregados desde la ultima vez 
        # El historial completo se va acumulando en la memoria del agente, y repetir busquedas viejas cada turno seria ruido
        mensajes_actuales = _mensajes(resultado)
        fuentes = _fuentes_de_busquedas(mensajes_actuales[mensajes_vistos:])
        mensajes_vistos = len(mensajes_actuales)
        if fuentes:
            print("\n[Fuentes consultadas - compara el texto contra la respuesta de arriba:]")
            for fuente in fuentes:
                print(fuente)
            print()

        print(f"[PARAMETROS aprobados: {PARAMETROS}]")


if __name__ == "__main__":
    chat()
