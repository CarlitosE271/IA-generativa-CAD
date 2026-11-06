"""
Interfaz web (Streamlit) del agente de Ingesta 3D - Fase 1.

Toda la logica (el grafo de LangGraph,las herramientas, el HITL, la deteccion de valores sospechosos) sigue viviendo en agente.py.
Este archivo solo traduce esa logica a widgets de Streamlit.

Diferencia clave con la consola (chat() en agente.py):
  Streamlit no tiene un input() bloqueante: cada clic/envio vuelve a ejecutar este archivo
  Por eso el estado que antes eran variables locales de chat() ahora viven en st.session_state, que Streamlit conserva
  entre reruns de la misma pestaña/sesion de navegador.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Igual que en agente.py: agrega src/ al path para poder importar LLM y herramientas
# sin importar desde donde se corra `streamlit run`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from langgraph.types import Command

from LLM import agente  # noqa: E402
from herramientas import documentos  # noqa: E402
from herramientas import geometria  # noqa: E402

st.set_page_config(page_title="Ingesta 3D - Ductos A/C", page_icon="🌀", layout="centered")

# Se crean una vez por rerun; mkdir(exist_ok=True) hace que sea barato repetirlo.
documentos.asegurar_carpetas()
geometria.asegurar_carpetas()


# ----
# ESTADO DE LA SESION 
# ----
def _iniciar_estado() -> None:
    import uuid

    if "thread_id" not in st.session_state:
        # Un thread_id por pestana de navegador -> cada quien tiene su propia memoria de
        # conversacion en el CHECKPOINTER compartido (PARAMETROS, en cambio, NO se separa asi).
        st.session_state.thread_id = str(uuid.uuid4())
    if "historial" not in st.session_state:
        # Lo que se muestra en pantalla: [{"role": "user"/"assistant", "content": str, "fuentes": [...]}]
        st.session_state.historial = []
    if "dichos" not in st.session_state:
        # Numeros que el usuario escribio o que salieron de un documento consultado (ver pedir_decision)
        st.session_state.dichos = set()
    if "mensajes_vistos" not in st.session_state:
        st.session_state.mensajes_vistos = 0
    if "pendiente" not in st.session_state:
        # None = sin pausa; dict = lo que la herramienta le paso a interrupt() (propuesta/actuales/mensaje)
        st.session_state.pendiente = None


_iniciar_estado()
config = {"configurable": {"thread_id": st.session_state.thread_id}}


# ----
# Traduccion de un resultado de invoke()/resume a estado de Streamlit
# ----
def _avanzar(resultado) -> None:
    """Equivalente al cuerpo del while/despues-del-while de chat(): si el agente se pauso,
    guarda la propuesta para mostrar el formulario; si no, guarda la respuesta final."""
    pausas = agente._pausas(resultado)
    if pausas:
        # Igual que en consola: se asume una sola llamada a herramienta pendiente a la vez.
        st.session_state.dichos |= agente._numeros_de_busquedas(resultado)
        st.session_state.pendiente = pausas[0].value
        return

    mensajes = agente._mensajes(resultado)
    respuesta = agente._limpiar_respuesta(mensajes[-1].content)
    nuevos = mensajes[st.session_state.mensajes_vistos:]
    fuentes = agente._fuentes_de_busquedas(nuevos)
    archivos = agente.informe_ducto.archivos_de_mensajes(nuevos)  # PNG de la ruta y STEP/STL del ducto de ESTE turno
    traza = agente.informe_ducto.traza_herramientas(nuevos)  # que herramientas llamo el agente y que devolvieron
    if agente.informe_ducto.respuesta_inutil(respuesta):
        # El modelo local a veces responde "Lo siento, no puedo continuar" aunque la herramienta SI funciono:
        # se muestra el resultado de la herramienta (es texto ya redactado y verificable), no la disculpa.
        resultado_herramienta = agente.informe_ducto.ultimo_resultado(nuevos)
        if resultado_herramienta:
            respuesta = "(El modelo no logro redactar la respuesta; este es el resultado de la herramienta)\n\n" + resultado_herramienta
    st.session_state.mensajes_vistos = len(mensajes)
    st.session_state.historial.append(
        {"role": "assistant", "content": respuesta or "(sin texto)", "fuentes": fuentes, "archivos": archivos, "traza": traza}
    )
    st.session_state.pendiente = None


# ----
# Formulario de aprobar/editar/rechazar (version web de pedir_decision)
# ----
def _sospechosos(propuesta: dict, actuales: dict) -> set[str]:
    """Misma regla que pedir_decision: un valor que no dijiste tu ni salio de un documento,
    y que ademas cambia algo ya aprobado, se marca como sospechoso de estar inventado."""
    dichos = st.session_state.dichos
    return {
        campo
        for campo, valor in propuesta.items()
        if actuales.get(campo) != valor
        and not any(abs(valor - n) < 1e-9 for n in dichos)
    }


def _operador() -> str:
    """Quien firma las aprobaciones (campo de la barra lateral). Vacio -> la bitacora usa el usuario del sistema."""
    return (st.session_state.get("operador") or "").strip()


def _mostrar_formulario_decision(contenido: dict) -> None:
    propuesta: dict = contenido["propuesta"]
    actuales: dict = contenido.get("actuales") or {}
    sospechosos = _sospechosos(propuesta, actuales)

    st.info(contenido.get("mensaje", "El agente propone registrar estos valores."))
    for campo, valor in propuesta.items():
        anterior = actuales.get(campo)
        texto = f"**{campo}**: {valor}"
        if anterior is not None and anterior != valor:
            texto += f"  — antes: {anterior}"
        st.markdown(texto)
        if campo in sospechosos:
            st.caption(f"⚠️ {campo} no aparece en lo que escribiste ni en los documentos consultados.")

    opcion = st.radio(
        "Que quieres hacer con estos valores?",
        ["Aprobar", "Editar", "Rechazar"],
        key="decision_opcion",
        horizontal=True,
    )

    if opcion == "Aprobar":
        habilitar = True
        if sospechosos:
            habilitar = st.checkbox(
                "Confirmo que quiero aprobar aunque estos valores no los escribi yo "
                f"ni salen de un documento consultado: {', '.join(sorted(sospechosos))}.",
                key="confirmar_sospechosos_aprobar",
            )
        if st.button("Enviar decision", key="enviar_aprobar", disabled=not habilitar):
            with st.spinner("Aplicando..."):
                resultado = agente.agente.invoke(
                    Command(resume={"decision": "approve", "operador": _operador()}), config
                )
            _avanzar(resultado)
            st.rerun()

    elif opcion == "Editar":
        st.caption("Cambia los valores que quieras corregir antes de aprobar.")
        valores_nuevos: dict = {}
        for campo, valor in propuesta.items():
            nuevo = st.number_input(campo, value=float(valor), key=f"editar_{campo}", format="%.4f")
            if abs(nuevo - valor) > 1e-9:
                valores_nuevos[campo] = nuevo
        pendientes_sin_editar = sospechosos - set(valores_nuevos)
        habilitar = True
        if pendientes_sin_editar:
            habilitar = st.checkbox(
                "Confirmo aprobar sin corregir estos valores sospechosos: "
                f"{', '.join(sorted(pendientes_sin_editar))}.",
                key="confirmar_sospechosos_editar",
            )
        if st.button("Enviar decision", key="enviar_editar", disabled=not habilitar):
            with st.spinner("Aplicando..."):
                resultado = agente.agente.invoke(
                    Command(resume={"decision": "edit", "valores": valores_nuevos, "operador": _operador()}), config
                )
            _avanzar(resultado)
            st.rerun()

    else:  # Rechazar
        motivo = st.text_input("Motivo (opcional)", key="motivo_rechazo")
        if st.button("Enviar decision", key="enviar_rechazar"):
            with st.spinner("Aplicando..."):
                resultado = agente.agente.invoke(
                    Command(resume={"decision": "reject", "motivo": motivo, "operador": _operador()}), config
                )
            _avanzar(resultado)
            st.rerun()


# ----
# Barra lateral: parametros vigentes, carga y procesamiento de archivos
# ----
def _guardar_subida(archivo, carpeta: Path) -> str:
    """Guarda un archivo subido por Streamlit dentro de `carpeta`.
    Solo se usa el NOMBRE del archivo (sin ninguna ruta que el navegador pudiera mandar),
    igual de estricto que _nombre_seguro en documentos.py/geometria.py."""
    nombre = Path(archivo.name).name
    (carpeta / nombre).write_bytes(archivo.getvalue())
    return nombre


with st.sidebar:
    st.text_input(
        "Operador (firma tus aprobaciones)", key="operador",
        help="Se guarda en la bitacora data/auditoria/aprobaciones.jsonl con cada aprobacion, edicion o rechazo.",
    )
    st.divider()
    st.subheader("Parametros aprobados")
    st.json(agente.PARAMETROS or {"info": "todavia no hay parametros aprobados"})

    st.divider()
    st.subheader("Documentos (PDF)")
    subida_pdf = st.file_uploader(
        "Sube una norma en PDF", type=["pdf"], accept_multiple_files=True, key="subida_pdf"
    )
    if subida_pdf:
        for archivo in subida_pdf:
            _guardar_subida(archivo, documentos.ruta_pdf)
        st.caption(f"{len(subida_pdf)} archivo(s) en data/docs/pdf.")
    if st.button("Procesar documentos pendientes"):
        with st.spinner("Convirtiendo PDF a Markdown..."):
            resultados = documentos.convertir_pendientes()
        if not resultados:
            st.warning("No hay PDF en data/docs/pdf.")
        for r in resultados:
            detalle = f" ({r['caracteres']} caracteres, ~{r['tokens_aprox']} tokens)" if "caracteres" in r else ""
            aviso = f" ⚠️ {r['advertencia']}" if "advertencia" in r else ""
            st.write(f"- {r['pdf']}: {r['estado']}{detalle}{aviso}")

    st.divider()
    st.subheader("Geometria (STEP)")
    subida_step = st.file_uploader(
        "Sube un archivo STEP", type=["step", "stp"], accept_multiple_files=True, key="subida_step"
    )
    if subida_step:
        for archivo in subida_step:
            _guardar_subida(archivo, geometria.ruta_step)
        st.caption(f"{len(subida_step)} archivo(s) en data/geometria/step.")
    if st.button("Procesar geometria pendiente"):
        with st.spinner("Convirtiendo STEP a malla..."):
            resultados = geometria.convertir_pendientes()
        if not resultados:
            st.warning("No hay archivos STEP en data/geometria/step.")
        for r in resultados:
            if "bounding_box_mm" in r:
                bb = r["bounding_box_mm"]
                detalle = (
                    f" (bbox mm x{bb['x']} y{bb['y']} z{bb['z']}, "
                    f"{r['num_piezas']} pieza(s), {r['num_triangulos']} triangulos)"
                )
            else:
                detalle = ""
            st.write(f"- {r['step']}: {r['estado']}{detalle}")


# ----
# Chat principal
# ----
st.title("Agente de Ingesta 3D — Ductos A/C")

def _mostrar_archivos(archivos: list, indice: int) -> None:
    """Imagen de la ruta (st.image) y botones de descarga para los STEP/STL del ducto."""
    for j, a in enumerate(archivos):
        ruta = Path(a["ruta"])
        if not ruta.is_file():
            st.caption(f"(no se encontro el archivo {ruta.name})")
            continue
        if a["tipo"] == "imagen":
            st.image(str(ruta), caption="Ruta del A* y bloques del entorno")
        else:
            st.download_button(
                f"Descargar {ruta.name}", data=ruta.read_bytes(), file_name=ruta.name,
                key=f"descarga_{indice}_{j}",
            )


for i, msg in enumerate(st.session_state.historial):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("archivos"):
            _mostrar_archivos(msg["archivos"], i)
        if msg.get("traza"):
            with st.expander("Que hizo el agente (herramientas llamadas)"):
                for paso in msg["traza"]:
                    if paso["tipo"] == "llamada":
                        st.markdown(f"**Llamo** `{paso['herramienta']}` con `{paso['argumentos']}`")
                    else:
                        st.markdown(f"**Resultado de** `{paso['herramienta']}`:")
                        st.text(paso["texto"])
        if msg.get("fuentes"):
            with st.expander("Fuentes consultadas (compara este texto contra la respuesta de arriba)"):
                for fuente in msg["fuentes"]:
                    st.text(fuente)

if st.session_state.pendiente is not None:
    _mostrar_formulario_decision(st.session_state.pendiente)
else:
    texto = st.chat_input("Escribe tu mensaje...")
    if texto:
        st.session_state.historial.append({"role": "user", "content": texto})
        st.session_state.dichos |= agente._numeros(texto)  # con el texto puro, sin la nota de _contexto()
        with st.spinner("Pensando..."):
            resultado = agente.agente.invoke(
                {"messages": [{"role": "user", "content": texto + agente._contexto()}]},
                config,  # type: ignore
            )
        _avanzar(resultado)
        st.rerun()
