"""
Bitacora de auditoria de aprobaciones (version corta).

Cada vez que un HUMANO aprueba, edita o rechaza algo que propuso el agente (parametros de diseno o la
generacion de un ducto) se agrega UNA linea JSON a data/auditoria/aprobaciones.jsonl (formato JSONL:
un objeto JSON por linea, solo se agrega al final, nunca se reescribe).

Cada linea responde: quien, cuando, que, que propuso la IA, que quedo finalmente, que corrigio el humano.

Hay dos tipos de linea (campo "evento"):
  - "aprobacion": la decision del humano. Lleva un "id" unico.
  - "resultado":  lo que paso DESPUES de aprobar un ducto (estado, nombre, archivos), con el mismo id en
                  "aprobacion_id", mas el contexto para reproducirlo (parametros vigentes, modelo LLM, campo de costo).
Asi se puede reconstruir "que ducto salio de esta aprobacion y con que resultado". Una aprobacion de ducto sin linea
de resultado significa que el proceso se interrumpio (o fallo la propia bitacora) antes de terminar.
Las lineas viejas (sin "evento" ni "id") siguen siendo validas: leer() no las rechaza.

Limites (para decirlos con honestidad en la presentacion):
  - Es "append-only" por convencion, NO a prueba de manipulacion: quien tenga acceso al archivo puede editarlo.
    (Si hiciera falta, el siguiente paso seria encadenar hashes: cada linea guarda el hash de la anterior.)
  - "operador" es lo que se escribe en la barra lateral de la interfaz (o la variable de entorno GM_OPERADOR,
    o el usuario del sistema operativo). No hay autenticacion real.
  - Este modulo no depende de LangChain ni de Streamlit: se puede probar solo.

Variable de entorno opcional GM_AUDITORIA: ruta alternativa del archivo (la usan las pruebas).
"""
import getpass
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

# parents[2] sube de herramientas -> src -> raiz del proyecto
RUTA_AUDITORIA = Path(__file__).resolve().parents[2] / "data" / "auditoria" / "aprobaciones.jsonl"

DECISIONES = ("approve", "edit", "reject")


def ruta_actual() -> Path:
    """Ruta de la bitacora (se lee la variable de entorno en cada llamada, no al importar)."""
    alterna = os.environ.get("GM_AUDITORIA")
    return Path(alterna) if alterna else RUTA_AUDITORIA


def operador_por_defecto() -> str:
    """Quien firma si la interfaz no lo dijo: GM_OPERADOR, o el usuario del sistema operativo."""
    nombre = os.environ.get("GM_OPERADOR")
    if nombre:
        return nombre
    try:
        return getpass.getuser()
    except Exception:  # sin usuario resoluble (contenedores, etc.): no es motivo para romper la aprobacion
        return "desconocido"


def nuevo_id() -> str:
    """Identificador corto y unico de una aprobacion (12 hex: sobra para un archivo de este tamano)."""
    return uuid.uuid4().hex[:12]


def operador_de(decision: dict) -> str:
    """Quien firmo: lo que mando la interfaz; si vino vacio, el valor por defecto."""
    return (decision.get("operador") or "").strip() or operador_por_defecto()


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def construir_evento(
    tipo: str,
    decision: dict,
    propuesta: dict,
    finales: dict | None,
    anteriores: dict | None = None,
    id_evento: str | None = None,
) -> dict:
    """Arma el registro de una decision humana (todavia no lo escribe).

    tipo:       "parametros" o "ducto" (que se estaba aprobando).
    decision:   el dict que devolvio el humano: {"decision": ..., "valores": {...}, "motivo": ..., "operador": ...}.
    propuesta:  lo que propuso el agente.
    finales:    lo que quedo realmente (None si se rechazo).
    anteriores: solo para "parametros": los valores aprobados ANTES de esta decision.
    id_evento:  identificador de la aprobacion (si no se da, se crea uno). Para ductos lo genera el agente antes
                de anotar, porque la linea de resultado necesita el mismo id.
    """
    resultado = decision.get("decision")
    if resultado not in DECISIONES:  # cualquier cosa rara se trata como rechazo, igual que en el agente
        resultado = "reject"

    evento = {
        "ts": _ahora(),
        "evento": "aprobacion",
        "id": id_evento or nuevo_id(),
        "operador": operador_de(decision),
        "tipo": tipo,
        "decision": resultado,
        "motivo": decision.get("motivo") or None,
        "propuesta": propuesta,
        "finales": finales if resultado != "reject" else None,
    }

    if resultado == "edit" and finales is not None:
        # Lo mas valioso para gobernanza: donde el humano le corrigio la plana a la IA.
        evento["correcciones_humanas"] = {
            k: {"propuesto": propuesta.get(k), "final": v}
            for k, v in finales.items()
            if propuesta.get(k) != v
        }
    if resultado != "reject" and finales is not None and anteriores is not None:
        evento["cambios"] = {
            k: {"antes": anteriores.get(k), "despues": v}
            for k, v in finales.items()
            if anteriores.get(k) != v
        }
    return evento


def construir_resultado(id_aprobacion: str, tipo: str, operador: str, resultado: dict, contexto: dict) -> dict:
    """Linea de "resultado": lo que paso despues de una aprobacion (todavia no la escribe).

    resultado: estado, nombre, archivos, etc. de lo que se genero (o del error).
    contexto:  con que se genero, para poder reproducirlo (parametros vigentes, modelo LLM, campo de costo...).
    """
    return {
        "ts": _ahora(),
        "evento": "resultado",
        "aprobacion_id": id_aprobacion,
        "operador": operador,
        "tipo": tipo,
        "resultado": resultado,
        "contexto": contexto,
    }


def registrar(evento: dict, ruta: Path | None = None) -> Path:
    """Agrega el evento como una linea al final de la bitacora. Lanza OSError si no puede escribir."""
    destino = Path(ruta) if ruta is not None else ruta_actual()
    destino.parent.mkdir(parents=True, exist_ok=True)
    linea = json.dumps(evento, ensure_ascii=False, default=str)  # sin saltos de linea: una decision = una linea
    with destino.open("a", encoding="utf-8") as f:
        f.write(linea + "\n")
    return destino


def leer(ruta: Path | None = None) -> tuple[list[dict], int]:
    """Lee la bitacora. Devuelve (eventos, lineas_danadas). Una linea ilegible no tira el resto."""
    origen = Path(ruta) if ruta is not None else ruta_actual()
    try:
        texto = origen.read_text(encoding="utf-8")
    except FileNotFoundError:
        return [], 0
    eventos, danadas = [], 0
    for linea in texto.splitlines():
        if not linea.strip():
            continue
        try:
            dato = json.loads(linea)
        except json.JSONDecodeError:
            danadas += 1
            continue
        if isinstance(dato, dict):
            eventos.append(dato)
        else:
            danadas += 1
    return eventos, danadas
