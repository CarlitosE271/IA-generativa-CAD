"""
Texto y datos que el AGENTE (LLM) y la INTERFAZ necesitan alrededor del entorno de bloques, la ruta y
el ducto. Todo es logica pura (sin LangChain, sin CadQuery, sin Streamlit), por eso se puede probar.

Por que existe un modulo aparte:
  - El LLM lee TEXTO, no dicts: se le da un resumen corto y con los numeros ya calculados, para que no
    tenga que "interpretar" un JSON enorme (y para que no invente cifras).
  - Los archivos que genera una herramienta (PNG de la ruta, STEP/STL del ducto) se avisan con una
    marca [IMAGEN_RUTA: ...] / [ARCHIVO_CAD: ...] dentro del resultado de la herramienta. La interfaz
    las recoge de los mensajes del hilo (como ya se hace con las fuentes de buscar_en_documentos), asi
    cada pestana ve SUS archivos sin variables globales compartidas.
"""
from __future__ import annotations

import re

MARCA_IMAGEN = "IMAGEN_RUTA"
MARCA_CAD = "ARCHIVO_CAD"
PATRON_MARCAS = re.compile(r"\[(?:%s|%s):\s*[^\]]*\]" % (MARCA_IMAGEN, MARCA_CAD))
_PATRON_ARCHIVO = re.compile(r"\[(%s|%s):\s*([^\]]*?)\s*\]" % (MARCA_IMAGEN, MARCA_CAD))
HERRAMIENTAS_CON_ARCHIVOS = ("calcular_ruta", "generar_ducto")

CAMPOS_PROPUESTA = (
    "inicio_x", "inicio_y", "inicio_z", "fin_x", "fin_y", "fin_z", "ancho_mm", "alto_mm", "espesor_mm",
)


# --------------------------------------------------------------------------
# Argumentos <-> propuesta plana (para reutilizar el formulario aprobar/editar/rechazar)
# --------------------------------------------------------------------------
def punto_3d(valor, nombre: str) -> list[float]:
    """[x, y, z] -> lista de 3 floats; ValueError con mensaje claro si no sirve."""
    try:
        p = [float(v) for v in valor]
    except (TypeError, ValueError):
        raise ValueError(f"'{nombre}' debe ser una lista de tres numeros [x, y, z] en mm.")
    if len(p) != 3:
        raise ValueError(f"'{nombre}' debe tener exactamente tres numeros [x, y, z] en mm.")
    return p


def propuesta_de_ducto(inicio, fin, ancho_mm: float, alto_mm: float, espesor_mm: float) -> dict:
    """Aplana los argumentos a un dict {campo: float}: el formulario del humano solo maneja numeros planos
    y, de paso, la deteccion de 'valores que no dijiste tu' revisa cada uno."""
    i, f = punto_3d(inicio, "inicio"), punto_3d(fin, "fin")
    return dict(zip(CAMPOS_PROPUESTA, [*i, *f, float(ancho_mm), float(alto_mm), float(espesor_mm)]))


def argumentos_de_propuesta(p: dict):
    """Inversa de propuesta_de_ducto -> (inicio, fin, ancho_mm, alto_mm, espesor_mm)."""
    return (
        [p["inicio_x"], p["inicio_y"], p["inicio_z"]],
        [p["fin_x"], p["fin_y"], p["fin_z"]],
        p["ancho_mm"], p["alto_mm"], p["espesor_mm"],
    )


# --------------------------------------------------------------------------
# Texto para el LLM
# --------------------------------------------------------------------------
def _n(v) -> str:
    """Numero corto: 12.0 -> '12', 12.345 -> '12.35'."""
    return f"{float(v):.2f}".rstrip("0").rstrip(".")


def texto_entorno(bloques: list[dict]) -> str:
    if not bloques:
        return "El entorno esta vacio: no hay bloques. Los bloques los coloca el usuario en la interfaz de bloques."
    lineas = [f"Bloques del entorno ({len(bloques)}), todo en mm:"]
    for b in bloques:
        c, s = b["centro"], b["size"]
        valor = f", valor {_n(b['valor'])}" if b["tipo"] != "neutro" else ""
        lineas.append(
            f"- {b['nombre']} ({b['tipo']}{valor}): centro ({_n(c['x'])}, {_n(c['y'])}, {_n(c['z'])}), "
            f"tamano ({_n(s['x'])} x {_n(s['y'])} x {_n(s['z'])})"
        )
    return "\n".join(lineas)


def texto_ruta(res: dict, imagen: str | None = None) -> str:
    """Resumen del resultado de ruteo_astar.calculo_ruta (+ la marca de la imagen si se pudo dibujar)."""
    if res["estado"] != "ok":
        return f"El A* no encontro ruta (estado: {res['estado']}). Motivo: {res['mensaje']}"
    lineas = [
        f"Ruta del A* calculada: {len(res['ruta'])} puntos, longitud {_n(res['longitud_mm'])} mm.",
        "Bloques cercanos a la ruta: " + (", ".join(res["bloques_cercanos"]) or "ninguno") + ".",
        "Esta es la ruta cruda del A* (en zigzag); aun no es el ducto.",
    ]
    if imagen:
        lineas.append(f"[{MARCA_IMAGEN}: {imagen}]")
    return "\n".join(lineas)


def texto_reporte(rep: dict, rutas_cad: list[str] | None = None) -> str:
    """Resumen de ducto_desde_entorno() para el LLM, para los tres estados que puede devolver."""
    estado = rep.get("estado")
    if estado in ("sin_ruta", "error"):
        return f"No se pudo generar el ducto (estado: {estado}). Motivo: {rep.get('mensaje', '')}"

    if estado == "infactible":
        lineas = [
            "ESTADO: INFACTIBLE. No se genero ningun archivo CAD.",
            rep["mensaje"],
        ]
        r_max = rep.get("radio_maximo_factible_mm")
        if r_max:
            lineas.append(f"Con este entorno el mayor R_min que si cabe es {r_max} mm.")
        lineas.append("Propuestas para el usuario:")
        for p in rep["propuestas"]:
            nota = " (requiere aprobacion del usuario)" if p.get("requiere_aprobacion") else ""
            lineas.append(f"- {p['id']}: {p['descripcion']}{nota}")
        lineas.append(
            "INSTRUCCIONES: anuncia el problema al usuario con estos numeros y preguntale que propuesta elige. "
            "NO cambies r_min por tu cuenta. Si elige bajar_r_min usa actualizar_parametro con r_min_mm "
            "(el usuario lo aprobara en el formulario) y luego vuelve a llamar generar_ducto. Si elige "
            "generar_igual vuelve a llamar generar_ducto con generar_si_infactible=true. Si elige "
            "mover_entorno, dile que mueva los bloques en la interfaz y luego pidele que lo intente de nuevo."
        )
        return "\n".join(lineas)

    # estado == "ok"
    sec, rad = rep["seccion"], rep["radio_curvatura"]
    curva = rep.get("curva", {})
    lineas = [
        f"Ducto '{rep['nombre']}' generado. Longitud {_n(rep['longitud_mm'])} mm, espesor {_n(rep['espesor_mm'])} mm, "
        f"volumen {_n(rep['volumen_mm3'])} mm3, solido valido: {'si' if rep['solido_valido'] else 'NO'}.",
        f"Seccion: area {_n(sec['area_mm2'])} mm2, relacion de aspecto {_n(sec['relacion_aspecto'])}, "
        f"diametro hidraulico {_n(sec['diametro_hidraulico_mm'])} mm. "
        f"Area dentro de limites: {_si_no(sec.get('area_ok'))}; aspecto dentro del limite: {_si_no(sec.get('aspecto_ok'))}.",
    ]
    if rad.get("radio_minimo_mm") is None:
        lineas.append("Radio de curvatura: ruta recta, sin curvas.")
    else:
        lineas.append(
            f"Radio de curvatura minimo medido: {_n(rad['radio_minimo_mm'])} mm "
            f"(exigido: {_n(rad['r_min_exigido_mm']) if 'r_min_exigido_mm' in rad else 'sin r_min aprobado'}), "
            f"cumple: {_si_no(rad['cumple'])}."
        )
    if "holguras_mm" in curva:
        h = ", ".join(f"{k}: {_n(v)} mm" for k, v in curva["holguras_mm"].items()) or "sin bloques"
        lineas.append(f"Holgura del ducto a cada bloque: {h}.")
        if curva.get("colisiones"):
            lineas.append("COLISION con: " + ", ".join(curva["colisiones"]) + ".")
        if curva.get("dentro_de_zona_de_seguridad"):
            lineas.append("Dentro de la zona de seguridad de: " + ", ".join(curva["dentro_de_zona_de_seguridad"]) + ".")
        if not curva.get("factible", True):
            lineas.append("OJO: este ducto se genero aunque NO cumple R_min en todas las esquinas (el usuario lo pidio).")
    elif "aviso" in curva:
        lineas.append("AVISO: " + curva["aviso"])
    lineas.append("Los numeros anteriores vienen del calculo: no los modifiques ni inventes otros.")
    for ruta in rutas_cad or []:
        lineas.append(f"[{MARCA_CAD}: {ruta}]")
    return "\n".join(lineas)


def _si_no(v) -> str:
    return {True: "si", False: "NO", None: "no verificado"}.get(v, str(v))


# --------------------------------------------------------------------------
# Archivos generados -> interfaz
# --------------------------------------------------------------------------
def archivos_de_mensajes(mensajes) -> list[dict]:
    """Recoge las marcas de archivo de los resultados de calcular_ruta/generar_ducto dentro de `mensajes`.
    [{"tipo": "imagen" | "cad", "ruta": str}], sin repetidos, en orden."""
    vistos, salida = set(), []
    for m in mensajes:
        if getattr(m, "name", None) not in HERRAMIENTAS_CON_ARCHIVOS:
            continue
        for marca, ruta in _PATRON_ARCHIVO.findall(str(getattr(m, "content", ""))):
            if ruta and ruta not in vistos:
                vistos.add(ruta)
                salida.append({"tipo": "imagen" if marca == MARCA_IMAGEN else "cad", "ruta": ruta})
    return salida


def quitar_marcas(texto: str) -> str:
    """Quita las marcas por si el LLM las copia en su respuesta."""
    return PATRON_MARCAS.sub("", texto)


# --------------------------------------------------------------------------
# Transparencia: que hizo el agente y que hacer si el modelo no logra redactar
# --------------------------------------------------------------------------
_PATRON_INUTIL = re.compile(
    r"(lo siento|disculpa|perd[oó]n)[^.]{0,60}no puedo (continuar|ayudar|responder|procesar|completar)", re.IGNORECASE
)


def respuesta_inutil(texto: str) -> bool:
    """True si la respuesta del LLM esta vacia o es solo una disculpa generica ('Lo siento, pero no puedo
    continuar...'). Un modelo local chico a veces se bloquea asi despues de ejecutar una herramienta
    que SI funciono; en ese caso la interfaz muestra el resultado de la herramienta."""
    t = (texto or "").strip()
    return not t or (len(t) < 200 and bool(_PATRON_INUTIL.search(t)))


def traza_herramientas(mensajes, max_caracteres: int = 1500) -> list[dict]:
    """Pasos del agente en `mensajes`: [{"tipo": "llamada", "herramienta", "argumentos"} |
    {"tipo": "resultado", "herramienta", "texto"}]. Se muestra al humano (y al panel) para que vea que
    herramienta se llamo, con que argumentos y que devolvio, sin fiarse solo del texto del LLM."""
    pasos = []
    for m in mensajes:
        for tc in getattr(m, "tool_calls", None) or []:
            pasos.append({"tipo": "llamada", "herramienta": tc.get("name"), "argumentos": tc.get("args")})
        if getattr(m, "type", None) == "tool":
            texto = quitar_marcas(str(getattr(m, "content", ""))).strip()
            if len(texto) > max_caracteres:
                texto = texto[:max_caracteres] + " [...]"
            pasos.append({"tipo": "resultado", "herramienta": getattr(m, "name", None), "texto": texto})
    return pasos


def ultimo_resultado(mensajes) -> str | None:
    """Texto del ultimo resultado de herramienta (sin marcas de archivo), o None si no hubo ninguno."""
    for m in reversed(list(mensajes)):
        if getattr(m, "type", None) == "tool":
            return quitar_marcas(str(getattr(m, "content", ""))).strip() or None
    return None
