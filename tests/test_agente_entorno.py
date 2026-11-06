"""
Pruebas de las herramientas del entorno en agente.py (consultar_entorno, calcular_ruta, generar_ducto).

LangChain/LangGraph/Ollama NO se usan de verdad: se instalan modulos falsos minimos en sys.modules para poder
importar agente.py, y las herramientas quedan como funciones normales (el decorador @tool falso no hace nada).
interrupt() tambien es falso: devuelve la decision que cada prueba quiera simular. Asi se prueba el PEGAMENTO
(que se pregunta al humano, con que numeros, que se le cuenta al LLM), no el LLM.

Ejecutar desde la raiz del proyecto:  pytest tests -v
"""
import importlib
import sys
import types

import pytest

from herramientas import auditoria


@pytest.fixture(autouse=True)
def bitacora(tmp_path, monkeypatch):
    """Cada prueba escribe su bitacora en una carpeta temporal: nunca ensucia data/auditoria del proyecto."""
    ruta = tmp_path / "auditoria" / "aprobaciones.jsonl"
    monkeypatch.setenv("GM_AUDITORIA", str(ruta))
    monkeypatch.delenv("GM_OPERADOR", raising=False)
    return ruta


def _modulo(nombre, **atributos):
    m = types.ModuleType(nombre)
    m.__dict__.update(atributos)
    return m


def _tool_falso(f=None, **opciones):
    """@tool y @tool(return_direct=True): la funcion queda tal cual, y se anota que opciones recibio."""
    def marcar(g):
        g.opciones_tool = opciones
        return g
    return marcar(f) if f is not None else marcar


def _importar_agente(monkeypatch, decision=None):
    llamadas = []

    def interrupt_falso(carga):
        llamadas.append(carga)
        return decision

    falsos = {
        "langchain": _modulo("langchain"),
        "langchain.agents": _modulo("langchain.agents", create_agent=lambda **kw: object()),
        "langchain_core": _modulo("langchain_core"),
        "langchain_core.tools": _modulo("langchain_core.tools", tool=_tool_falso),
        "langchain_ollama": _modulo("langchain_ollama", ChatOllama=lambda **kw: object()),
        "langgraph": _modulo("langgraph"),
        "langgraph.checkpoint": _modulo("langgraph.checkpoint"),
        "langgraph.checkpoint.memory": _modulo("langgraph.checkpoint.memory", InMemorySaver=lambda: object()),
        "langgraph.types": _modulo("langgraph.types", Command=dict, interrupt=interrupt_falso),
    }
    for nombre, modulo in falsos.items():
        monkeypatch.setitem(sys.modules, nombre, modulo)
    monkeypatch.delitem(sys.modules, "LLM.agente", raising=False)
    ag = importlib.import_module("LLM.agente")
    monkeypatch.setitem(sys.modules, "LLM.agente", ag)  # para que se limpie al terminar la prueba
    return ag, llamadas


BLOQUE = {"id": "m", "nombre": "Motor", "tipo": "termico", "valor": 100.0,
          "centro": {"x": 1.0, "y": 2.0, "z": 3.0}, "size": {"x": 4.0, "y": 5.0, "z": 6.0}}


def _ruteo_falso(monkeypatch, ag, bloques=(BLOQUE,)):
    r = types.SimpleNamespace(
        listar_bloques=lambda: list(bloques),
        calculo_ruta=lambda p0, p1, cfg=None: {"estado": "ok", "ruta": [p0, p1], "longitud_mm": 10.0,
                                              "bloques_cercanos": ["Motor"], "mensaje": ""},
    )
    monkeypatch.setattr(ag, "ruteo_astar", r)
    return r


def _reporte_ok():
    return {"estado": "ok", "nombre": "ducto_x", "longitud_mm": 100.0, "espesor_mm": 2.0, "volumen_mm3": 5000.0,
            "solido_valido": True,
            "seccion": {"area_mm2": 1000.0, "relacion_aspecto": 1.6, "diametro_hidraulico_mm": 30.0, "area_ok": True, "aspecto_ok": True},
            "radio_curvatura": {"cumple": True, "radio_minimo_mm": 47.25, "r_min_exigido_mm": 45.0},
            "curva": {"factible": True, "holguras_mm": {"Motor": 80.0}, "colisiones": [], "dentro_de_zona_de_seguridad": []},
            "archivos": {"step": "ducto_x.step", "stl": "ducto_x.stl"}}


# ---- consultar_entorno / calcular_ruta -----------------------------------------------------------
def test_consultar_entorno_lista_los_bloques(monkeypatch):
    ag, _ = _importar_agente(monkeypatch)
    _ruteo_falso(monkeypatch, ag)
    assert "Motor (termico, valor 100)" in ag.consultar_entorno()


def test_sin_ruteo_astar_las_herramientas_avisan_en_vez_de_romperse(monkeypatch):
    ag, llamadas = _importar_agente(monkeypatch)
    monkeypatch.setattr(ag, "ruteo_astar", None)
    assert "ruteo_astar" in ag.consultar_entorno()
    assert "ruteo_astar" in ag.calcular_ruta([0, 0, 0], [1, 1, 1])
    assert "ruteo_astar" in ag.generar_ducto([0, 0, 0], [1, 1, 1], 40, 25, 2)
    assert llamadas == []


def test_calcular_ruta_devuelve_resumen_y_marca_de_imagen(monkeypatch, tmp_path):
    ag, _ = _importar_agente(monkeypatch)
    _ruteo_falso(monkeypatch, ag)
    monkeypatch.setattr(ag, "RUTA_SALIDAS", tmp_path)
    fake_viz = types.SimpleNamespace(visualizar_ruta=lambda p0, p1, cfg, ruta_png: {"imagen": ruta_png})
    monkeypatch.setitem(sys.modules, "visualizar_ruteo", fake_viz)

    texto = ag.calcular_ruta([0, 0, 500], [2300, 0, 500], 25.0)

    assert "2 puntos" in texto and "Motor" in texto
    archivos = ag.informe_ducto.archivos_de_mensajes([types.SimpleNamespace(name="calcular_ruta", content=texto)])
    assert [a["tipo"] for a in archivos] == ["imagen"] and archivos[0]["ruta"].startswith(str(tmp_path))


def test_calcular_ruta_con_punto_mal_formado_no_revienta(monkeypatch):
    ag, _ = _importar_agente(monkeypatch)
    _ruteo_falso(monkeypatch, ag)
    assert "tres numeros" in ag.calcular_ruta([0, 0], [1, 1, 1])


# ---- generar_ducto: aprobacion humana ------------------------------------------------------------
def _preparar_generar(monkeypatch, decision, reporte=None):
    ag, llamadas = _importar_agente(monkeypatch, decision)
    _ruteo_falso(monkeypatch, ag)
    monkeypatch.setattr(ag, "PARAMETROS", {"r_min_mm": 45.0})
    recibido = {}

    def ducto_falso(p0, p1, ancho, alto, espesor, **kw):
        recibido.update(p0=p0, p1=p1, ancho=ancho, alto=alto, espesor=espesor, **kw)
        return reporte or _reporte_ok()

    monkeypatch.setattr(ag.curva_ruta, "ducto_desde_entorno", ducto_falso)
    return ag, llamadas, recibido


def test_generar_ducto_sin_parametros_aprobados_no_pregunta_ni_genera(monkeypatch):
    ag, llamadas, recibido = _preparar_generar(monkeypatch, {"decision": "approve"})
    monkeypatch.setattr(ag, "PARAMETROS", {})
    assert "registrar_parametros" in ag.generar_ducto([0, 0, 0], [1, 0, 0], 40, 25, 2)
    assert llamadas == [] and recibido == {}


def test_generar_ducto_pregunta_con_numeros_planos_y_luego_genera(monkeypatch):
    ag, llamadas, recibido = _preparar_generar(monkeypatch, {"decision": "approve"})

    texto = ag.generar_ducto([0, 0, 500], [2300, 0, 500], 40, 25, 2)

    pausa = llamadas[0]
    assert pausa["tipo"] == "aprobar_ducto" and pausa["actuales"] == {}
    assert pausa["propuesta"] == {"inicio_x": 0.0, "inicio_y": 0.0, "inicio_z": 500.0, "fin_x": 2300.0, "fin_y": 0.0,
                                  "fin_z": 500.0, "ancho_mm": 40.0, "alto_mm": 25.0, "espesor_mm": 2.0}
    assert recibido["p0"] == [0.0, 0.0, 500.0] and recibido["ancho"] == 40.0
    assert recibido["parametros"] == {"r_min_mm": 45.0} and recibido["generar_si_infactible"] is False
    assert "47.25" in texto and "[ARCHIVO_CAD:" in texto and "ducto_x.step" in texto


def test_generar_ducto_con_edicion_usa_los_valores_del_humano_y_se_lo_dice_al_llm(monkeypatch):
    ag, _, recibido = _preparar_generar(monkeypatch, {"decision": "edit", "valores": {"espesor_mm": 3.0}})
    texto = ag.generar_ducto([0, 0, 500], [2300, 0, 500], 40, 25, 2)
    assert recibido["espesor"] == 3.0 and recibido["ancho"] == 40.0
    assert "corrigio" in texto and "3.0" in texto


def test_generar_ducto_rechazado_no_genera_nada(monkeypatch):
    ag, _, recibido = _preparar_generar(monkeypatch, {"decision": "reject", "motivo": "seccion muy chica"})
    texto = ag.generar_ducto([0, 0, 500], [2300, 0, 500], 40, 25, 2)
    assert "RECHAZO" in texto and "seccion muy chica" in texto and recibido == {}


def test_generar_ducto_infactible_se_lo_cuenta_al_llm_sin_archivos(monkeypatch):
    rep = {"estado": "infactible", "mensaje": "No se puede cumplir R_min = 45 mm", "radio_maximo_factible_mm": 30,
           "propuestas": [{"id": "bajar_r_min", "descripcion": "Bajar a 30", "requiere_aprobacion": True}]}
    ag, _, _ = _preparar_generar(monkeypatch, {"decision": "approve"}, reporte=rep)
    texto = ag.generar_ducto([0, 0, 500], [2300, 0, 500], 40, 25, 2)
    assert "INFACTIBLE" in texto and "bajar_r_min" in texto and "ARCHIVO_CAD" not in texto


def test_generar_ducto_pasa_la_opcion_generar_si_infactible(monkeypatch):
    ag, _, recibido = _preparar_generar(monkeypatch, {"decision": "approve"})
    ag.generar_ducto([0, 0, 500], [2300, 0, 500], 40, 25, 2, generar_si_infactible=True)
    assert recibido["generar_si_infactible"] is True


def test_generar_ducto_si_falta_cadquery_avisa_sin_reventar(monkeypatch):
    ag, _, _ = _preparar_generar(monkeypatch, {"decision": "approve"})

    def sin_cadquery(*a, **kw):
        raise ModuleNotFoundError("No module named 'cadquery'")

    monkeypatch.setattr(ag.curva_ruta, "ducto_desde_entorno", sin_cadquery)
    assert "falta una libreria" in ag.generar_ducto([0, 0, 500], [2300, 0, 500], 40, 25, 2)


def test_la_interfaz_no_ve_las_marcas_en_la_respuesta_del_llm(monkeypatch):
    ag, _ = _importar_agente(monkeypatch)
    limpio = ag._limpiar_respuesta("Listo [IMAGEN_RUTA: a.png] [Contexto interno: x] fin [ARCHIVO_CAD: b.step]")
    assert limpio.split() == ["Listo", "fin"]


def test_las_herramientas_de_lectura_devuelven_directo_y_las_que_escriben_no(monkeypatch):
    ag, _ = _importar_agente(monkeypatch)
    assert ag.consultar_entorno.opciones_tool == {"return_direct": True}
    assert ag.calcular_ruta.opciones_tool == {"return_direct": True}
    # generar_ducto necesita que el modelo explique infactibilidad/propuestas y pregunte: NO es directa
    assert ag.generar_ducto.opciones_tool == {}


# ---- bitacora de auditoria: se anota la decision del humano, y solo una vez ----------------------
def test_aprobar_un_ducto_queda_en_la_bitacora_con_quien_lo_firmo(monkeypatch, bitacora):
    ag, _, _ = _preparar_generar(monkeypatch, {"decision": "approve", "operador": "Carlos"})
    ag.generar_ducto([0, 0, 500], [2300, 0, 500], 40, 25, 2)
    eventos, danadas = auditoria.leer(bitacora)
    assert danadas == 0 and len(eventos) == 2  # una aprobacion + una linea de resultado, ni una mas
    ev = eventos[0]
    assert ev["evento"] == "aprobacion" and ev["id"]
    assert ev["tipo"] == "ducto" and ev["decision"] == "approve" and ev["operador"] == "Carlos"
    assert ev["finales"]["ancho_mm"] == 40.0 and ev["finales"] == ev["propuesta"]


def test_editar_un_ducto_anota_la_correccion_humana(monkeypatch, bitacora):
    ag, _, _ = _preparar_generar(monkeypatch, {"decision": "edit", "valores": {"espesor_mm": 3.0}})
    ag.generar_ducto([0, 0, 500], [2300, 0, 500], 40, 25, 2)
    ev = auditoria.leer(bitacora)[0][0]
    assert ev["decision"] == "edit"
    assert ev["correcciones_humanas"] == {"espesor_mm": {"propuesto": 2.0, "final": 3.0}}
    assert ev["finales"]["espesor_mm"] == 3.0


def test_rechazar_un_ducto_tambien_se_anota(monkeypatch, bitacora):
    ag, _, recibido = _preparar_generar(monkeypatch, {"decision": "reject", "motivo": "seccion muy chica"})
    ag.generar_ducto([0, 0, 500], [2300, 0, 500], 40, 25, 2)
    ev = auditoria.leer(bitacora)[0][0]
    assert ev["decision"] == "reject" and ev["motivo"] == "seccion muy chica" and ev["finales"] is None
    assert recibido == {}


def test_si_no_hay_decision_humana_no_se_anota_nada(monkeypatch, bitacora):
    ag, llamadas, _ = _preparar_generar(monkeypatch, {"decision": "approve"})
    monkeypatch.setattr(ag, "PARAMETROS", {})  # la herramienta se detiene antes de preguntarle al humano
    ag.generar_ducto([0, 0, 500], [2300, 0, 500], 40, 25, 2)
    assert llamadas == [] and not bitacora.exists()


def _preparar_parametros(monkeypatch, tmp_path, decision, actuales=None):
    ag, _ = _importar_agente(monkeypatch, decision)
    monkeypatch.setattr(ag, "PARAMETROS", dict(actuales or {}))
    monkeypatch.setattr(ag, "RUTA_PARAMETROS", tmp_path / "parametros.json")  # no tocar el JSON real del proyecto
    return ag


def test_editar_parametros_anota_antes_despues_y_la_correccion(monkeypatch, tmp_path, bitacora):
    ag = _preparar_parametros(monkeypatch, tmp_path, {"decision": "edit", "valores": {"r_min_mm": 55.0}},
                              actuales={"r_min_mm": 45.0})
    texto = ag._aprobar_y_guardar({"r_min_mm": 60.0, "area_min_mm2": 300.0})
    assert "55.0" in texto and ag.PARAMETROS == {"r_min_mm": 55.0, "area_min_mm2": 300.0}
    ev = auditoria.leer(bitacora)[0][0]
    assert ev["tipo"] == "parametros" and ev["decision"] == "edit"
    assert ev["cambios"] == {"r_min_mm": {"antes": 45.0, "despues": 55.0},
                             "area_min_mm2": {"antes": None, "despues": 300.0}}
    assert ev["correcciones_humanas"] == {"r_min_mm": {"propuesto": 60.0, "final": 55.0}}


def test_rechazar_parametros_no_cambia_nada_pero_se_anota(monkeypatch, tmp_path, bitacora):
    ag = _preparar_parametros(monkeypatch, tmp_path, {"decision": "reject", "motivo": "valor inventado"},
                              actuales={"r_min_mm": 45.0})
    texto = ag._aprobar_y_guardar({"r_min_mm": 60.0})
    assert "RECHAZO" in texto and ag.PARAMETROS == {"r_min_mm": 45.0}
    ev = auditoria.leer(bitacora)[0][0]
    assert ev["decision"] == "reject" and ev["motivo"] == "valor inventado" and ev["finales"] is None


def test_si_la_bitacora_falla_la_aprobacion_sigue_valida_y_se_avisa(monkeypatch, tmp_path):
    archivo = tmp_path / "soy_un_archivo"
    archivo.write_text("x")
    monkeypatch.setenv("GM_AUDITORIA", str(archivo / "log.jsonl"))  # imposible de escribir
    ag = _preparar_parametros(monkeypatch, tmp_path, {"decision": "approve"})
    texto = ag._aprobar_y_guardar({"r_min_mm": 45.0})
    assert ag.PARAMETROS == {"r_min_mm": 45.0}  # el humano si aprobo: no se pierde
    assert "AVISO" in texto and "bitacora" in texto


def test_si_la_bitacora_falla_generar_ducto_genera_igual_y_avisa(monkeypatch, tmp_path):
    archivo = tmp_path / "soy_un_archivo"
    archivo.write_text("x")
    ag, _, recibido = _preparar_generar(monkeypatch, {"decision": "approve"})
    monkeypatch.setenv("GM_AUDITORIA", str(archivo / "log.jsonl"))
    texto = ag.generar_ducto([0, 0, 500], [2300, 0, 500], 40, 25, 2)
    assert recibido["ancho"] == 40.0 and "ARCHIVO_CAD" in texto and "AVISO" in texto


# ---- bitacora: la linea de RESULTADO se enlaza con la aprobacion --------------------------------------
def test_el_resultado_lleva_el_id_de_la_aprobacion_y_el_contexto_para_reproducirlo(monkeypatch, bitacora):
    ag, _, _ = _preparar_generar(monkeypatch, {"decision": "approve", "operador": "Carlos"})
    ag.generar_ducto([0, 0, 500], [2300, 0, 500], 40, 25, 2)
    aprobacion, resultado = auditoria.leer(bitacora)[0]
    assert resultado["evento"] == "resultado" and resultado["aprobacion_id"] == aprobacion["id"]
    assert resultado["operador"] == "Carlos" and resultado["tipo"] == "ducto"
    r = resultado["resultado"]
    assert r["estado"] == "ok" and r["archivos"] == {"step": "ducto_x.step", "stl": "ducto_x.stl"}
    assert r["radio_minimo_mm"] == 47.25 and r["r_min_exigido_mm"] == 45.0 and r["solido_valido"] is True
    c = resultado["contexto"]
    assert c["parametros_vigentes"] == {"r_min_mm": 45.0}  # con que r_min se genero
    assert c["modelo_llm"] == ag.NOMBRE_MODELO_LLM == "llama3.1"
    assert c["campo_costo"] == "ninguno" and c["generar_si_infactible"] is False


def test_el_resultado_anota_los_parametros_vigentes_al_generar_no_los_de_despues(monkeypatch, bitacora):
    ag, _, _ = _preparar_generar(monkeypatch, {"decision": "approve"})
    ag.generar_ducto([0, 0, 500], [2300, 0, 500], 40, 25, 2)
    ag.PARAMETROS["r_min_mm"] = 99.0  # alguien cambia r_min despues: la linea ya escrita no debe cambiar
    assert auditoria.leer(bitacora)[0][1]["contexto"]["parametros_vigentes"] == {"r_min_mm": 45.0}


def test_un_ducto_infactible_queda_como_infactible_en_el_resultado(monkeypatch, bitacora):
    rep = {"estado": "infactible", "mensaje": "No se puede cumplir R_min = 45 mm", "radio_maximo_factible_mm": 30,
           "propuestas": [{"id": "bajar_r_min", "descripcion": "Bajar a 30", "requiere_aprobacion": True}]}
    ag, _, _ = _preparar_generar(monkeypatch, {"decision": "approve"}, reporte=rep)
    ag.generar_ducto([0, 0, 500], [2300, 0, 500], 40, 25, 2)
    r = auditoria.leer(bitacora)[0][1]["resultado"]
    assert r["estado"] == "infactible" and "R_min" in r["mensaje"] and "archivos" not in r
    assert "propuestas" not in r  # la bitacora guarda lo esencial, no el reporte completo


def test_si_falta_cadquery_el_resultado_dice_error_pero_la_aprobacion_queda(monkeypatch, bitacora):
    ag, _, _ = _preparar_generar(monkeypatch, {"decision": "approve"})

    def sin_cadquery(*a, **kw):
        raise ModuleNotFoundError("No module named 'cadquery'")

    monkeypatch.setattr(ag.curva_ruta, "ducto_desde_entorno", sin_cadquery)
    ag.generar_ducto([0, 0, 500], [2300, 0, 500], 40, 25, 2)
    aprobacion, resultado = auditoria.leer(bitacora)[0]
    assert aprobacion["decision"] == "approve"
    assert resultado["resultado"]["estado"] == "error" and "cadquery" in resultado["resultado"]["mensaje"]
    assert resultado["resultado"]["nombre"].startswith("ducto_")


def test_si_se_pasa_generar_si_infactible_el_contexto_lo_dice(monkeypatch, bitacora):
    ag, _, _ = _preparar_generar(monkeypatch, {"decision": "approve"})
    ag.generar_ducto([0, 0, 500], [2300, 0, 500], 40, 25, 2, generar_si_infactible=True)
    assert auditoria.leer(bitacora)[0][1]["contexto"]["generar_si_infactible"] is True


def test_un_ducto_rechazado_no_tiene_linea_de_resultado(monkeypatch, bitacora):
    ag, _, _ = _preparar_generar(monkeypatch, {"decision": "reject", "motivo": "no"})
    ag.generar_ducto([0, 0, 500], [2300, 0, 500], 40, 25, 2)
    eventos, _ = auditoria.leer(bitacora)
    assert [e["evento"] for e in eventos] == ["aprobacion"]
