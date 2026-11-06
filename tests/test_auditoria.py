"""
Pruebas de la bitacora de auditoria (src/herramientas/auditoria.py).

No usan LangChain ni Streamlit. Ejecutar desde la raiz del proyecto:  pytest tests/test_auditoria.py -v
"""
import json

import pytest

from herramientas import auditoria


@pytest.fixture(autouse=True)
def _sin_variables(monkeypatch):
    """Que el entorno real de quien corre las pruebas no cambie los resultados."""
    monkeypatch.delenv("GM_OPERADOR", raising=False)
    monkeypatch.delenv("GM_AUDITORIA", raising=False)


# ---- construir_evento -----------------------------------------------------------------------------
def test_aprobar_registra_propuesta_finales_y_cambios_contra_lo_anterior():
    ev = auditoria.construir_evento(
        "parametros", {"decision": "approve", "operador": "Carlos"},
        propuesta={"r_min_mm": 45.0, "aspect_ratio_max": 3.0}, finales={"r_min_mm": 45.0, "aspect_ratio_max": 3.0},
        anteriores={"r_min_mm": 178.0, "aspect_ratio_max": 3.0},
    )
    assert ev["tipo"] == "parametros" and ev["decision"] == "approve" and ev["operador"] == "Carlos"
    assert ev["finales"] == {"r_min_mm": 45.0, "aspect_ratio_max": 3.0}
    # solo aparece lo que realmente cambio (aspect_ratio_max ya valia 3.0)
    assert ev["cambios"] == {"r_min_mm": {"antes": 178.0, "despues": 45.0}}
    assert "correcciones_humanas" not in ev  # aprobo tal cual: el humano no corrigio nada


def test_primera_vez_el_valor_anterior_es_none():
    ev = auditoria.construir_evento("parametros", {"decision": "approve"}, {"r_min_mm": 45.0}, {"r_min_mm": 45.0}, {})
    assert ev["cambios"] == {"r_min_mm": {"antes": None, "despues": 45.0}}


def test_editar_registra_donde_el_humano_corrigio_a_la_ia():
    ev = auditoria.construir_evento(
        "ducto", {"decision": "edit", "valores": {"espesor_mm": 3.0}},
        propuesta={"ancho_mm": 40.0, "espesor_mm": 2.0}, finales={"ancho_mm": 40.0, "espesor_mm": 3.0},
    )
    assert ev["decision"] == "edit"
    assert ev["correcciones_humanas"] == {"espesor_mm": {"propuesto": 2.0, "final": 3.0}}
    assert "cambios" not in ev  # para ductos no hay "valor anterior"


def test_rechazar_guarda_motivo_y_no_hay_finales():
    ev = auditoria.construir_evento(
        "ducto", {"decision": "reject", "motivo": "seccion muy chica"}, {"ancho_mm": 40.0}, None
    )
    assert ev["decision"] == "reject" and ev["motivo"] == "seccion muy chica" and ev["finales"] is None
    assert "cambios" not in ev and "correcciones_humanas" not in ev


def test_una_decision_desconocida_se_trata_como_rechazo_igual_que_en_el_agente():
    ev = auditoria.construir_evento("ducto", {"decision": "tal vez"}, {"ancho_mm": 40.0}, {"ancho_mm": 40.0})
    assert ev["decision"] == "reject" and ev["finales"] is None


def test_motivo_vacio_queda_como_none():
    ev = auditoria.construir_evento("ducto", {"decision": "reject", "motivo": ""}, {}, None)
    assert ev["motivo"] is None


def test_el_operador_de_la_interfaz_gana_a_la_variable_de_entorno(monkeypatch):
    monkeypatch.setenv("GM_OPERADOR", "Fer")
    firmado = auditoria.construir_evento("ducto", {"decision": "approve", "operador": "Maikol"}, {}, {})
    sin_firma = auditoria.construir_evento("ducto", {"decision": "approve", "operador": "   "}, {}, {})
    assert firmado["operador"] == "Maikol"
    assert sin_firma["operador"] == "Fer"  # campo vacio en la interfaz -> cae a la variable de entorno


def test_sin_nada_el_operador_es_el_usuario_del_sistema_o_desconocido(monkeypatch):
    monkeypatch.setattr(auditoria.getpass, "getuser", lambda: "carlos_pc")
    assert auditoria.operador_por_defecto() == "carlos_pc"

    def sin_usuario():
        raise OSError("sin usuario")

    monkeypatch.setattr(auditoria.getpass, "getuser", sin_usuario)
    assert auditoria.operador_por_defecto() == "desconocido"


def test_la_marca_de_tiempo_es_utc_con_zona():
    ev = auditoria.construir_evento("ducto", {"decision": "approve"}, {}, {})
    assert ev["ts"].endswith("+00:00")


# ---- registrar / leer -----------------------------------------------------------------------------
def test_registrar_crea_carpetas_y_agrega_una_linea_por_evento(tmp_path):
    ruta = tmp_path / "data" / "auditoria" / "aprobaciones.jsonl"
    auditoria.registrar({"n": 1}, ruta)
    auditoria.registrar({"n": 2}, ruta)
    lineas = ruta.read_text(encoding="utf-8").splitlines()
    assert [json.loads(l)["n"] for l in lineas] == [1, 2]


def test_registrar_no_reescribe_lo_anterior(tmp_path):
    ruta = tmp_path / "b.jsonl"
    auditoria.registrar({"n": 1}, ruta)
    antes = ruta.read_text(encoding="utf-8")
    auditoria.registrar({"n": 2}, ruta)
    assert ruta.read_text(encoding="utf-8").startswith(antes)


def test_registrar_conserva_acentos_y_no_mete_saltos_de_linea(tmp_path):
    ruta = tmp_path / "b.jsonl"
    auditoria.registrar({"motivo": "sección\ncon salto"}, ruta)
    texto = ruta.read_text(encoding="utf-8")
    assert len(texto.splitlines()) == 1 and "sección" in texto  # el \n del texto queda escapado dentro del JSON
    assert json.loads(texto)["motivo"] == "sección\ncon salto"


def test_registrar_acepta_valores_raros_sin_reventar(tmp_path):
    ruta = tmp_path / "b.jsonl"
    auditoria.registrar({"ruta": tmp_path}, ruta)  # un Path no es JSON: se guarda como texto
    assert json.loads(ruta.read_text(encoding="utf-8"))["ruta"] == str(tmp_path)


def test_registrar_usa_la_variable_de_entorno_si_existe(tmp_path, monkeypatch):
    alterna = tmp_path / "otra" / "log.jsonl"
    monkeypatch.setenv("GM_AUDITORIA", str(alterna))
    assert auditoria.registrar({"n": 1}) == alterna
    assert alterna.exists()


def test_registrar_lanza_oserror_si_no_puede_escribir(tmp_path):
    archivo = tmp_path / "soy_un_archivo"
    archivo.write_text("x")
    with pytest.raises(OSError):
        auditoria.registrar({"n": 1}, archivo / "log.jsonl")  # la "carpeta" es un archivo


def test_leer_devuelve_eventos_y_cuenta_lineas_danadas(tmp_path):
    ruta = tmp_path / "b.jsonl"
    ruta.write_text('{"n": 1}\n\nesto no es json\n[1, 2]\n{"n": 2}\n', encoding="utf-8")
    eventos, danadas = auditoria.leer(ruta)
    assert [e["n"] for e in eventos] == [1, 2]
    assert danadas == 2  # el texto suelto y la lista (no es un objeto)


def test_leer_sin_archivo_devuelve_vacio(tmp_path):
    assert auditoria.leer(tmp_path / "no_existe.jsonl") == ([], 0)


# ---- id de aprobacion y linea de resultado ---------------------------------------------------------
def test_cada_aprobacion_tiene_un_id_distinto_y_el_tipo_de_evento():
    a = auditoria.construir_evento("ducto", {"decision": "approve"}, {}, {})
    b = auditoria.construir_evento("ducto", {"decision": "approve"}, {}, {})
    assert a["evento"] == "aprobacion" and a["id"] and b["id"] and a["id"] != b["id"]


def test_se_puede_dar_el_id_para_enlazarlo_con_el_resultado():
    ev = auditoria.construir_evento("ducto", {"decision": "approve"}, {}, {}, id_evento="abc123")
    assert ev["id"] == "abc123"


def test_operador_de_usa_la_interfaz_y_si_viene_vacio_el_de_respaldo(monkeypatch):
    monkeypatch.setenv("GM_OPERADOR", "Fer")
    assert auditoria.operador_de({"operador": "Carlos"}) == "Carlos"
    assert auditoria.operador_de({"operador": "  "}) == "Fer"
    assert auditoria.operador_de({}) == "Fer"


def test_construir_resultado_enlaza_con_la_aprobacion():
    r = auditoria.construir_resultado("abc123", "ducto", "Carlos", {"estado": "ok"}, {"modelo_llm": "llama3.1"})
    assert r["evento"] == "resultado" and r["aprobacion_id"] == "abc123" and r["operador"] == "Carlos"
    assert r["resultado"] == {"estado": "ok"} and r["contexto"] == {"modelo_llm": "llama3.1"}
    assert r["ts"].endswith("+00:00")


def test_leer_acepta_lineas_viejas_sin_evento_ni_id(tmp_path):
    ruta = tmp_path / "b.jsonl"
    ruta.write_text('{"ts": "2026-11-05T17:58:43+00:00", "tipo": "parametros", "decision": "approve"}\n', encoding="utf-8")
    eventos, danadas = auditoria.leer(ruta)
    assert danadas == 0 and eventos[0]["tipo"] == "parametros" and "id" not in eventos[0]
