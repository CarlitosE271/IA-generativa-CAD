"""
Pruebas de src/cad/informe_ducto.py: texto que lee el LLM, marcas de archivos y propuesta plana del ducto.
Logica pura, sin LangChain ni CadQuery.  Ejecutar:  pytest tests -v
"""
import types

import pytest

from cad import informe_ducto as ic


def _bloque(nombre, tipo="neutro", valor=0.0):
    return {"id": nombre, "nombre": nombre, "tipo": tipo, "valor": valor,
            "centro": {"x": 100.0, "y": 0.0, "z": 50.5}, "size": {"x": 10.0, "y": 20.0, "z": 30.0}}


def _msg(nombre, contenido):
    return types.SimpleNamespace(name=nombre, content=contenido)


# ---- propuesta plana ----------------------------------------------------------------------------
def test_propuesta_y_argumentos_son_inversas():
    p = ic.propuesta_de_ducto([0, 0, 500], [2300, 0, 500], 40, 25, 2)
    assert list(p) == list(ic.CAMPOS_PROPUESTA)
    assert all(isinstance(v, float) for v in p.values())
    assert ic.argumentos_de_propuesta(p) == ([0.0, 0.0, 500.0], [2300.0, 0.0, 500.0], 40.0, 25.0, 2.0)


@pytest.mark.parametrize("malo", [[1, 2], [1, 2, 3, 4], "abc", None, [1, "x", 3]])
def test_punto_3d_rechaza_lo_que_no_son_tres_numeros(malo):
    with pytest.raises(ValueError):
        ic.punto_3d(malo, "inicio")


# ---- texto del entorno y la ruta ----------------------------------------------------------------
def test_texto_entorno_vacio_y_con_bloques():
    assert "vacio" in ic.texto_entorno([])
    t = ic.texto_entorno([_bloque("Motor", "termico", 100.0), _bloque("Chasis")])
    assert "Motor (termico, valor 100)" in t and "centro (100, 0, 50.5)" in t
    assert "Chasis (neutro)" in t and "valor" not in t.split("Chasis")[1]


def test_texto_ruta_ok_incluye_marca_de_imagen_y_error_no():
    ok = {"estado": "ok", "ruta": [[0, 0, 0]] * 4, "longitud_mm": 2626.619, "bloques_cercanos": ["Tablero"], "mensaje": ""}
    t = ic.texto_ruta(ok, "C:/x/ruta.png")
    assert "4 puntos" in t and "2626.62" in t and "Tablero" in t and "[IMAGEN_RUTA: C:/x/ruta.png]" in t
    assert "IMAGEN_RUTA" not in ic.texto_ruta(ok, None)
    malo = {"estado": "sin_ruta", "ruta": [], "longitud_mm": 0, "bloques_cercanos": [], "mensaje": "no hay camino"}
    assert "no hay camino" in ic.texto_ruta(malo, "x.png") and "IMAGEN_RUTA" not in ic.texto_ruta(malo, "x.png")


# ---- texto del reporte --------------------------------------------------------------------------
def _reporte_ok(**extra):
    r = {
        "estado": "ok", "nombre": "d1", "longitud_mm": 2482.5, "espesor_mm": 2.0, "volumen_mm3": 605502.1, "solido_valido": True,
        "seccion": {"area_mm2": 1000.0, "relacion_aspecto": 1.6, "diametro_hidraulico_mm": 30.77, "area_ok": True, "aspecto_ok": True},
        "radio_curvatura": {"cumple": True, "radio_minimo_mm": 47.25, "r_min_exigido_mm": 45.0},
        "curva": {"factible": True, "holguras_mm": {"Tablero": 44.87}, "colisiones": [], "dentro_de_zona_de_seguridad": []},
        "archivos": {"step": "d1.step", "stl": "d1.stl"},
    }
    r.update(extra)
    return r


def test_texto_reporte_ok_trae_los_numeros_y_las_marcas_cad():
    t = ic.texto_reporte(_reporte_ok(), ["C:/cad/d1.step", "C:/cad/d1.stl"])
    for esperado in ("2482.5", "47.25", "45", "Tablero: 44.87", "605502.1", "solido valido: si", "[ARCHIVO_CAD: C:/cad/d1.step]"):
        assert esperado in t


def test_texto_reporte_ok_avisa_colision_y_ducto_que_no_cumple():
    rep = _reporte_ok()
    rep["curva"].update(factible=False, colisiones=["Motor"])
    t = ic.texto_reporte(rep)
    assert "COLISION con: Motor" in t and "NO cumple R_min" in t


def test_texto_reporte_sin_r_min_aprobado_no_dice_que_cumple():
    rep = _reporte_ok()
    rep["radio_curvatura"] = {"cumple": "no_verificado", "radio_minimo_mm": 12.0}
    rep["curva"] = {"aviso": "No hay r_min aprobado"}
    t = ic.texto_reporte(rep)
    assert "no_verificado" in t and "AVISO" in t


def test_texto_reporte_infactible_da_propuestas_e_instrucciones_sin_marcas():
    rep = {"estado": "infactible", "mensaje": "No se puede cumplir R_min = 600 mm", "radio_maximo_factible_mm": 178,
           "propuestas": [{"id": "bajar_r_min", "descripcion": "Bajar a 178", "requiere_aprobacion": True},
                          {"id": "mover_entorno", "descripcion": "Mover bloques", "requiere_aprobacion": False}]}
    t = ic.texto_reporte(rep)
    assert "INFACTIBLE" in t and "178" in t and "bajar_r_min" in t and "(requiere aprobacion" in t
    assert "NO cambies r_min por tu cuenta" in t and "ARCHIVO_CAD" not in t


@pytest.mark.parametrize("estado", ["sin_ruta", "error"])
def test_texto_reporte_de_errores_del_astar(estado):
    assert "no hay camino" in ic.texto_reporte({"estado": estado, "mensaje": "no hay camino"})


# ---- marcas -> archivos -------------------------------------------------------------------------
def test_archivos_de_mensajes_solo_de_las_herramientas_correctas_y_sin_repetir():
    mensajes = [
        _msg("calcular_ruta", "Ruta...\n[IMAGEN_RUTA: C:\\p\\data\\salidas\\ruta_1.png]"),
        _msg("generar_ducto", "ok\n[ARCHIVO_CAD: C:\\p\\data\\cad\\d.step]\n[ARCHIVO_CAD: C:\\p\\data\\cad\\d.stl]"),
        _msg("generar_ducto", "[ARCHIVO_CAD: C:\\p\\data\\cad\\d.step]"),            # repetido
        _msg("buscar_en_documentos", "[ARCHIVO_CAD: C:\\otra\\cosa.step]"),            # otra herramienta: se ignora
        types.SimpleNamespace(content="[IMAGEN_RUTA: x.png]"),                        # sin name: se ignora
    ]
    assert ic.archivos_de_mensajes(mensajes) == [
        {"tipo": "imagen", "ruta": "C:\\p\\data\\salidas\\ruta_1.png"},
        {"tipo": "cad", "ruta": "C:\\p\\data\\cad\\d.step"},
        {"tipo": "cad", "ruta": "C:\\p\\data\\cad\\d.stl"},
    ]


def test_quitar_marcas_borra_las_marcas_y_deja_el_resto():
    assert ic.quitar_marcas("Listo [IMAGEN_RUTA: a.png] gracias [ARCHIVO_CAD: b.step]").split() == ["Listo", "gracias"]
    assert ic.quitar_marcas("valor [1] intacto") == "valor [1] intacto"


# ---- transparencia / respuesta inutil -------------------------------------------------------------
@pytest.mark.parametrize("texto", ["", "   ", "Lo siento, pero no puedo continuar con la respuesta.",
                                   "Disculpa, no puedo ayudar con eso."])
def test_respuesta_inutil_detecta_vacias_y_disculpas(texto):
    assert ic.respuesta_inutil(texto) is True


@pytest.mark.parametrize("texto", ["Hay 4 bloques en el entorno: Motor, Bateria, Chasis y Tablero.",
                                   "Lo siento, el motor no cabe: " + "x" * 300])
def test_respuesta_inutil_no_marca_respuestas_reales(texto):
    assert ic.respuesta_inutil(texto) is False


def test_traza_herramientas_lista_llamadas_y_resultados_sin_marcas():
    mensajes = [
        types.SimpleNamespace(type="ai", tool_calls=[{"name": "calcular_ruta", "args": {"inicio": [0, 0, 0]}}], content=""),
        types.SimpleNamespace(type="tool", name="calcular_ruta", content="Ruta ok [IMAGEN_RUTA: a.png]"),
        types.SimpleNamespace(type="ai", tool_calls=[], content="Listo"),
    ]
    assert ic.traza_herramientas(mensajes) == [
        {"tipo": "llamada", "herramienta": "calcular_ruta", "argumentos": {"inicio": [0, 0, 0]}},
        {"tipo": "resultado", "herramienta": "calcular_ruta", "texto": "Ruta ok"},
    ]
    assert ic.ultimo_resultado(mensajes) == "Ruta ok"
    assert ic.ultimo_resultado([types.SimpleNamespace(type="ai", content="hola")]) is None


def test_traza_corta_los_resultados_muy_largos():
    m = types.SimpleNamespace(type="tool", name="x", content="a" * 5000)
    assert len(ic.traza_herramientas([m], max_caracteres=100)[0]["texto"]) < 120
