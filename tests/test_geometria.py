"""
Pruebas de src/herramientas/geometria.py. NO usan CadQuery real: este entorno de trabajo no
tuvo acceso a PyPI para instalarlo, asi que se simula con objetos falsos que respetan la API
documentada de CadQuery 2.x (BoundingBox, Solids, tessellate, importers.importStep,
exporters.export). _resumen_de_shape esta separado de convertir_step_a_malla justo para que
la parte de calculo se pueda probar sin CadQuery.

Ejecutar desde la raiz del proyecto:  pytest tests -v
"""
import sys
import types
from pathlib import Path

import pytest

from herramientas import geometria as g


def _preparar(tmp_path, monkeypatch):
    """Redirige las carpetas del modulo a una carpeta temporal para no tocar data/ real."""
    step, malla = tmp_path / "step", tmp_path / "malla"
    step.mkdir()
    malla.mkdir()
    monkeypatch.setattr(g, "ruta_step", step)
    monkeypatch.setattr(g, "ruta_malla", malla)
    return step, malla


# ---- seguridad: el LLM solo puede pedir NOMBRES de archivo (misma logica que documentos.py) ----
def test_rechaza_rutas_y_escapes(tmp_path, monkeypatch):
    step, _ = _preparar(tmp_path, monkeypatch)
    (step / "pieza.step").write_bytes(b"ISO-10303-21;")
    for malo in ["..\\..\\.env", "../secreto.step", "C:/Windows/x.step", "sub/pieza.step", "", "/etc/passwd"]:
        with pytest.raises(ValueError):
            g._nombre_seguro(malo, step, g.EXTENSIONES_STEP)
    with pytest.raises(ValueError):
        g._nombre_seguro("pieza.txt", step, g.EXTENSIONES_STEP)  # extension incorrecta
    with pytest.raises(FileNotFoundError):
        g._nombre_seguro("no_existe.step", step, g.EXTENSIONES_STEP)
    assert g._nombre_seguro("pieza.step", step, g.EXTENSIONES_STEP).name == "pieza.step"


def test_acepta_extension_stp(tmp_path, monkeypatch):
    step, _ = _preparar(tmp_path, monkeypatch)
    (step / "pieza.stp").write_bytes(b"ISO-10303-21;")
    assert g._nombre_seguro("pieza.stp", step, g.EXTENSIONES_STEP).name == "pieza.stp"


# ---- calculo del resumen: no necesita CadQuery real, solo la API documentada -------------------
class _BBFalso:
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax):
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.zmin, self.zmax = zmin, zmax
        self.xlen, self.ylen, self.zlen = xmax - xmin, ymax - ymin, zmax - zmin


class _ShapeFalso:
    """Simula el Shape de CadQuery: solo los metodos que _resumen_de_shape usa."""

    def __init__(self, bb, n_solidos=1, triangulos=None):
        self._bb = bb
        self._n_solidos = n_solidos
        self._triangulos = triangulos if triangulos is not None else [(0, 1, 2)]

    def BoundingBox(self):
        return self._bb

    def Solids(self):
        return list(range(self._n_solidos))  # solo importa la cantidad

    def tessellate(self, tolerancia):
        return ([], self._triangulos)


def test_resumen_de_shape_calcula_bounding_box_dimensiones_y_piezas():
    bb = _BBFalso(xmin=-10, xmax=50, ymin=0, ymax=25.5, zmin=-5, zmax=5)
    shape = _ShapeFalso(bb, n_solidos=3)
    resumen = g._resumen_de_shape(shape)
    assert resumen["unidades"] == "mm"
    assert resumen["bounding_box_mm"] == {"x": [-10, 50], "y": [0, 25.5], "z": [-5, 5]}
    assert resumen["dimensiones_mm"] == {"x": 60, "y": 25.5, "z": 10}
    assert resumen["num_piezas"] == 3


# ---- conversion completa: CadQuery se simula con un modulo falso en sys.modules ----------------
class _ResultadoFalso:
    def __init__(self, shape):
        self._shape = shape

    def val(self):
        return self._shape


def _instalar_cadquery_falso(monkeypatch, shape, escribir_stl=True, falla_export=False):
    """Registra un modulo 'cadquery' falso en sys.modules, con solo lo que geometria.py usa."""
    resultado = _ResultadoFalso(shape)
    rutas_importadas = []
    rutas_exportadas = []

    def importStep(ruta):
        rutas_importadas.append(ruta)
        return resultado

    def export(resultado_o_shape, ruta):
        rutas_exportadas.append(ruta)
        if falla_export:
            raise RuntimeError("fallo simulado al exportar")
        if escribir_stl:
            Path(ruta).write_text("STL FALSO (contenido simulado)")

    modulo = types.ModuleType("cadquery")
    modulo.importers = types.SimpleNamespace(importStep=importStep)
    modulo.exporters = types.SimpleNamespace(export=export)
    monkeypatch.setitem(sys.modules, "cadquery", modulo)
    return rutas_importadas, rutas_exportadas


def test_convertir_step_a_malla_devuelve_resumen_y_escribe_la_malla(tmp_path, monkeypatch):
    step, malla = _preparar(tmp_path, monkeypatch)
    (step / "motor.step").write_bytes(b"ISO-10303-21;")
    bb = _BBFalso(0, 100, 0, 50, 0, 30)
    shape = _ShapeFalso(bb, n_solidos=2, triangulos=[(0, 1, 2), (1, 2, 3)])
    _instalar_cadquery_falso(monkeypatch, shape)

    info = g.convertir_step_a_malla("motor.step")

    assert info["step"] == "motor.step"
    assert info["malla"] == "motor.stl"
    assert info["num_piezas"] == 2
    assert info["num_triangulos"] == 2
    assert info["unidades"] == "mm"
    assert (malla / "motor.stl").exists()  # el archivo se escribe de verdad


def test_convertir_pendientes_omite_si_ya_esta_al_dia(tmp_path, monkeypatch):
    step, malla = _preparar(tmp_path, monkeypatch)
    (step / "motor.step").write_bytes(b"ISO-10303-21;")
    bb = _BBFalso(0, 10, 0, 10, 0, 10)
    _instalar_cadquery_falso(monkeypatch, _ShapeFalso(bb))

    r1 = g.convertir_pendientes()
    assert r1[0]["estado"] == "convertido"
    r2 = g.convertir_pendientes()
    assert r2[0]["estado"] == "ya estaba convertido"  # no se vuelve a convertir


def test_convertir_pendientes_no_se_detiene_si_un_step_falla(tmp_path, monkeypatch):
    step, _ = _preparar(tmp_path, monkeypatch)
    (step / "roto.step").write_bytes(b"esto no es un step valido")
    (step / "bueno.step").write_bytes(b"ISO-10303-21;")
    bb = _BBFalso(0, 10, 0, 10, 0, 10)

    # Un cadquery falso cuyo importStep revienta SOLO para "roto.step"
    def importStep(ruta):
        if "roto" in str(ruta):
            raise RuntimeError("STEP invalido: fallo simulado")
        return _ResultadoFalso(_ShapeFalso(bb))

    modulo = types.ModuleType("cadquery")
    modulo.importers = types.SimpleNamespace(importStep=importStep)
    modulo.exporters = types.SimpleNamespace(export=lambda r, ruta: Path(ruta).write_text("x"))
    monkeypatch.setitem(sys.modules, "cadquery", modulo)

    resultados = {r["step"]: r for r in g.convertir_pendientes()}
    assert resultados["bueno.step"]["estado"] == "convertido"
    assert resultados["roto.step"]["estado"].startswith("error"), resultados["roto.step"]
