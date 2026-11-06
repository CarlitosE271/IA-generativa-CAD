"""
Pruebas de src/herramientas/documentos.py. NO usan el LLM ni Ollama: por eso se separaron
las herramientas en un modulo aparte (funciones puras).

Ejecutar desde la raiz del proyecto:  pytest tests -v
(requiere `pip install pytest`; la prueba de conversion tambien usa reportlab y se omite si falta)
"""
import pytest

from herramientas import documentos as d

NORMA = """Norma de diseno de ductos HVAC

1. Alcance
Esta norma aplica a ductos de aire acondicionado automotrices de plastico inyectado.

2. Radio de curvatura
El radio de curvatura minimo debe ser igual o mayor a 1.5 veces el diametro hidraulico de la seccion.
Para una seccion con diametro hidraulico de 40 mm, el radio minimo es de 60 mm.

3. Relacion de aspecto
La relacion de aspecto maxima de la seccion transversal no debe exceder 3:1 en tramos rectos.

4. Angulo de desmoldeo
El angulo de desmoldeo minimo es de 1.5 grados por lado para superficies lisas.
"""


def _preparar(tmp_path, monkeypatch):
    """Redirige las carpetas del modulo a una carpeta temporal para no tocar data/ real."""
    pdf, md = tmp_path / "pdf", tmp_path / "md"
    pdf.mkdir()
    md.mkdir()
    monkeypatch.setattr(d, "ruta_pdf", pdf)
    monkeypatch.setattr(d, "ruta_md", md)
    return pdf, md


# ---- seguridad: el LLM solo puede pedir NOMBRES de archivo ----------------
def test_rechaza_rutas_y_escapes(tmp_path, monkeypatch):
    pdf, _ = _preparar(tmp_path, monkeypatch)
    (pdf / "norma.pdf").write_bytes(b"%PDF-1.4")
    for malo in ["..\\..\\.env", "../secreto.pdf", "C:/Windows/x.pdf", "sub/norma.pdf", "", "/etc/passwd"]:
        with pytest.raises(ValueError):
            d._nombre_seguro(malo, pdf, ".pdf")
    with pytest.raises(ValueError):
        d._nombre_seguro("norma.txt", pdf, ".pdf")  # extension incorrecta
    with pytest.raises(FileNotFoundError):
        d._nombre_seguro("no_existe.pdf", pdf, ".pdf")
    assert d._nombre_seguro("norma.pdf", pdf, ".pdf").name == "norma.pdf"


# ---- division en fragmentos ------------------------------------------------
def test_divide_por_secciones_numeradas():
    frag = d.dividir_en_fragmentos(NORMA, "norma.md")
    secciones = [f.seccion for f in frag]
    assert "2. Radio de curvatura" in secciones
    assert "4. Angulo de desmoldeo" in secciones
    assert all(f.archivo == "norma.md" for f in frag)


def test_ningun_fragmento_excede_el_maximo():
    largo = "\n\n".join(f"Parrafo {i}. " + "texto de relleno con varias palabras. " * 12 for i in range(60))
    gigante = "Una sola oracion sin puntos " * 200                      # obliga al corte duro
    for cuerpo in (f"1. Seccion larga\n{largo}", f"1. Seccion gigante\n{gigante}"):
        frag = d.dividir_en_fragmentos(cuerpo, "x.md")
        assert len(frag) > 1
        assert all(len(f.texto) <= d.max_caracteres for f in frag), max(len(f.texto) for f in frag)


def test_encabezados_heuristica():
    assert d._es_encabezado("2. Radio de curvatura")
    assert d._es_encabezado("3.1 Alcance general")
    assert d._es_encabezado("# Titulo markdown")
    assert d._es_encabezado("ALCANCE")
    assert not d._es_encabezado("1. Verificar el radio antes de continuar.")  # item de lista, termina en punto
    assert not d._es_encabezado("mm")
    assert not d._es_encabezado("60")


# ---- busqueda --------------------------------------------------------------
def test_buscar_devuelve_la_seccion_correcta():
    frag = d.dividir_en_fragmentos(NORMA, "norma.md")
    top = d.buscar("radio minimo de curvatura", top_k=1, fragmentos=frag)[0]
    assert top["seccion"] == "2. Radio de curvatura" and "60 mm" in top["texto"]
    top = d.buscar("relacion de aspecto maxima", top_k=1, fragmentos=frag)[0]
    assert top["seccion"] == "3. Relacion de aspecto"
    top = d.buscar("angulo desmoldeo", top_k=1, fragmentos=frag)[0]
    assert top["seccion"] == "4. Angulo de desmoldeo"


def test_buscar_numeros_decimales_y_plurales():
    frag = d.dividir_en_fragmentos(NORMA, "norma.md")
    assert d.buscar("1.5 veces", top_k=1, fragmentos=frag)[0]["seccion"] == "2. Radio de curvatura"
    assert d.buscar("radios", top_k=1, fragmentos=frag)[0]["seccion"] == "2. Radio de curvatura"  # plural


def test_buscar_sin_coincidencias_devuelve_vacio():
    frag = d.dividir_en_fragmentos(NORMA, "norma.md")
    assert d.buscar("zzz qqq", fragmentos=frag) == []
    assert d.buscar("de la el", fragmentos=frag) == []   # solo palabras vacias
    assert d.buscar("radio", fragmentos=[]) == []


def test_buscar_lee_los_md_de_la_carpeta(tmp_path, monkeypatch):
    _, md = _preparar(tmp_path, monkeypatch)
    (md / "norma.md").write_text(NORMA, encoding="utf-8")
    res = d.buscar("desmoldeo")
    assert res and res[0]["archivo"] == "norma.md"


def _requiere_pymupdf():
    """PyMuPDF se expone bajo dos nombres de paquete segun la version instalada."""
    try:
        import pymupdf  # noqa: F401
    except ImportError:
        pytest.importorskip("fitz")


# ---- orden de lectura: no necesita PyMuPDF real, solo el formato de tupla que documenta --------
class _Rect:
    def __init__(self, width):
        self.width = width


class _PaginaFalsa:
    """Simula el 'page' de PyMuPDF: (x0, y0, x1, y1, texto, num_bloque, tipo); tipo 0 = texto."""

    def __init__(self, ancho, bloques):
        self.rect = _Rect(ancho)
        self._bloques = bloques

    def get_text(self, modo):
        assert modo == "blocks"
        return self._bloques


def test_orden_de_lectura_reordena_dos_columnas_revueltas():
    # El PDF entrega los bloques en el orden interno del archivo, no en orden de lectura:
    # asi se ve el bug original (frases de la columna derecha intercaladas con la izquierda).
    titulo = (72, 20, 540, 50, "TITULO DE LA NORMA", 0, 0)
    izq1 = (72, 100, 290, 150, "Columna izquierda parrafo 1", 1, 0)
    izq2 = (72, 200, 290, 250, "Columna izquierda parrafo 2", 2, 0)
    der1 = (320, 100, 540, 150, "Columna derecha parrafo 1", 3, 0)
    der2 = (320, 200, 540, 250, "Columna derecha parrafo 2", 4, 0)
    pie = (72, 700, 540, 720, "Pie de pagina a todo lo ancho", 5, 0)
    pagina = _PaginaFalsa(612, [titulo, der1, izq1, izq2, pie, der2])
    assert d._bloques_en_orden_de_lectura(pagina) == [
        "TITULO DE LA NORMA",
        "Columna izquierda parrafo 1",
        "Columna izquierda parrafo 2",
        "Columna derecha parrafo 1",
        "Columna derecha parrafo 2",
        "Pie de pagina a todo lo ancho",
    ]


def test_orden_de_lectura_una_columna_no_activa_modo_columnas():
    # Sin bloques del lado derecho no hay dos columnas: se ordena solo por altura (y).
    b1 = (72, 100, 150, 120, "Titulo corto", 0, 0)
    b2 = (72, 140, 500, 160, "Parrafo largo que cruza el centro de la pagina", 1, 0)
    b3 = (72, 180, 140, 200, "Otra linea corta", 2, 0)
    pagina = _PaginaFalsa(612, [b3, b1, b2])
    assert d._bloques_en_orden_de_lectura(pagina) == [
        "Titulo corto",
        "Parrafo largo que cruza el centro de la pagina",
        "Otra linea corta",
    ]


def test_orden_de_lectura_descarta_imagenes_y_bloques_vacios():
    imagen = (72, 100, 300, 300, "", 0, 1)  # tipo 1 = imagen, no texto
    vacio = (72, 310, 300, 320, "   ", 1, 0)
    texto = (72, 330, 300, 350, "Unico texto real", 2, 0)
    pagina = _PaginaFalsa(612, [imagen, vacio, texto])
    assert d._bloques_en_orden_de_lectura(pagina) == ["Unico texto real"]
    assert d._bloques_en_orden_de_lectura(_PaginaFalsa(612, [])) == []


# ---- conversion real PDF -> MD (necesita PyMuPDF y reportlab instalados) -----------------------
def _hacer_pdf(ruta, lineas):
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    c = canvas.Canvas(str(ruta))
    y = 800
    for linea in lineas:
        c.drawString(72, y, linea)
        y -= 18
    c.save()


def test_convertir_pdf_y_omitir_si_ya_esta_al_dia(tmp_path, monkeypatch):
    _requiere_pymupdf()
    pdf, md = _preparar(tmp_path, monkeypatch)
    _hacer_pdf(pdf / "norma.pdf", ["1. Alcance"] + ["Texto de la norma numero %d, con contenido suficiente." % i for i in range(12)]
               + ["2. Radio de curvatura", "El radio minimo es de 60 mm."])
    r = d.convertir_pendientes()
    assert r[0]["estado"] == "convertido" and (md / "norma.md").exists()
    assert "Radio de curvatura" in (md / "norma.md").read_text(encoding="utf-8")
    assert d.convertir_pendientes()[0]["estado"] == "ya estaba convertido"   # no repite el trabajo
    assert d.buscar("radio minimo", top_k=1)[0]["seccion"] == "2. Radio de curvatura"


def test_pdf_casi_vacio_genera_advertencia(tmp_path, monkeypatch):
    _requiere_pymupdf()
    pdf, _ = _preparar(tmp_path, monkeypatch)
    _hacer_pdf(pdf / "escaneo.pdf", ["hola"])
    assert "advertencia" in d.convertir_pdf_a_md("escaneo.pdf")


def test_pdf_danado_no_detiene_a_los_demas(tmp_path, monkeypatch):
    _requiere_pymupdf()
    pdf, _ = _preparar(tmp_path, monkeypatch)
    (pdf / "roto.pdf").write_bytes(b"esto no es un pdf")
    _hacer_pdf(pdf / "bueno.pdf", ["1. Alcance"] + ["Texto suficiente para pasar el umbral %d de caracteres." % i for i in range(10)])
    resultados = {r["pdf"]: r for r in d.convertir_pendientes()}
    assert resultados["bueno.pdf"]["estado"] == "convertido"
    # A diferencia de markitdown, PyMuPDF si lanza una excepcion clara al abrir un PDF invalido.
    assert resultados["roto.pdf"]["estado"].startswith("error"), resultados["roto.pdf"]
