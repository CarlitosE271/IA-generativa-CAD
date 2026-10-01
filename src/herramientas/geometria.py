"""
Herramientas de geometria: STEP -> malla + metadatos. Es la parte de "Ingesta 3D" que ingiere la malla del entorno.

Son funciones puras o sea no dependen de LangChain ni del LLM, igual que documentos.py. 

Carpetas:
  data/geometria/step    <- aqui se copian los STEP originales
  data/geometria/malla   <- aqui se guardan las conversiones a malla (.stl)

Esta herramienta solo ingiere la malla delentorno. 
"""

from __future__ import annotations

from pathlib import Path

raiz = Path(__file__).resolve().parents[2]
ruta_step = raiz / "data" / "geometria" / "step"
ruta_malla = raiz / "data" / "geometria" / "malla"

EXTENSIONES_STEP = (".step", ".stp")  # dos extensiones validas para el mismo formato


def asegurar_carpetas() -> None:
    ruta_step.mkdir(parents=True, exist_ok=True)
    ruta_malla.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Seguridad (misma logica que _nombre_seguro en documentos.py)
# --------------------------------------------------------------------------
def _nombre_seguro(nombre: str, carpeta: Path, extensiones: tuple[str, ...]) -> Path:
    """Convierte un NOMBRE de archivo en una ruta dentro de `carpeta`.

    Los argumentos de una herramienta los elige el LLM: si aceptaramos rutas libres podria
    pedir '..\\..\\.env'. Por eso solo se acepta un nombre simple, sin carpetas, y se
    comprueba que el resultado quede dentro de `carpeta`.
    """
    if not nombre or any(c in nombre for c in "/\\:") or Path(nombre).name != nombre:
        raise ValueError("Solo se acepta el NOMBRE del archivo, sin carpetas ni rutas.")
    if not nombre.lower().endswith(extensiones):
        raise ValueError(f"El archivo debe terminar en {' o '.join(extensiones)}.")
    ruta = (carpeta / nombre).resolve()
    if carpeta.resolve() not in ruta.parents:
        raise ValueError("Ruta fuera de la carpeta permitida.")
    if not ruta.is_file():
        disponibles = sorted(p.name for p in carpeta.iterdir() if p.suffix.lower() in extensiones)
        raise FileNotFoundError(f"No existe '{nombre}'. Disponibles: {disponibles or 'ninguno'}")
    return ruta


# --------------------------------------------------------------------------
# STEP -> malla + metadatos
# --------------------------------------------------------------------------
def _resumen_de_shape(shape) -> dict:
    """Arma el diccionario resumen a partir de un Shape de CadQuery ya cargado.

    Separado de convertir_step_a_md para poder probarlo con un objeto simulado, sin
    necesitar CadQuery real instalado (ver tests/test_geometria.py).
    """
    bb = shape.BoundingBox()
    piezas = shape.Solids()  # cuenta cuantos solidos independientes trae el STEP
    return {
        # CadQuery/OCCT normaliza el STEP importado a milimetros al leerlo
        "unidades": "mm",
        "bounding_box_mm": {
            "x": [round(bb.xmin, 2), round(bb.xmax, 2)],
            "y": [round(bb.ymin, 2), round(bb.ymax, 2)],
            "z": [round(bb.zmin, 2), round(bb.zmax, 2)],
        },
        "dimensiones_mm": {
            "x": round(bb.xlen, 2),
            "y": round(bb.ylen, 2),
            "z": round(bb.zlen, 2),
        },
        "num_piezas": len(piezas),
    }


def convertir_step_a_malla(nombre_step: str) -> dict:
    """Convierte UN STEP de data/geometria/step a una malla .stl en data/geometria/malla.
    Devuelve un resumen (bounding box, unidades, piezas, triangulos), no la geometria cruda."""
    import cadquery as cq  # import perezoso: es pesada y solo se usa aqui

    asegurar_carpetas()
    origen = _nombre_seguro(nombre_step, ruta_step, EXTENSIONES_STEP)
    destino = ruta_malla / (origen.stem + ".stl")

    resultado = cq.importers.importStep(str(origen))
    shape = resultado.val()
    # .val() puede regresar Vector/Location/Sketch/Shape segun que haya en la pila de CadQuery
    # (por eso el editor marca "tessellate" como desconocido: esos otros tipos no lo tienen).
    # Tras importStep() siempre debe ser un Shape; si no, es mejor un error claro aqui que uno
    # críptico mas adelante dentro de CadQuery.
    if not hasattr(shape, "tessellate"):
        raise ValueError(
            f"El STEP no contiene un solido/superficie valido (se obtuvo {type(shape).__name__})."
        )
    info = _resumen_de_shape(shape)

    # Tesela la geometria (superficies curvas -> triangulos) para poder exportarla como malla;
    # 0.1 mm de tolerancia lineal es un punto de partida razonable, se puede ajustar despues.
    vertices, triangulos = shape.tessellate(0.1)
    info["num_triangulos"] = len(triangulos)

    cq.exporters.export(resultado, str(destino))

    info["step"] = origen.name
    info["malla"] = destino.name
    return info


def convertir_pendientes() -> list[dict]:
    """Convierte todos los STEP que no tengan .stl o cuyo .stl sea mas viejo que el STEP."""
    asegurar_carpetas()
    resultados = []
    for step in sorted(p for p in ruta_step.iterdir() if p.suffix.lower() in EXTENSIONES_STEP):
        malla = ruta_malla / (step.stem + ".stl")
        if malla.exists() and malla.stat().st_mtime >= step.stat().st_mtime:
            resultados.append({"step": step.name, "malla": malla.name, "estado": "ya estaba convertido"})
            continue
        try:
            info = convertir_step_a_malla(step.name)
            info["estado"] = "convertido"
        except Exception as e:  # un STEP dañado o con geometria invalida no debe detener a los demas
            info = {"step": step.name, "estado": f"error: {e}"}
        resultados.append(info)
    return resultados
