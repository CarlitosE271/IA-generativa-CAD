"""
Al ser funciones puras, no dependen de LangChain ni del agente. Asi se pueden probar con
pytest y reutilizar desde la interfaz o las fases siguientes. El agente las usa a
traves de @tool en agente.py.

Carpetas:
  data/docs/pdf:    aqui se copian los PDF originales (No se suben a Git, son confidenciales)
  data/docs/md:     aqui se guardan las conversiones a Markdown

El LLM tiene una ventana de contexto pequeña (num_ctx=8192 tokens; en la prueba real dieron unas
9 paginas de texto denso, no 16: un PDF de 9 paginas ya llego a ~7000 tokens). Con este motivo,
al agente no se le pasa el documento entero, solo se le entregan 3 fragmentos que son relevantes
para cada pregunta.

La extraccion de texto usa PyMuPDF, no MarkItDown: varios PDF reales (normas tipo ASHRAE) estan
a dos columnas, y una extraccion linea por linea mezcla frases de la columna izquierda con la
derecha cuando quedan a la misma altura de la pagina. PyMuPDF entrega el texto en bloques
(parrafos) con su posicion en la pagina, lo que permite leer toda la columna izquierda y luego
toda la derecha en vez de intercalarlas. Ver _bloques_en_orden_de_lectura para el detalle.

Flujo de trabajo:
-> input PDF
-> Markdown
-> Fragmentos
-> Busqueda

"""
# Posponer la evaluación de type annotatios, guardandolas como strings.
# Permite evaluar clases antes de que esten definidos, acelera la carga del scipt.
# Permite referencias hacia delante.
from __future__ import annotations

import math
import re # importa el modulo de expresiones regulares. Sirve para extraer, buscar o manipulas patrones en textos.
import unicodedata # Normaliza el texto, como quitar acentos. 
from collections import Counter # Sirve para contar frecuencias. Para contar cuantas veces aparece una palabra. 
from dataclasses import dataclass # Permite almacenar datos. 
from pathlib import Path # una forma de trabajar con carpetas, archivos en python.

# parents[2] sube de herramientas -> src -> raiz del proyecto.
raiz = Path(__file__).resolve().parents[2]
ruta_pdf = raiz / "data" / "docs" / "pdf"
ruta_md = raiz / "data" / "docs" / "md"

max_caracteres = 1200  # tamaño maximo de un fragmento (unos 300-350 tokens).
holgura = 150           # se repite la ultima oracion del fragmento anterior si cabe en este margen.

# Creo que el nombre es bastante obvio 
def asegurar_carpetas() -> None:
    ruta_pdf.mkdir(parents=True, exist_ok=True)  # exist_ok=True evita errores en caso de que ya existan las carpetas.
    ruta_md.mkdir(parents=True, exist_ok=True) # Crea los directorios intermedios.


# ----
#  PDF a Markdown
# ----


def _nombre_seguro(nombre: str, carpeta: Path, extension: str) -> Path:
    """Convierte un nombre de archivo en una ruta dentro de la \"carpeta\".

    Debido a que los argumentos de una herramienta los elige el LLM o en el pero de los casos, un texto malicioso dentro
    de un PDF, si aceptaramos rutas libres podria pedir '..\\..\\.env'. Por eso solo se acepta
    un nombre simple, sin carpetas, y se comprueba que el resultado quede dentro de \"carpeta\".
    """
    # Se debe rechazar cualquier texto con \ para que no se ejecuten rutas libres que podrían afectar la integridad del codigo o computadora.
    if not nombre or any(c in nombre for c in "/\\:") or Path(nombre).name != nombre: 
        raise ValueError("Solo se acepta el nombre del archivo, sin carpetas ni rutas.")
    if not nombre.lower().endswith(extension): # la exntensión se refiere a una cadena, para este caso es .pdf
        raise ValueError(f"El archivo debe terminar en {extension}.") # De no terminar en el formato deseado lanza el error.
    ruta = (carpeta / nombre).resolve() # Junta todo a una ruta final
    if carpeta.resolve() not in ruta.parents: # Condicional para sabr si el documetno se encuentra en la ruta.
        raise ValueError("Ruta fuera de la carpeta permitida.") # Si el documento no se encuentra en la ruta colocada, se genera este error.
    if not ruta.is_file(): # comprueba si la ruta existente es un archivo existente.
        disponibles = sorted(p.name for p in carpeta.glob(f"*{extension}")) # Compila los documentos disponibles
        raise FileNotFoundError(f"No existe '{nombre}'. Disponibles: {disponibles or 'ninguno'}") # Muestra todos los documentos disponibles en la carpeta.
    return ruta # Regresa la ruta resuelta


def _bloques_en_orden_de_lectura(pagina) -> list[str]:
    """Ordena los bloques de texto (parrafos) de UNA pagina de PyMuPDF en orden de lectura.

    pagina.get_text("blocks") regresa tuplas (x0, y0, x1, y1, texto, num_bloque, tipo); tipo 0
    es texto (se descartan las imagenes, tipo 1). El problema es que NO vienen en orden de
    lectura, vienen en el orden interno del PDF: en un documento a dos columnas eso puede
    saltar de una columna a la otra en cualquier momento (justo el bug que veiamos con
    MarkItDown).

    Heuristica: si casi todos los bloques quedan claramente a un lado u otro del centro de la
    pagina sin cruzarlo, es un documento a dos columnas: se lee toda la columna izquierda de
    arriba a abajo y luego toda la derecha. Si no, es un documento normal a una columna: se
    ordena solo por posicion vertical. Los bloques que si cruzan el centro (titulos, tablas a
    todo lo ancho) se ubican antes o despues de las columnas segun si aparecen arriba o abajo
    de donde estas empiezan.

    Limite conocido: un bloque ancho que cae A MITAD de las columnas (por ejemplo una tabla a
    mitad de pagina) se coloca despues de ambas, no en su lugar exacto.
    """
    ancho_pagina = pagina.rect.width
    medio = ancho_pagina / 2
    bloques = [b for b in pagina.get_text("blocks") if b[6] == 0 and b[4].strip()]
    if not bloques:
        return []

    izquierda, derecha, cruza_el_centro = [], [], []
    for b in bloques:
        x0, x1 = b[0], b[2]
        if x1 <= medio * 1.08:  # no llega ni cerca del centro: columna izquierda
            izquierda.append(b)
        elif x0 >= medio * 0.92:  # empieza ya del otro lado del centro: columna derecha
            derecha.append(b)
        else:
            cruza_el_centro.append(b)  # ocupa ambos lados: titulo, tabla ancha, etc.

    es_dos_columnas = len(izquierda) >= 2 and len(derecha) >= 2

    if es_dos_columnas:
        izquierda.sort(key=lambda b: b[1])  # y0: de arriba a abajo
        derecha.sort(key=lambda b: b[1])
        cruza_el_centro.sort(key=lambda b: b[1])
        y_inicio = min(izquierda[0][1], derecha[0][1])
        antes = [b for b in cruza_el_centro if b[1] < y_inicio]   # ej. titulo de la pagina
        despues = [b for b in cruza_el_centro if b[1] >= y_inicio]  # ej. pie de pagina
        ordenados = antes + izquierda + derecha + despues
    else:
        ordenados = sorted(bloques, key=lambda b: (b[1], b[0]))  # solo de arriba a abajo

    return [b[4].strip() for b in ordenados]


def _extraer_texto_pymupdf(ruta: Path) -> str:
    """Extrae el texto de un PDF completo, pagina por pagina, respetando columnas."""
    try:
        import pymupdf  # nombre nuevo del paquete
    except ImportError:
        import fitz as pymupdf  # nombre historico del mismo paquete (PyMuPDF)

    with pymupdf.open(ruta) as doc:
        paginas = ["\n\n".join(_bloques_en_orden_de_lectura(pagina)) for pagina in doc]
    return "\n\n".join(p for p in paginas if p.strip())


def convertir_pdf_a_md(nombre_pdf: str) -> dict:
    """Convierte UN pdf de data/docs/pdf a data/docs/md. Devuelve un resumen (no el texto)."""
    # Comenzamos asegurandonos que sí existen las carpetas
    asegurar_carpetas()
    origen = _nombre_seguro(nombre_pdf, ruta_pdf, ".pdf")
    # El .stem extrae el nombre del documento sin su extensión, por eso despues se utiliza el + ".md" para colocarle el formato
    destino = ruta_md / (origen.stem + ".md")
    texto = _extraer_texto_pymupdf(origen)
    # Crea el archivo y escribe lo que la variable texto obtuvo
    # El utf-8 soporta caracteres como el chino, arabe, acentos, etc.
    destino.write_text(texto, encoding="utf-8")

    # Creamos un diccionario que guarde la infromación del texto que creamos
    info = {
        "pdf": origen.name,
        "md": destino.name,
        "caracteres": len(texto), # Contar la cantidad de letras que hay en el texto
        "tokens_aprox": len(texto) // 4,  # regla practica: ~4 caracteres por token, se recomienda esto para el ingles
    }
    if len(texto.strip()) < 200: #  Condicional de seguridad si el documento parece ser escaneado, pues PyMuPDF no realiza OCR
        info["advertencia"] = (
            "Casi no tiene texto, puede ser un PDF escaneado."
        )
    return info


def convertir_pendientes() -> list[dict]:
    """Convierte todos los PDF que no tengan .md o cuyo .md sea mas viejo que el PDF."""
    asegurar_carpetas()
    resultados = [] # Vector vacio que permite alamcenar los resultados obtenidos en el ciclo
    # ruta_pdf.iterdir() se encarga de recorrer todos los archivos en la ruta seleccionada
    # p.suffix.lower() == ".pdf" se encarga que los documentos que terminen en .pdf sean seleccionados
    # El conjunto de todo lo que se encuentra dentro de sorted es una condicional
    # para cada elemento en la ruta donde el sufijo sea .pdf se aplica la funcion sorted
    # sorted ordena alfabeticamente
    # for pdf in sorted itera sobre los pdfs ya ordenados
    for pdf in sorted(p for p in ruta_pdf.iterdir() if p.suffix.lower() == ".pdf"):
        # El .stem extrae el nombre del documento sin su extensión, por eso despues se utiliza el + ".md" para colocarle el formato
        md = ruta_md / (pdf.stem + ".md")
        # Esto dice que si el pdf existe y que el tiempo de la ultima modificacion del .md es mayor o igual a la del .pdf no se crea otro documento
        # el >= indica que le documento está al día
        if md.exists() and md.stat().st_mtime >= pdf.stat().st_mtime:
            resultados.append({"pdf": pdf.name, "md": md.name, "estado": "ya estaba convertido"}) # Agrega la info a resultados
            continue # si se cumple la condicion, continua con el proceso
        try:
            info = convertir_pdf_a_md(pdf.name) # Aqui se utiliza la funcion convertir_pdf_a_md con su nombre completo, o sea con todo y formato
            info["estado"] = "convertido" # Agrega al diccionario la conversion 
        except Exception as e:  # un PDF dañado no debe detener a los demas
            info = {"pdf": pdf.name, "estado": f"error: {e}"}
        resultados.append(info) # Se agrega el diccionario a la lista de resultados
    return resultados 


# ----
# Markdown a fragmentos digeribles por el LLM 
# ----

# El dataclass es para que se sepa de que archivo y fragmento proviene un texto
@dataclass
class Fragmento:
    archivo: str
    seccion: str
    texto: str


def _es_encabezado(linea: str) -> bool:
    """MarkItDown no recupera los titulos de un PDF que se usan en un .md que normalmente se caracterisa por un #, pero, esta función reconoce lo siguiente:
    1. # Titulo por si el .md viene de otra fuente. 
    2. secciones numeradas como 2. Sección X o 3.1 Subsección Y 
    3. Lineas cortas en mayusculas.
    """
    linea = linea.strip() # Elimina los espacios en blanco
    if not linea or len(linea) > 100: # Si la lingitud del texto es mayor a 100 entonces supone que no es un encabezado
        return False
    if linea.startswith("#"): # Si el texto proviene de un .md o tiene los caracteristicos # al inicio, lo detecta como encabezado
        return True
    # la siguiente linea es regex para entender si la linea es un encabezado
    # ^ indica que el texto debe comenzar ahí. d+ para indigar uno o más digitos. (?:\.\d+){0,3} busca si hay subniveles tipo 2.2 o 2.3.3.
    # [.)]? punto de cierre pero es opcional, por eso tiene el ?. \s+ uno o más espacios en blanco. [A-ZÁÉÍÓÚÑ] Busca letras mayusculas  de la A a la Z y letras con acento. 
    # endswith busca que la linea no termine con esos signos de puntuación
    if re.match(r"^\d+(?:\.\d+){0,3}[.)]?\s+[A-ZÁÉÍÓÚÑ]", linea) and not linea.endswith((".", ",", ";", ":")):
        return True
    # La linea debe tener al menos 5 caracteres, estar todas en mayuscula y se asegura que haya letras, no solo numeros
    return len(linea) >= 5 and linea.isupper() and any(c.isalpha() for c in linea)


def _unidades(cuerpo: str):
    """los parrafos mas largos que max_caracteres se parten en oraciones."""
    for parrafo in re.split(r"\n\s*\n", cuerpo):     # Busca saltos de linea y espacios en blanco
        parrafo = " ".join(parrafo.split())  # colapsa los saltos de linea internos del PDF
        if not parrafo: 
            continue
        if len(parrafo) <= max_caracteres: 
            yield parrafo # El yield devuelve el parrafo si es más corto que el máximo de caracteres propuesto pero no es tan pesado como un return
            continue
        for oracion in re.split(r"(?<=[.!?])\s+", parrafo): # Separa el texto si encuentra alguno de esos signos de puntuación
            while len(oracion) > max_caracteres:  # Permite que las oraciones sean cortas en caso de ser necesario
                yield oracion[:max_caracteres]
                oracion = oracion[max_caracteres:] # Retoma la oración desde el último corte
            if oracion:
                yield oracion


def _cortar(cuerpo: str) -> list[str]:
    """Acumula párrafos y oraciones hasta exceder el criterio de max_caracteres"""
    trozos: list[str] = [] # Crea una lista de caracteres vacia donde se almacenará la info
    actual, ultima = "", ""
    for unidad in _unidades(cuerpo): # Se llama la funcion anterior que ya tiene sus anotaciones
        if actual and len(actual) + 1 + len(unidad) > max_caracteres: # Si ya tiene el texto y si valor es mayor que la suma indicada se agrega actual al vector trozos
            trozos.append(actual)
            actual = "" # Bloque nuevo desde 0
            # solape repite la ultima oracion para no perder contexto
            # Para evitar esto el codigo toma la ultima oración que utilizó del bloque y la coloca al inicio del bloque nuevo solo en caso de la que oración no sea gigantesca
            # y siempre y cuando quepa en max_caracteres
            if 0 < len(ultima) <= holgura and len(ultima) + 1 + len(unidad) <= max_caracteres:
                actual = ultima
        actual = f"{actual}\n{unidad}" if actual else unidad # Agrega un alto de linea y la nueva unidad de texto. Si la variable actual está vacia entonees se coloca la variable unodad
        ultima = unidad # Saber que fue lo ultimo que agregamos
    if actual: # Si existe actual entonces se agrega a la lista trozos
        trozos.append(actual)
    return trozos


def dividir_en_fragmentos(texto: str, archivo: str) -> list[Fragmento]:
    """Divide un documento en fragmentos de <= max_caracteres, respetando las secciones."""
    secciones: list[tuple[str, str]] = [] # Esta lista de tuplas guarda el titulo de la seccion y el texto de la seccion
    titulo, buffer = "(inicio del documento)", []

    # Funcion interna que permite cerrar la seccion para saber que texto pertenece a donde
    def cerrar():
        cuerpo = "\n".join(buffer).strip() # toma las lineas dentro de la variable buffer
        if cuerpo:# Si la variable cuarpo no esta vacia entonces se agrega la info a la variable secciones
            secciones.append((titulo, cuerpo)) 

    for linea in texto.splitlines(): # lee el texto linea por linea en 
        if _es_encabezado(linea): # Se utiliza la funcion _es_encabezado para reconocer donde utilizar la funcion interna cerrar
            cerrar()
            titulo, buffer = linea.strip().lstrip("#").strip(), [] # Se actualiza el titulo con la linea y comenzar a llenar el texto con una nueva seccion
            # El # quita los posible titulos si el documneto estaba en .md
        else:
            buffer.append(linea) # Si no es titulo entonces se agrega a la variable buffer
    cerrar()

    # Primero el ciclo for trozo in _cortar(cuerpo) para el cuarpo por la funcion _cortar(), entregando el cuerpo trozo a trozo
    # Despues, for t, cuerpo in secciones se encarga de sacar cada seccion con su contenido de secciones
    # Fragmento(archivo, t, trozo) por cada elemento crea informacion en el dataclass Fragmento
    return [Fragmento(archivo, t, trozo) for t, cuerpo in secciones for trozo in _cortar(cuerpo)]


def cargar_fragmentos() -> list[Fragmento]:
    asegurar_carpetas() # Asegura que existen las carpetas
    fragmentos: list[Fragmento] = []     # Lista de fragmentos almacenados
    for md in sorted(ruta_md.glob("*.md")): # Ciclo que lee todos los .md en la carpeta seleccionada en una sola lista
        fragmentos += dividir_en_fragmentos(md.read_text(encoding="utf-8"), md.name) # Agrega a una lista todos los fragmentos divididos de los .md en la carpeta seleccionada
    return fragmentos


# ----
#  Busqueda
# ----
_PALABRAS_VACIAS = {
    "de", "la", "el", "en", "y", "a", "los", "las", "del", "que", "se", "un", "una", "por",
    "con", "para", "es", "al", "lo", "como", "su", "o", "no", "si", "sus", "este", "esta",
} # Lista de palabras que podrian ser omitibles


def _tokens(texto: str) -> list[str]: # basicamente normalizamos el texto
    """Minusculas, sin acentos, numeros decimales como una sola palabra, plural simple."""
    sin_acentos = "".join(
        c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn"
    ) # unicodedata.normalize("NFD", texto.lower()) se convierte el texto a minuscula y se eliminan los acentos gracias NFD
    # Mn se refiere a Mark nonspacing, en pocas palabras si el caracter es una tilde la descarta
    salida = []
    for t in re.findall(r"[a-z_]+|\d+(?:[.,]\d+)?", sin_acentos): # busca lo que está de la a-z o dos o mas digitos seguidos de un punto y coma
        t = t.replace(",", ".") # Remplaza la coma por punto 
        if t in _PALABRAS_VACIAS or (len(t) < 2 and not t.isdigit()): # si t no es un numero, la longitud del texto es mayor a un caracter o no esta en la tupla _PALABRAS_VACIAS continua
            continue
        if len(t) > 3 and t.endswith("s"): 
            t = t[:-1]  # radios -> radio
        salida.append(t) # Se agrega a la lista salida
    return salida


def buscar(consulta: str, top_k: int = 3, fragmentos: list[Fragmento] | None = None) -> list[dict]:
    """Devuelve los `top_k` fragmentos mas relevantes para la consulta, con su fuente."""
    fragmentos = cargar_fragmentos() if fragmentos is None else fragmentos # Utiliza la funcion cargar_fragmentos si no existen fragmentos todavia 
    palabras = set(_tokens(consulta)) # Convierte la busqueda en tokens y los guarda en un set
    if not palabras or not fragmentos: # Si no hay nada que buscar se devuelve una lista vacia
        return []

    # Une la seccion y el texto guardado en el dataclass Fragmentos utilizando la funcion _tokens() para cada fragmento
    docs = [_tokens(f.seccion + " " + f.texto) for f in fragmentos]  
    n = len(docs) 
    longitud_media = sum(len(d) for d in docs) / n or 1
    frecuencia_doc = Counter(t for d in docs for t in set(d)) # cuenta en cuantos fragmentos distintos aparece la palabra
    k1, b = 1.5, 0.75 # K1 controla que tanto importa cuando se repite una palabra. b indica la importancia del tamaño del documento. Se utiliza BM25 

    puntuados = []
    for fragmento, d in zip(fragmentos, docs):
        tf, largo = Counter(d), len(d)
        puntaje = 0.0 
        for t in palabras: 
            if tf[t] == 0: # Si la palabra no esta en el fragmento, entonces pasa a otro
                continue
            idf = math.log(1 + (n - frecuencia_doc[t] + 0.5) / (frecuencia_doc[t] + 0.5)) # Frecuencia inversa en el documento o sea que tan raro es que tan rara es una palabra
            puntaje += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * largo / longitud_media)) # Formula BM25
        if puntaje > 0: # si el puntaje es mayor a 0 entonces se guarda en la lista puntuados
            puntuados.append((puntaje, fragmento))

    puntuados.sort(key=lambda par: -par[0]) # Ordena la lista de mayor a menor puntaje (por eso el signo -)
    return [
        {"archivo": f.archivo, "seccion": f.seccion, "texto": f.texto, "puntaje": round(p, 2)}
        for p, f in puntuados[:top_k]
    ] # Solo regresa los 3 mejores puntajes, redondeando el puntaje a dos decimales
