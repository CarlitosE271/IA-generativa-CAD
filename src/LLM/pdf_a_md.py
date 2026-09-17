"""
Este programa crea archivos markdown a partir de archivos pdf, con el motivo de reducir el consumo de tokens del LLM
"""

from markitdown import MarkItDown
import os

# 1. Configurar rutas
pdf_path = r"ruta_del_archivo"         
md_path = r"ruta_deseada"

try:
    markitdown = MarkItDown() # Esto inicializa el convertidor
  
    result = markitdown.convert(pdf_path) # Esto realiza la conversión

    # Guarda el contenido en el archivo .md
    with open(md_path, "w", encoding="utf-8") as file:
        file.write(result.text_content)

    # Mensaje de confirmación
    print("Conversión exitosa")
    print(f"Archivo guardado en: {md_path}")

# Casos en los que no sirve el convertidor
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{pdf_path}'. Revisa la ruta.")
except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")