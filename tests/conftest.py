"""Hace importable la carpeta src/ desde las pruebas: `from herramientas import documentos`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
