"""Rutas del proyecto y coordenadas del índice de Pinecone.

Las rutas se resuelven desde la ubicación de este archivo y no desde el directorio de
trabajo, para que los scripts funcionen invocados desde cualquier lado.

Los parámetros de cada etapa (tamaño de fragmento, top-k, pesos del ensemble) NO viven
acá: están en el módulo que los aplica, para que se lean junto al código que los usa.
Acá solo vive lo que necesita saber más de un módulo.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from .errores import ErrorDeUso

RAIZ = Path(__file__).resolve().parent.parent

load_dotenv(RAIZ / ".env")

DATA_DIR = RAIZ / "data"                        # corpus de origen (.md)
GOLDEN_SET_PATH = RAIZ / "golden_set.json"      # benchmark de evaluación

# El nombre del índice y el namespace son configurables por entorno a propósito: es lo que
# permite levantar un índice de staging sin tocar el código.
INDEX_NAME = os.getenv("INDEX_NAME", "pydantic-docs-rag")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "pydantic-concepts-v1")

PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")


class ConfiguracionFaltante(ErrorDeUso):
    """Falta una variable de entorno obligatoria."""


def variable_obligatoria(nombre: str) -> str:
    """Devuelve la variable de entorno o falla con un mensaje accionable.

    Preferimos esto a `os.environ[nombre]`: un KeyError pelado en medio de la ingesta no
    le dice a nadie que lo que falta es copiar `.env.example`.
    """
    valor = (os.getenv(nombre) or "").strip()
    if not valor:
        raise ConfiguracionFaltante(
            f"Falta la variable {nombre}.\n"
            f"  1. copy .env.example .env      (cp en Linux / macOS)\n"
            f"  2. completá {nombre} en el archivo .env\n"
            f"El .env está en .gitignore: nunca se sube al repositorio."
        )
    return valor
