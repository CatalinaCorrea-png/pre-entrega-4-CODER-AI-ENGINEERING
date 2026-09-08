"""Tokenizador para BM25.

`BM25Retriever` tokeniza por defecto con `str.split()`: corta por espacios y nada más.
Sobre documentación técnica eso desactiva justamente aquello para lo que sirve BM25.

    "¿Cómo uso TypeAdapter?".split()  ->  ['¿Cómo', 'uso', 'TypeAdapter?']

`TypeAdapter?` con el signo pegado no es el mismo término que el `TypeAdapter` del
documento, así que la coincidencia exacta —el único aporte real de BM25 frente al
vectorial— no ocurre. Tampoco coincidirían `ConfigDict` y `configdict`, ni `cómo` y
`como`.

Este tokenizador normaliza ambos lados igual: minúsculas, sin acentos, y cortando por
cualquier cosa que no sea letra, dígito o guion bajo. Se conserva el guion bajo porque en
Python es parte del identificador: `model_validator` es un término, no dos.
"""

import re
import unicodedata

_TOKEN = re.compile(r"[a-z0-9_]+")


def tokenizar(texto: str) -> list[str]:
    """Normaliza y parte en términos comparables."""
    descompuesto = unicodedata.normalize("NFKD", texto.lower())
    sin_acentos = "".join(c for c in descompuesto if not unicodedata.combining(c))
    return _TOKEN.findall(sin_acentos)
