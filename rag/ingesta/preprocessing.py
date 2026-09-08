"""Limpieza previa al chunking.

El corpus son los `.md` fuente de la documentación de Pydantic, no la web renderizada:
traen directivas de MkDocs que no son contenido y que, si se indexan, gastan tokens y
ensucian tanto el embedding como el índice léxico de BM25.

Un caso concreto de por qué importa para BM25: la sintaxis de cross-reference de MkDocs
escribe ``[`TypeAdapter`][pydantic.TypeAdapter]``. Sin limpiar, el tokenizador ve también
`pydantic.TypeAdapter` y esa ruta de import se repite en decenas de documentos, así que
deja de discriminar. Nos quedamos con la etiqueta visible y descartamos el destino.
"""

import re

_COMENTARIO_HTML = re.compile(r"<!--.*?-->", re.DOTALL)
_INCLUDE_SNIPPET = re.compile(r"\{!.*?!\}")           # {!.../index.md!}
_ANCLA_TITULO = re.compile(r"\s*\{#[\w-]+\}")          # ## Título {#anchor}
_ATRIBUTOS_FENCE = re.compile(r"^(\s*```+\s*[\w-]*)\s*\{[^}]*\}", re.MULTILINE)
_CROSSREF = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")      # [`X`][pydantic.X] -> `X`
_LINEAS_VACIAS = re.compile(r"\n{3,}")


def limpiar(texto: str) -> str:
    """Devuelve el texto sin los artefactos de MkDocs."""
    texto = _COMENTARIO_HTML.sub("", texto)
    texto = _INCLUDE_SNIPPET.sub("", texto)
    texto = _ANCLA_TITULO.sub("", texto)
    texto = _ATRIBUTOS_FENCE.sub(r"\1", texto)
    texto = _CROSSREF.sub(r"\1", texto)
    texto = _LINEAS_VACIAS.sub("\n\n", texto)
    return texto.strip()


def titulo_de(texto: str, por_defecto: str = "") -> str:
    """Primer encabezado H1 del documento, para guardarlo en la metadata."""
    for linea in texto.splitlines():
        if linea.startswith("# "):
            return linea[2:].strip()
    return por_defecto
