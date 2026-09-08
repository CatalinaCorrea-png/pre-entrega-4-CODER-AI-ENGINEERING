"""Carga de los documentos crudos desde `data/`."""

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

from ..config import DATA_DIR
from ..errores import ErrorDeUso
from .preprocessing import limpiar, titulo_de


class CorpusVacio(ErrorDeUso):
    """No hay documentos que ingestar."""


def cargar_documentos(directorio: Path = DATA_DIR, patron: str = "*.md") -> list[Document]:
    """Lee el corpus y lo devuelve limpio, con `source` normalizado al nombre de archivo.

    Normalizar `source` acá y no más adelante es deliberado: el loader guarda la ruta
    absoluta de esta máquina, y esa ruta terminaría publicada en la metadata de Pinecone.
    Además el golden set referencia los documentos por nombre de archivo, así que los dos
    lados tienen que hablar el mismo idioma.
    """
    if not directorio.exists() or not any(directorio.glob(patron)):
        raise CorpusVacio(
            f"No hay archivos {patron} en {directorio}.\n"
            "Corré primero: python scripts/descargar_dataset.py"
        )

    loader = DirectoryLoader(
        str(directorio),
        glob=patron,
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    documentos = []
    for doc in loader.load():
        nombre = Path(doc.metadata["source"]).name
        texto = limpiar(doc.page_content)
        documentos.append(
            Document(
                page_content=texto,
                metadata={"source": nombre, "titulo": titulo_de(texto, nombre)},
            )
        )

    return sorted(documentos, key=lambda d: d.metadata["source"])
