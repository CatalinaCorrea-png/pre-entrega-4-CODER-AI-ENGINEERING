"""El corpus léxico de BM25, reconstruido desde Pinecone.

BM25 no es un servicio: es un índice invertido que se arma en memoria y necesita el texto
de **todos** los fragmentos. La pregunta es de dónde sale ese texto.

Lo obvio sería volver a leer `data/` y re-fragmentar. El problema es que ahí ya hay dos
corpus: el que está indexado en Pinecone y el que está en el disco de quien corre el
script. Si alguien edita `data/` y no reingesta, o cambia `CHUNK_SIZE`, los dos
recuperadores del híbrido pasan a buscar sobre colecciones distintas — y el ensemble
fusiona rankings de universos que no coinciden. No falla: devuelve métricas que no
significan nada.

Por eso el origen por defecto es Pinecone. Leemos `metadata["text"]`, que es exactamente
para lo que la consigna pide guardar el texto ahí: una sola fuente de verdad, sin segunda
base de datos. `--corpus local` queda como salida de emergencia para trabajar sin red.
"""

from langchain_core.documents import Document

from ..config import NAMESPACE
from ..errores import ErrorDeUso
from ..infra.pinecone_setup import obtener_indice
from ..ingesta.metadata import TEXT_KEY

TAMANO_LOTE_FETCH = 100


class CorpusVacioEnPinecone(ErrorDeUso):
    """El namespace existe pero no tiene vectores."""


def _metadata_de(vector) -> dict:
    """El SDK devuelve objetos o dicts según la versión; normalizamos a dict."""
    if isinstance(vector, dict):
        return vector.get("metadata") or {}
    return getattr(vector, "metadata", None) or {}


def corpus_desde_pinecone(namespace: str = NAMESPACE) -> list[Document]:
    """Pagina el namespace y reconstruye los fragmentos desde la metadata."""
    indice = obtener_indice()
    documentos: list[Document] = []

    for lote_ids in indice.list(namespace=namespace):
        for inicio in range(0, len(lote_ids), TAMANO_LOTE_FETCH):
            ids = lote_ids[inicio : inicio + TAMANO_LOTE_FETCH]
            respuesta = indice.fetch(ids=ids, namespace=namespace)
            vectores = respuesta.get("vectors") if isinstance(respuesta, dict) else respuesta.vectors
            for vector in (vectores or {}).values():
                metadata = dict(_metadata_de(vector))
                texto = metadata.pop(TEXT_KEY, "")
                if texto:
                    documentos.append(Document(page_content=texto, metadata=metadata))

    if not documentos:
        raise CorpusVacioEnPinecone(
            f"El namespace '{namespace}' no tiene vectores.\n"
            "Corré primero: python scripts/ingestar.py"
        )

    return _ordenar(documentos)


def corpus_desde_disco() -> list[Document]:
    """Re-deriva los fragmentos desde `data/`, sin tocar la red.

    Reproduce exactamente los mismos pasos que la ingesta (misma limpieza, mismo splitter,
    misma metadata), así que coincide con Pinecone **siempre que el índice esté al día**.
    """
    from ..ingesta.carga import cargar_documentos
    from ..ingesta.chunking import fragmentar
    from ..ingesta.metadata import aplicar_metadata

    return _ordenar(aplicar_metadata(fragmentar(cargar_documentos())))


def obtener_corpus(origen: str = "pinecone", namespace: str = NAMESPACE) -> list[Document]:
    if origen == "pinecone":
        return corpus_desde_pinecone(namespace)
    if origen == "local":
        return corpus_desde_disco()
    raise ValueError(f"Origen desconocido: {origen!r}. Usá 'pinecone' o 'local'.")


def _ordenar(documentos: list[Document]) -> list[Document]:
    """Orden estable: BM25 desempata por posición, así que el orden altera resultados."""
    return sorted(
        documentos,
        key=lambda d: (d.metadata.get("source", ""), d.metadata.get("chunk_index", 0)),
    )
