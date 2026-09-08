"""El `PineconeVectorStore` del lado de la recuperación.

La consigna admite usar el vectorstore de LangChain o el SDK nativo. Acá se usan los dos,
cada uno donde encaja: la ingesta upserta con el SDK nativo porque necesita controlar por
separado el lote de embedding y el de upsert (ver `rag/ingesta/upsert.py`), y la consulta
usa este vectorstore porque es lo que `EnsembleRetriever` sabe consumir.

Lo que los mantiene compatibles es `TEXT_KEY`: la ingesta escribe el contenido en esa clave
de la metadata y `text_key` le dice al vectorstore que lo lea de ahí. Si divergieran, los
documentos volverían con `page_content` vacío y sin ningún error visible — por eso la clave
está definida una sola vez, en `rag/ingesta/metadata.py`.
"""

from functools import lru_cache

from langchain_pinecone import PineconeVectorStore

from .config import NAMESPACE
from .embeddings import TASK_CONSULTA, TASK_DOCUMENTO, obtener_embeddings
from .infra.pinecone_setup import obtener_indice
from .ingesta.metadata import TEXT_KEY

__all__ = ["obtener_vectorstore", "TASK_CONSULTA", "TASK_DOCUMENTO"]


@lru_cache(maxsize=4)
def obtener_vectorstore(
    task_type: str = TASK_CONSULTA,
    namespace: str = NAMESPACE,
) -> PineconeVectorStore:
    """Vector store apuntado al índice y namespace configurados.

    `text_key=TEXT_KEY` es lo que hace que el contenido del fragmento viaje dentro de la
    metadata del vector, como pide la consigna: una sola consulta a Pinecone devuelve
    vector, texto y fuente, sin una segunda base de datos que consultar.
    """
    return PineconeVectorStore(
        index=obtener_indice(),
        embedding=obtener_embeddings(task_type),
        text_key=TEXT_KEY,
        namespace=namespace,
    )
