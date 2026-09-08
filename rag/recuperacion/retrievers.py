"""Los tres recuperadores: léxico, vectorial y el híbrido que los fusiona.

La clase lo resume así: BM25 acierta en coincidencias exactas —nombres propios, siglas,
identificadores— y los embeddings entienden el significado, incluso reformulado. Sobre
documentación técnica los dos casos aparecen mezclados en el mismo corpus: "¿cómo valido
antes de construir el modelo?" es semántica pura, y `AliasGenerator` es léxica pura.

Cómo se combinan importa tanto como combinarlos. `EnsembleRetriever` aplica **Reciprocal
Rank Fusion**: puntúa cada documento por 1/(c + posición) en cada ranking, con c=60, y
suma esas contribuciones ponderadas. Es decir, fusiona **posiciones, no puntajes** — que
es exactamente la advertencia de la clase. Un score de BM25 puede valer 14.7 y un coseno
0.83: sumarlos directamente deja que la escala arbitraria de BM25 domine el resultado.
"""

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ..config import NAMESPACE
from ..vectorstore import TASK_CONSULTA, obtener_vectorstore
from .tokenizacion import tokenizar

K_POR_DEFECTO = 5

# 50/50, el punto neutro. `evaluate.py --sweep` mide el efecto de moverlo: sobre este corpus
# el barrido es monótono a favor del vectorial y 0.3/0.7 da mejores números en las cuatro
# métricas. Aun así el default no se mueve — ajustarlo para ganar en un benchmark de 10
# preguntas es ajustar el sistema al examen, y la diferencia no se distingue del ruido con
# esa muestra. Está a un flag de distancia: `--pesos 0.3 0.7`.
PESOS_POR_DEFECTO = (0.5, 0.5)


def retriever_bm25(corpus: list[Document], k: int = K_POR_DEFECTO) -> BM25Retriever:
    """Recuperador léxico sobre el corpus completo, en memoria.

    `preprocess_func=tokenizar` no es un detalle: con el tokenizador por defecto, BM25 no
    hace match entre `TypeAdapter?` y `TypeAdapter`. Ver `tokenizacion.py`.
    """
    return BM25Retriever.from_documents(corpus, k=k, preprocess_func=tokenizar)


def retriever_vectorial(
    k: int = K_POR_DEFECTO,
    namespace: str = NAMESPACE,
    filtro: dict | None = None,
) -> BaseRetriever:
    """Recuperador semántico contra Pinecone.

    `filtro` es el "hard filter" por metadata de la clase: `{"categoria": {"$eq":
    "validacion"}}` acota el espacio de búsqueda antes de comparar similitud.
    """
    search_kwargs: dict = {"k": k, "namespace": namespace}
    if filtro:
        search_kwargs["filter"] = filtro

    return obtener_vectorstore(task_type=TASK_CONSULTA, namespace=namespace).as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )


def retriever_hibrido(
    corpus: list[Document],
    k: int = K_POR_DEFECTO,
    pesos: tuple[float, float] = PESOS_POR_DEFECTO,
    namespace: str = NAMESPACE,
    filtro: dict | None = None,
) -> EnsembleRetriever:
    """BM25 + vectorial, fusionados por RRF ponderado.

    Cada recuperador aporta sus k mejores y la fusión los reordena; la unión puede tener
    hasta 2k documentos, así que quien consume esto recorta al top-k final (lo hace
    `RAGSystem`). La deduplicación es por contenido: el mismo fragmento recuperado por
    ambos lados cuenta una vez, y sumar dos posiciones lo empuja hacia arriba, que es
    precisamente la señal que buscamos.
    """
    return EnsembleRetriever(
        retrievers=[
            retriever_bm25(corpus, k=k),
            retriever_vectorial(k=k, namespace=namespace, filtro=filtro),
        ],
        weights=list(pesos),
    )
