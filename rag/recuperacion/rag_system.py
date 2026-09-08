"""`RAGSystem`: la fachada del módulo de recuperación.

Recibe una consulta y devuelve los top-k fragmentos. Encapsula el `EnsembleRetriever` para
que ni los scripts ni `evaluate.py` tengan que saber cómo se arma.

Los tres modos existen por la evaluación. Decir "implementé un híbrido" no significa nada
sin el contrafáctico: hay que poder correr el mismo golden set contra BM25 solo y contra
el vectorial solo, y mostrar el número. Sin eso no se sabe si el ensemble suma o estorba.
"""

from functools import cached_property

from langchain_core.documents import Document

from ..config import NAMESPACE
from .corpus import obtener_corpus
from .retrievers import (
    K_POR_DEFECTO,
    PESOS_POR_DEFECTO,
    retriever_bm25,
    retriever_hibrido,
    retriever_vectorial,
)

MODOS = ("hibrido", "vectorial", "bm25")


class RAGSystem:
    """Sistema de recuperación sobre el índice de Pinecone.

    La construcción es perezosa: instanciar la clase no descarga el corpus ni contacta a
    Pinecone. El trabajo se paga en la primera consulta y se cachea.
    """

    def __init__(
        self,
        k: int = K_POR_DEFECTO,
        modo: str = "hibrido",
        pesos: tuple[float, float] = PESOS_POR_DEFECTO,
        namespace: str = NAMESPACE,
        origen_corpus: str = "pinecone",
        corpus: list[Document] | None = None,
        filtro: dict | None = None,
    ) -> None:
        if modo not in MODOS:
            raise ValueError(f"Modo desconocido: {modo!r}. Opciones: {', '.join(MODOS)}")

        self.k = k
        self.modo = modo
        self.pesos = tuple(pesos)
        self.namespace = namespace
        self.origen_corpus = origen_corpus
        self.filtro = filtro
        # El corpus se puede inyectar para compartirlo entre varias instancias: en la
        # evaluación se comparan tres modos y bajarlo tres veces sería tiempo tirado.
        self._corpus = corpus

    @cached_property
    def corpus(self) -> list[Document]:
        if self._corpus is None:
            self._corpus = obtener_corpus(self.origen_corpus, self.namespace)
        return self._corpus

    @cached_property
    def retriever(self):
        if self.modo == "bm25":
            return retriever_bm25(self.corpus, k=self.k)
        if self.modo == "vectorial":
            return retriever_vectorial(k=self.k, namespace=self.namespace, filtro=self.filtro)
        return retriever_hibrido(
            self.corpus, k=self.k, pesos=self.pesos, namespace=self.namespace, filtro=self.filtro
        )

    def recuperar(self, consulta: str) -> list[Document]:
        """Los `k` fragmentos más relevantes para la consulta.

        El recorte final es necesario: el ensemble devuelve la unión de ambos rankings, que
        puede tener hasta 2k elementos. Sin el corte, "top-5" sería un top-10 disfrazado y
        la Precision@5 estaría midiendo otra cosa.
        """
        return self.retriever.invoke(consulta)[: self.k]

    def obtener_top_k(self, consulta: str) -> list[dict]:
        """Igual que `recuperar`, en dicts planos listos para imprimir o serializar."""
        return [
            {
                "contenido": documento.page_content,
                "fuente": documento.metadata.get("source", "desconocida"),
                "titulo": documento.metadata.get("titulo", ""),
                "categoria": documento.metadata.get("categoria", "sin_categoria"),
                "chunk_index": documento.metadata.get("chunk_index"),
                "url": documento.metadata.get("url", ""),
            }
            for documento in self.recuperar(consulta)
        ]

    def fuentes(self, consulta: str) -> list[str]:
        """Los nombres de archivo del top-k, en orden. Es lo que consume la evaluación."""
        return [d.metadata.get("source", "desconocida") for d in self.recuperar(consulta)]

    def __repr__(self) -> str:
        detalle = f", pesos={self.pesos}" if self.modo == "hibrido" else ""
        return f"RAGSystem(modo={self.modo!r}, k={self.k}{detalle}, namespace={self.namespace!r})"
