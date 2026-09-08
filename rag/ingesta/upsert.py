"""Subida de los fragmentos a Pinecone, en dos fases con lotes de distinto tamaño.

Insertar vector por vector no escala: cada uno paga una ida y vuelta de red completa. La
clase fija el punto dulce del upsert en lotes de 100 a 200. Pero el upsert no es el único
lote que hay acá, y **los dos están limitados por servicios distintos**:

- El lote de *embedding* lo acota Gemini, por tokens por minuto. Un lote de 100 fragmentos
  de este corpus son ~50.000 tokens y la capa gratuita lo rechaza con 429 RESOURCE_EXHAUSTED.
  Medido contra la API: ~8.000 tokens pasan, ~16.000 ya fallan.
- El lote de *upsert* lo acota Pinecone, por tamaño de request. Ahí sí valen los 100-200.

Por eso la ingesta corre en dos fases explícitas —embeber todo, después subir todo— en vez
de delegar ambas cosas en `PineconeVectorStore.add_documents()`, que las hace juntas y deja
el tamaño del lote de embedding fuera de nuestro control.

La contrapartida es que el texto original hay que escribirlo a mano en `metadata[TEXT_KEY]`.
No es una pérdida: la consigna pide explícitamente guardar el contenido en la metadata, y
así queda visible en el código del pipeline en lugar de ser un efecto secundario del
vectorstore. El lado de la recuperación lo lee de la misma clave.
"""

import time
from collections.abc import Callable, Iterator

from langchain_core.documents import Document

from ..config import NAMESPACE
from ..embeddings import TASK_DOCUMENTO, obtener_embeddings
from ..infra.pinecone_setup import obtener_indice
from .metadata import TEXT_KEY, ids_de

# Acotado por los tokens por minuto de Gemini, no por Pinecone: ~15 fragmentos de este
# corpus son ~8.000 tokens, que es lo que la capa gratuita acepta por llamada.
LOTE_EMBEDDING = 15

# El punto dulce del upsert que marca la clase. Es independiente del anterior.
LOTE_UPSERT = 100

# Pausa entre lotes de embedding, para no agotar la cuota por minuto. Con lotes de ~8.000
# tokens cada 20 segundos quedan ~24.000 tokens por minuto, debajo del límite medido.
PAUSA_ENTRE_LOTES = 20.0

INTENTOS = 4

# La cuota de Gemini se repone por ventana de un minuto, así que el primer reintento tiene
# que esperar más de un minuto. Un backoff que arranca en 4s agota los intentos dentro de
# la misma ventana y falla igual, solo que más tarde.
ESPERA_INICIAL = 65.0

SENALES_TRANSITORIAS = (
    "429", "rate limit", "resource_exhausted", "quota",
    "503", "unavailable", "timeout", "deadline",
)


def _es_transitorio(error: Exception) -> bool:
    mensaje = str(error).lower()
    return any(senal in mensaje for senal in SENALES_TRANSITORIAS)


def _con_reintentos(accion: Callable, descripcion: str):
    """Reintenta con backoff exponencial solo ante fallas transitorias.

    Una API key inválida no mejora esperando: reintentarla son cinco minutos perdidos para
    llegar al mismo error. Solo se reintenta lo que la espera puede arreglar.
    """
    espera = ESPERA_INICIAL
    for intento in range(1, INTENTOS + 1):
        try:
            return accion()
        except Exception as error:
            if intento == INTENTOS or not _es_transitorio(error):
                raise
            print(
                f"   ⏳ {descripcion}: cuota agotada, "
                f"reintento {intento}/{INTENTOS - 1} en {espera:.0f}s"
            )
            time.sleep(espera)
            espera *= 2


def _en_lotes(elementos: list, tamano: int) -> Iterator[list]:
    for inicio in range(0, len(elementos), tamano):
        yield elementos[inicio : inicio + tamano]


def embeber(
    chunks: list[Document],
    al_terminar_lote: Callable[[int, int], None] | None = None,
) -> list[list[float]]:
    """Genera los embeddings de todos los fragmentos, respetando la cuota por minuto."""
    modelo = obtener_embeddings(TASK_DOCUMENTO)
    vectores: list[list[float]] = []
    lotes = list(_en_lotes(chunks, LOTE_EMBEDDING))

    for numero, lote in enumerate(lotes, start=1):
        textos = [chunk.page_content for chunk in lote]
        vectores.extend(
            _con_reintentos(lambda: modelo.embed_documents(textos), f"embedding {numero}/{len(lotes)}")
        )
        if al_terminar_lote:
            al_terminar_lote(len(vectores), len(chunks))
        if numero < len(lotes):
            time.sleep(PAUSA_ENTRE_LOTES)

    return vectores


def subir(
    chunks: list[Document],
    vectores: list[list[float]],
    namespace: str = NAMESPACE,
    al_terminar_lote: Callable[[int, int], None] | None = None,
) -> int:
    """Hace upsert de los vectores con su metadata, en lotes de `LOTE_UPSERT`.

    Los IDs son determinísticos (`validators.md::3`), así que correr esto dos veces sobre
    el mismo corpus actualiza los vectores en lugar de duplicarlos.
    """
    indice = obtener_indice()
    ids = ids_de(chunks)

    registros = [
        {
            "id": identificador,
            "values": vector,
            # El texto original viaja acá adentro: una sola consulta a Pinecone devuelve
            # vector, contenido y fuente, sin una segunda base de datos que consultar.
            "metadata": {**chunk.metadata, TEXT_KEY: chunk.page_content},
        }
        for identificador, chunk, vector in zip(ids, chunks, vectores, strict=True)
    ]

    subidos = 0
    for lote in _en_lotes(registros, LOTE_UPSERT):
        _con_reintentos(
            lambda lote=lote: indice.upsert(vectors=lote, namespace=namespace),
            "upsert",
        )
        subidos += len(lote)
        if al_terminar_lote:
            al_terminar_lote(subidos, len(registros))

    return subidos


def borrar_namespace(namespace: str = NAMESPACE) -> None:
    """Vacía el namespace. Útil para reingestar desde cero tras cambiar el chunking.

    Hace falta porque los IDs determinísticos sobrescriben, pero no borran: si el corpus
    nuevo produce menos fragmentos que el anterior, los sobrantes quedan huérfanos en el
    índice y siguen apareciendo en las búsquedas.
    """
    indice = obtener_indice()
    stats = indice.describe_index_stats()
    if namespace in (stats.get("namespaces") or {}):
        indice.delete(delete_all=True, namespace=namespace)
