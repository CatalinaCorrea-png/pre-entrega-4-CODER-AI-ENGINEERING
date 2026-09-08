"""Infraestructura: cliente de Pinecone e índice Serverless idempotente.

Dos reglas acá:

1. Crear el índice si no existe, y no romper si ya existe (se puede correr N veces).
2. Si existe, **verificar que sea compatible** antes de que alguien intente escribirle.
   Es lo que convierte el "mismatch de dimensiones" en un mensaje que dice qué hacer, en
   lugar de un `400 Vector dimension 1536 does not match the dimension of the index 768`
   a mitad de una ingesta ya empezada.
"""

import time
from functools import lru_cache

from pinecone import Pinecone, ServerlessSpec

from ..config import (
    INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    variable_obligatoria,
)
from ..embeddings import DIMENSION, METRICA
from ..errores import ErrorDeUso

ESPERA_MAXIMA_SEGUNDOS = 120


class IndiceIncompatible(ErrorDeUso):
    """El índice existe pero no coincide con la configuración del proyecto."""


@lru_cache(maxsize=1)
def obtener_cliente() -> Pinecone:
    """Cliente de Pinecone, cacheado por proceso."""
    return Pinecone(api_key=variable_obligatoria("PINECONE_API_KEY"))


def listar_indices() -> list[str]:
    return [indice["name"] for indice in obtener_cliente().list_indexes()]


def verificar_compatibilidad(nombre: str = INDEX_NAME) -> dict:
    """Compara el índice existente contra la configuración local.

    Falla ante una diferencia de dimensión o de métrica: son las dos cosas que no se
    pueden corregir después. Un índice de 768 no acepta vectores de 1536, y un índice
    creado con distancia euclidiana rankea distinto a uno con coseno aunque acepte los
    mismos vectores — el segundo error es peor porque no lanza ninguna excepción.
    """
    descripcion = obtener_cliente().describe_index(nombre)
    problemas = []

    if descripcion.dimension != DIMENSION:
        problemas.append(
            f"dimensión del índice = {descripcion.dimension}, "
            f"pero el modelo produce vectores de {DIMENSION}"
        )
    if descripcion.metric != METRICA:
        problemas.append(f"métrica del índice = '{descripcion.metric}', esperada '{METRICA}'")

    if problemas:
        raise IndiceIncompatible(
            f"El índice '{nombre}' ya existe pero no sirve para este proyecto:\n"
            + "".join(f"  - {p}\n" for p in problemas)
            + "Elegí una: borralo desde app.pinecone.io, o usá otro nombre con "
            "INDEX_NAME en tu .env."
        )

    return {
        "nombre": nombre,
        "dimension": descripcion.dimension,
        "metrica": descripcion.metric,
        "host": descripcion.host,
        "listo": descripcion.status.get("ready", False),
    }


def crear_indice_si_falta(nombre: str = INDEX_NAME) -> dict:
    """Crea el índice Serverless si no existe; si existe, lo valida. Idempotente."""
    if nombre in listar_indices():
        return {"creado": False, **verificar_compatibilidad(nombre)}

    obtener_cliente().create_index(
        name=nombre,
        dimension=DIMENSION,
        metric=METRICA,
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )
    esperar_a_que_este_listo(nombre)
    return {"creado": True, **verificar_compatibilidad(nombre)}


def esperar_a_que_este_listo(nombre: str = INDEX_NAME) -> None:
    """El índice tarda unos segundos en aprovisionarse; escribirle antes falla."""
    limite = time.monotonic() + ESPERA_MAXIMA_SEGUNDOS
    while time.monotonic() < limite:
        if obtener_cliente().describe_index(nombre).status.get("ready"):
            return
        time.sleep(2)
    raise TimeoutError(
        f"El índice '{nombre}' no quedó listo en {ESPERA_MAXIMA_SEGUNDOS}s. "
        "Revisá su estado en app.pinecone.io."
    )


def obtener_indice(nombre: str = INDEX_NAME):
    """Handle del índice para operaciones del SDK nativo (upsert, list, fetch)."""
    return obtener_cliente().Index(nombre)


def estadisticas(nombre: str = INDEX_NAME) -> dict:
    """`describe_index_stats()` como dict plano, para imprimir."""
    stats = obtener_indice(nombre).describe_index_stats()
    return {
        "vectores_totales": stats.get("total_vector_count", 0),
        "dimension": stats.get("dimension"),
        "namespaces": {
            nombre_ns: datos.get("vector_count", 0)
            for nombre_ns, datos in (stats.get("namespaces") or {}).items()
        },
    }
