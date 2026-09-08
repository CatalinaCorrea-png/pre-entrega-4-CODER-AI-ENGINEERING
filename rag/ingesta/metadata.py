"""El esquema de metadatos: un solo lugar donde se construye, para todos los pipelines.

La clase le puso nombre al error que esto evita: **schema drift**. Si un proceso escribe
`categoria` y otro `category`, ningún filtro falla — simplemente devuelven cero resultados
y el sistema parece funcionar. Por eso la metadata no se arma ad hoc en cada script: se
arma acá, y `CAMPOS` documenta el contrato.

Sobre `text`: la consigna pide guardar el texto original dentro de la metadata para no
tener que consultar otra base. No lo escribimos nosotros — `PineconeVectorStore` copia el
`page_content` en la clave `TEXT_KEY` al hacer el upsert. Lo declaramos igual acá porque
es parte del contrato: `rag/recuperacion/corpus.py` lee de ahí para reconstruir BM25.
"""

from pathlib import Path

from langchain_core.documents import Document

# Clave donde vive el texto original dentro de la metadata del vector.
TEXT_KEY = "text"

# Contrato de metadatos. Cualquier pipeline que escriba en este índice escribe esto.
CAMPOS = (
    "text",           # str  - contenido del fragmento (lo inyecta PineconeVectorStore)
    "source",         # str  - nombre de archivo, ej. "validators.md"
    "titulo",         # str  - H1 del documento de origen
    "categoria",      # str  - agrupación temática, para filtrar por metadata
    "doc_type",       # str  - tipo de documento
    "chunk_index",    # int  - posición del fragmento dentro de su documento
    "total_chunks",   # int  - cuántos fragmentos produjo ese documento
    "n_caracteres",   # int  - tamaño del fragmento
    "url",            # str  - página pública de origen, para citar la fuente
)

# Pinecone rechaza el vector entero si su metadata supera 40 KB. Un fragmento de 600
# tokens ronda los 2.5 KB, así que hay margen de sobra, pero el corte explícito evita que
# un documento raro tire abajo un lote completo.
LIMITE_TEXTO_BYTES = 30_000

DOC_TYPE = "documentacion_tecnica"

# Agrupación temática de la documentación de Pydantic. Es la metadata que habilita el
# "hard filter" del que habla la clase: `filter={"categoria": {"$eq": "validacion"}}`
# acota el espacio de búsqueda antes de comparar similitud.
CATEGORIAS = {
    "models.md": "modelado",
    "fields.md": "modelado",
    "dataclasses.md": "modelado",
    "alias.md": "modelado",
    "validators.md": "validacion",
    "validation_decorator.md": "validacion",
    "strict_mode.md": "validacion",
    "serialization.md": "serializacion",
    "json.md": "serializacion",
    "json_schema.md": "serializacion",
    "types.md": "tipos",
    "unions.md": "tipos",
    "type_adapter.md": "tipos",
    "forward_annotations.md": "tipos",
    "config.md": "configuracion",
    "performance.md": "configuracion",
}

URL_BASE = "https://docs.pydantic.dev/latest/concepts/{slug}/"


def categoria_de(source: str) -> str:
    return CATEGORIAS.get(source, "sin_categoria")


def url_de(source: str) -> str:
    return URL_BASE.format(slug=Path(source).stem)


def id_vector(source: str, chunk_index: int) -> str:
    """ID determinístico del vector: `archivo.md::7`.

    Determinístico y no aleatorio a propósito. Con UUIDs, reingestar el mismo corpus
    duplica cada fragmento y el top-5 se llena de copias del mismo texto: la precisión
    medida sube sin que el sistema haya mejorado en nada. Con este esquema, reingestar
    sobrescribe — que es lo que la palabra *upsert* promete.
    """
    return f"{source}::{chunk_index}"


def aplicar_metadata(chunks: list[Document]) -> list[Document]:
    """Completa la metadata de cada fragmento según el contrato de `CAMPOS`.

    Numera los fragmentos por documento (no globalmente) para que `chunk_index` siga
    siendo válido si mañana se reingesta un solo archivo.
    """
    por_documento: dict[str, list[Document]] = {}
    for chunk in chunks:
        por_documento.setdefault(chunk.metadata.get("source", "desconocido"), []).append(chunk)

    resultado = []
    for source, fragmentos in por_documento.items():
        total = len(fragmentos)
        for indice, chunk in enumerate(fragmentos):
            texto = chunk.page_content[:LIMITE_TEXTO_BYTES]
            chunk.page_content = texto
            chunk.metadata = {
                "source": source,
                "titulo": chunk.metadata.get("titulo", source),
                "categoria": categoria_de(source),
                "doc_type": DOC_TYPE,
                "chunk_index": indice,
                "total_chunks": total,
                "n_caracteres": len(texto),
                "url": url_de(source),
            }
            resultado.append(chunk)

    return sorted(resultado, key=lambda d: (d.metadata["source"], d.metadata["chunk_index"]))


def ids_de(chunks: list[Document]) -> list[str]:
    return [id_vector(c.metadata["source"], c.metadata["chunk_index"]) for c in chunks]
