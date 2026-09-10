"""El pipeline de ingesta, en orden de ejecución.

    data/*.md -> limpiar -> fragmentar -> metadata -> embeber -> subir a Pinecone

Las cinco etapas están una debajo de la otra para poder seguir el recorrido de un documento
leyendo el archivo de arriba abajo. Los parámetros viven junto al código que los aplica; el
porqué de cada número está en el README.
"""

import re
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import DATA_DIR, NAMESPACE, TEXT_KEY, ErrorDeUso
from .pinecone import TASK_DOCUMENTO, obtener_embeddings, obtener_indice


# --- 1. Limpieza: sacar los artefactos de MkDocs ----------------------------

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
    """Encabezado H1 del documento, si lo tiene, para guardarlo en la metadata."""
    for linea in texto.splitlines():
        if not linea.strip():
            continue
        return linea[2:].strip() if linea.startswith("# ") else por_defecto
    return por_defecto


# MkDocs saca el título de la navegación (`mkdocs.yml`), no del cuerpo del archivo: 14 de
# los 16 documentos de este corpus no tienen ningún H1. Sin este respaldo, `titulo` caía al
# nombre de archivo y quedaba duplicando `source` — un campo de metadata que no aporta
# ninguna información nueva es peso muerto en cada uno de los 162 vectores.
_ACRONIMOS = {"json": "JSON"}


def titulo_desde_archivo(nombre: str) -> str:
    """Título legible derivado del nombre: `json_schema.md` -> `JSON Schema`."""
    palabras = Path(nombre).stem.split("_")
    return " ".join(_ACRONIMOS.get(palabra, palabra.capitalize()) for palabra in palabras)


# --- 2. Carga: leer el corpus desde data/ -----------------------------------

class CorpusVacio(ErrorDeUso):
    """No hay documentos que ingestar."""


def cargar_documentos(directorio: Path = DATA_DIR, patron: str = "*.md") -> list[Document]:
    """Lee el corpus y lo devuelve limpio, con `source` normalizado al nombre de archivo."""
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
                metadata={"source": nombre, "titulo": titulo_de(texto, titulo_desde_archivo(nombre))},
            )
        )

    return sorted(documentos, key=lambda d: d.metadata["source"])


# --- 3. Fragmentación -------------------------------------------------------

CHUNK_SIZE = 800      # tokens (tope; la mediana realizada es ~528)
CHUNK_OVERLAP = 140   # tokens

# Se prueban en orden: el splitter usa el primero que logre respetar el tamaño.
SEPARADORES = ["\n\n", "\n", " ", ""]


def obtener_splitter() -> RecursiveCharacterTextSplitter:
    """Splitter que mide en tokens reales, no en caracteres."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARADORES,
        keep_separator=True,
    )


def fragmentar(documentos: list[Document]) -> list[Document]:
    """Divide los documentos en fragmentos, preservando su metadata de origen."""
    return obtener_splitter().split_documents(documentos)


# --- 4. Metadatos: el contrato, en un solo lugar ----------------------------

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
    """ID determinístico del vector: `archivo.md::7`."""
    return f"{source}::{chunk_index}"


def aplicar_metadata(chunks: list[Document]) -> list[Document]:
    """Completa la metadata de cada fragmento según el contrato de `CAMPOS`."""
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


# --- 5. Embedding y upsert: dos lotes, dos límites --------------------------

# Acotado por los tokens por minuto de Gemini, no por Pinecone: ~15 fragmentos de este
# corpus son ~8.000 tokens, que es lo que la capa gratuita acepta por llamada.
LOTE_EMBEDDING = 15

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
    """Reintenta con backoff exponencial solo ante fallas transitorias."""
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
) -> int:
    """Hace upsert de los vectores con su metadata, en lotes de `LOTE_UPSERT`."""
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

    for lote in _en_lotes(registros, LOTE_UPSERT):
        _con_reintentos(
            lambda lote=lote: indice.upsert(vectors=lote, namespace=namespace), "upsert"
        )

    return len(registros)


def borrar_namespace(namespace: str = NAMESPACE) -> None:
    """Vacía el namespace. Útil para reingestar desde cero tras cambiar el chunking."""
    indice = obtener_indice()
    stats = indice.describe_index_stats()
    if namespace in (stats.get("namespaces") or {}):
        indice.delete(delete_all=True, namespace=namespace)
