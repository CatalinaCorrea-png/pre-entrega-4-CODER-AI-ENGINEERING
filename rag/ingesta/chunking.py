"""Fragmentación del corpus.

La consigna pide apuntar a fragmentos de 500-800 tokens. Lo que importa no es el parámetro
sino el tamaño **realizado**: el splitter corta en límites naturales del texto, así que
nunca llena el cupo. Con `chunk_size=600` la mediana real daba 390 tokens — por debajo del
rango pedido, aunque el número configurado estuviera adentro.

Medido sobre este corpus (16 documentos, 299 KB limpios):

    separadores   chunk_size   fragmentos   mediana   en 500-800
    encabezados      600          258        305          0
    encabezados      800          200        396         52
    párrafos         600          220        390          2
    párrafos         800          162        528         95      <- elegido

Por qué los separadores por párrafo y no por encabezado: la documentación de Pydantic
alterna prosa corta con bloques de código largos. Cortar en cada `##` produce fragmentos
de una o dos líneas —19 de ellos por debajo de 120 tokens— que no responden nada por sí
solos y que igual ocupan un vector y una entrada en el índice léxico. Con `\n\n` como
primer separador no queda ninguno por debajo de 120 tokens, y el encabezado igual viaja
pegado al párrafo que le sigue porque el merge es hacia adelante.

El solapamiento de 140 tokens (~17%) existe para que una definición partida entre dos
fragmentos aparezca completa por lo menos en uno.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 800      # tokens (tope; la mediana realizada es ~528)
CHUNK_OVERLAP = 140   # tokens

# Se prueban en orden: el splitter usa el primero que logre respetar el tamaño.
SEPARADORES = ["\n\n", "\n", " ", ""]


def obtener_splitter() -> RecursiveCharacterTextSplitter:
    """Splitter que mide en tokens reales, no en caracteres.

    `from_tiktoken_encoder` importa: 800 caracteres y 800 tokens difieren en un factor
    de ~4, y el límite que de verdad importa —cuánto contexto entra en el prompt— se
    mide en tokens.
    """
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARADORES,
        keep_separator=True,
    )


def fragmentar(documentos: list[Document]) -> list[Document]:
    """Divide los documentos en fragmentos, preservando su metadata de origen."""
    return obtener_splitter().split_documents(documentos)
