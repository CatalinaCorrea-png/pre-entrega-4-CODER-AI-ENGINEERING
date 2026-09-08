"""El modelo de embeddings y la dimensión del índice, en un único lugar.

`DIMENSION` es la única fuente de verdad del proyecto: la lee el setup del índice y la
lee la ingesta. Ese es el punto — el "mismatch de dimensiones" que la consigna marca como
error típico solo puede ocurrir si el número está escrito dos veces y las copias divergen.
Acá no hay dos copias.

`gemini-embedding-001` acepta `output_dimensionality`, así que producimos vectores de 1536
como pide la consigna sin depender de OpenAI. El modelo entrena con Matryoshka
Representation Learning: recortar de 3072 a 1536 conserva la información más significativa
en vez de truncar a ciegas.
"""

from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .config import variable_obligatoria

# Cuidado si se cambia `METRICA`: al recortar por debajo de 3072, Gemini devuelve vectores
# sin normalizar (norma medida ~0.69). Con `cosine` da igual, porque el coseno divide por
# las normas; con `dotproduct`, la magnitud entraría directo en el score y el ranking
# quedaría sesgado hacia los vectores más largos.
MODELO = "models/gemini-embedding-001"
DIMENSION = 1536
METRICA = "cosine"

# Gemini pide declarar para qué se va a usar el vector. No es cosmético: el modelo proyecta
# distinto un pasaje que se va a indexar que una pregunta que lo busca, y usar el par
# correcto mejora la recuperación sobre usar el mismo tipo para ambos lados.
TASK_DOCUMENTO = "RETRIEVAL_DOCUMENT"
TASK_CONSULTA = "RETRIEVAL_QUERY"

# La API rechaza lotes más grandes; ver rag/ingesta/upsert.py.
MAX_TEXTOS_POR_LOTE = 100


@lru_cache(maxsize=2)
def obtener_embeddings(task_type: str = TASK_DOCUMENTO) -> GoogleGenerativeAIEmbeddings:
    """Cliente de embeddings, cacheado por task_type.

    Se cachea para no reconstruir el cliente en cada llamada, y por task_type porque la
    ingesta y la consulta necesitan instancias distintas: LangChain aplica el task_type
    configurado tanto a `embed_documents` como a `embed_query`.
    """
    return GoogleGenerativeAIEmbeddings(
        model=MODELO,
        google_api_key=variable_obligatoria("GOOGLE_API_KEY"),
        output_dimensionality=DIMENSION,
        task_type=task_type,
    )
