# Pre-entrega 4 — Módulo de recuperación escalable con Pinecone

Sistema RAG de recuperación sobre documentación técnica, con el índice vectorial en la
nube. Ingesta un corpus de documentación de una librería de Python a **Pinecone
Serverless** con metadatos estructurados, recupera con un **retriever híbrido** que fusiona
búsqueda léxica (BM25) y semántica (embeddings), y **mide** el resultado contra un golden
set con Precision@5, Recall@5 y MRR@5.

| Componente | Elección |
|---|---|
| Base vectorial | Pinecone Serverless (`aws / us-east-1`), métrica coseno |
| Embeddings | `gemini-embedding-001` con `output_dimensionality=1536` |
| Orquestación | LangChain — `PineconeVectorStore`, `BM25Retriever`, `EnsembleRetriever` |
| Fusión | Reciprocal Rank Fusion ponderado (c=60) |
| Corpus | Documentación de conceptos de **Pydantic** `v2.13.5` — 16 documentos, 299 KB |
| Evaluación | Golden set de 10 preguntas · Precision@5, Recall@5, hit-rate, MRR@5 |

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt

copy .env.example .env          # Windows  (cp en Linux / macOS)
```

Requiere **Python 3.12** — `langchain-pinecone` todavía no soporta 3.13+.

Completar en `.env`:

- `PINECONE_API_KEY` — gratis en [app.pinecone.io](https://app.pinecone.io)
- `GOOGLE_API_KEY` — gratis en [Google AI Studio](https://aistudio.google.com/apikey)

El `.env` está en `.gitignore` y nunca se sube al repositorio.

## Replicar el índice de Pinecone

Cuatro comandos, en orden. Todos son idempotentes: se pueden repetir sin romper nada.

```bash
python scripts/descargar_dataset.py   # 1. baja el corpus a data/  (clavado al tag v2.13.5)
python scripts/setup_indice.py        # 2. crea el índice Serverless si no existe
python scripts/ingestar.py            # 3. limpia, fragmenta, embebe y sube en lotes
python evaluate.py                    # 4. mide contra el golden set e imprime el reporte
```

**1. Corpus.** Descarga 16 archivos `.md` de `pydantic/pydantic@v2.13.5`. Clavado a un tag
y no a `main`: si el corpus cambia, las métricas de este README dejan de ser reproducibles.

**2. Índice.** Crea uno Serverless con `dimension=1536`, `metric=cosine`,
`ServerlessSpec(cloud="aws", region="us-east-1")` — la combinación que admite la capa
gratuita de Pinecone. Si el índice ya existe, **verifica** que su dimensión y su métrica
coincidan con las del modelo, y aborta con un mensaje accionable si no.

**3. Ingesta.** Genera 162 embeddings y los sube al namespace `pydantic-concepts-v1`. Tarda
unos 4 minutos: el cuello de botella no es Pinecone sino la cuota por minuto de Gemini (ver
[Dos lotes, dos límites](#dos-lotes-dos-límites)).

**4. Evaluación.** Corre las 10 preguntas del golden set contra las tres configuraciones
(BM25 solo, vectorial solo, híbrido) e imprime la comparativa.

Para reindexar desde cero después de cambiar el chunking:

```bash
python scripts/ingestar.py --limpiar
```

## Uso

```bash
python evaluate.py                              # los tres modos, top-5
python evaluate.py --sweep                      # + barrido de pesos del ensemble
python evaluate.py --modos hibrido --sin-detalle
python evaluate.py --corpus local               # BM25 sin depender del índice

python scripts/consultar.py                     # consola interactiva
python scripts/consultar.py "como uso TypeAdapter"
python scripts/consultar.py --modo bm25 "AliasGenerator"
python scripts/consultar.py --categoria validacion "alias"   # filtro por metadata

python tests/test_metricas.py                   # pruebas offline, sin red ni API keys
```

Los scripts funcionan invocados desde cualquier directorio.

## Mapa del entregable

Cada requisito de la consigna y dónde está resuelto, **en orden de ejecución**.

### 0 · Configuración de variables

> *"Crea un archivo `.env` con `PINECONE_API_KEY`, `OPENAI_API_KEY` (o Anthropic) e `INDEX_NAME`."*

| Qué | Dónde |
|---|---|
| Plantilla de variables | `.env.example` |
| `INDEX_NAME` y namespace | `rag/config.py` → `INDEX_NAME`, `NAMESPACE` |
| Región del índice serverless | `rag/config.py` → `PINECONE_CLOUD`, `PINECONE_REGION` |
| Falla accionable si falta una clave | `rag/config.py` → `variable_obligatoria()` |
| Que el `.env` nunca se suba | `.gitignore` |

En lugar de `OPENAI_API_KEY` va `GOOGLE_API_KEY`: la consigna admite otro proveedor, y
Gemini llega a las mismas 1536 dimensiones sin costo.

### 1 · Dataset — `python scripts/descargar_dataset.py`

> *"Carga un dataset de documentos técnicos (puedes usar la documentación de una librería de Python)."*

| Qué | Dónde |
|---|---|
| Descarga los 16 `.md` de Pydantic, clavados al tag `v2.13.5` | `scripts/descargar_dataset.py` |
| Qué archivos entran al corpus, y su categoría | `rag/ingesta.py` → `CATEGORIAS` |

### 2 · Infraestructura — `python scripts/setup_indice.py`

> *"Escribe un script de inicialización que verifique si el índice existe y lo cree si es necesario (modo Serverless)."* · *"Crea un índice Serverless (usa la dimensión 1536)."*

| Qué | Dónde |
|---|---|
| Punto de entrada | `scripts/setup_indice.py` |
| Crea si falta, valida si ya existe | `rag/pinecone.py` → `crear_indice_si_falta()` |
| **Evita el mismatch de dimensiones** | `rag/pinecone.py` → `verificar_compatibilidad()` |
| `DIMENSION = 1536`, única fuente de verdad | `rag/pinecone.py` → `DIMENSION` |
| `ServerlessSpec(cloud, region)` | `rag/pinecone.py` → `crear_indice_si_falta()` |
| Espera a que el índice quede listo | `rag/pinecone.py` → `esperar_a_que_este_listo()` |
| **Namespace** (error a evitar) | `rag/config.py` → `NAMESPACE` |

### 3 · Ingesta — `python scripts/ingestar.py`

> *"Un script que tome un conjunto de documentos, los procese y los suba a un índice de Pinecone Serverless utilizando metadatos avanzados (fuente, página, etiquetas de categoría)."*

Las cinco etapas viven en `rag/ingesta.py`, una debajo de la otra en el orden en que corren
(cada una marcada con su separador `# --- N.`):

| # | Qué hace | Dónde |
|---|---|---|
| 1 | Punto de entrada, orquesta todo | `scripts/ingestar.py` |
| 2 | Limpia directivas de MkDocs | `rag/ingesta.py` → `limpiar()` |
| 3 | Lee los `.md`, normaliza `source` | `rag/ingesta.py` → `cargar_documentos()` |
| 4 | **`RecursiveCharacterTextSplitter`** | `rag/ingesta.py` → `obtener_splitter()` |
| 5 | Chunking ~500-800 tokens (error a evitar) | `rag/ingesta.py` → `CHUNK_SIZE = 800`, `CHUNK_OVERLAP = 140` |
| 6 | **Metadatos avanzados** — el contrato | `rag/ingesta.py` → `CAMPOS` |
| 7 | Etiquetas de categoría | `rag/ingesta.py` → `categoria_de()` |
| 8 | IDs determinísticos (upsert real, no duplicado) | `rag/ingesta.py` → `id_vector()` |
| 9 | Arma la metadata de cada fragmento | `rag/ingesta.py` → `aplicar_metadata()` |
| 10 | **Genera los embeddings**, lotes de 15 | `rag/ingesta.py` → `embeber()` |
| 11 | **Batch upsert a Pinecone**, lotes de 100 | `rag/ingesta.py` → `subir()`, `LOTE_UPSERT` |
| 12 | **Guarda el texto original en la metadata** | `rag/config.py` → `TEXT_KEY`, escrito en `subir()` |

El punto 12 es el que la consigna subraya (*"guardá el texto original dentro de los
metadatos para evitar consultas adicionales a una base relacional"*). `TEXT_KEY` vive en
`config.py` y no en `ingesta.py` porque lo comparten tres módulos: la ingesta lo escribe, el
vector store lo lee como `text_key`, y la recuperación lo usa para reconstruir el corpus de
BM25.

### 4 · Recuperador — lo usan los pasos 5 y 6

> *"Crea una clase `RAGSystem` que encapsule un `EnsembleRetriever`. El sistema debe recibir una consulta y devolver los top-5 documentos combinando resultados léxicos y semánticos."*

Todo en `rag/recuperacion.py`, ordenado de las piezas hacia la fachada:

| Qué | Dónde |
|---|---|
| Tokenizador que hace funcionar a BM25 | `rag/recuperacion.py` → `tokenizar()` |
| Corpus léxico traído desde Pinecone | `rag/recuperacion.py` → `corpus_desde_pinecone()` |
| **`BM25Retriever`** | `rag/recuperacion.py` → `retriever_bm25()` |
| Retriever vectorial (semántico) | `rag/recuperacion.py` → `retriever_vectorial()` |
| **`EnsembleRetriever`** | `rag/recuperacion.py` → `retriever_hibrido()` |
| Pesos de la fusión RRF | `rag/recuperacion.py` → `PESOS_POR_DEFECTO` |
| **Clase `RAGSystem`** | `rag/recuperacion.py` → `RAGSystem` |
| Recibe consulta → devuelve top-5 | `rag/recuperacion.py` → `RAGSystem.recuperar()` |
| Mismo top-5 en dicts planos | `rag/recuperacion.py` → `RAGSystem.obtener_top_k()` |
| **`PineconeVectorStore` de LangChain** | `rag/pinecone.py` → `obtener_vectorstore()` |

Para probarlo a mano: `python scripts/consultar.py`.

### 5 · Golden set

> *"Crea un pequeño archivo JSON con pares `{"pregunta": "...", "documento_id_esperado": "..."}`"* · *"Define un benchmark de 5 preguntas."*

| Qué | Dónde |
|---|---|
| El JSON, 10 casos (la consigna pide 5) | `golden_set.json` |
| Lectura y validación contra el corpus | `rag/evaluacion.py` → `cargar()` |
| Acepta el formato `documento_id_esperado` de la consigna | `rag/evaluacion.py` → `_documentos_de()` |

### 6 · Evaluación — `python evaluate.py`

> *"Crea un script `evaluate.py`... calcula Recall@5 y Precision@5... imprime en consola un breve resumen."*

| Qué | Dónde |
|---|---|
| **El script que nombra la consigna** | `evaluate.py` |
| **`Precision@5`** | `rag/evaluacion.py` → `precision_at_k()` |
| **`Recall@5`** | `rag/evaluacion.py` → `recall_at_k()` |
| "¿Está el documento correcto entre los 5?" | `rag/evaluacion.py` → `hit_at_k()` |
| Extras: MRR y techo de precisión | `rag/evaluacion.py` → `mrr_at_k()`, `techo_precision()` |
| Promedios sobre el golden set | `rag/evaluacion.py` → `Reporte` |
| **Reporte por consola** | `evaluate.py` → `imprimir_comparativa()` |
| Detalle pregunta por pregunta | `evaluate.py` → `imprimir_detalle()` |
| Barrido de pesos | `evaluate.py` → `PESOS_DEL_BARRIDO`, `imprimir_barrido()` |

### Fuera de la consigna

`tests/test_metricas.py` (10 pruebas que corren sin red ni API keys) y `rag/config.py`
(rutas, `ErrorDeUso` y consola UTF-8, que sostienen al resto).

## Estructura

```
├── data/                    # corpus descargado (16 .md + FUENTE.txt)
├── golden_set.json          # benchmark de 10 preguntas
├── evaluate.py              # script de evaluación (el que nombra la consigna)
├── rag/                     # librería: no imprime ni se ejecuta sola
│   ├── config.py            # rutas, .env, ErrorDeUso, consola
│   ├── pinecone.py          # embeddings + índice serverless + vector store
│   ├── ingesta.py           # limpiar → fragmentar → metadata → embeber → subir
│   ├── recuperacion.py      # tokenizador, corpus, retrievers, RAGSystem
│   └── evaluacion.py        # golden set + métricas
├── scripts/                 # puntos de entrada ejecutables
│   ├── descargar_dataset.py
│   ├── setup_indice.py
│   ├── ingestar.py
│   └── consultar.py
└── tests/test_metricas.py   # pruebas offline, sin red ni API keys
```

Cinco módulos de librería, uno por responsabilidad, y cada uno se lee de arriba abajo en el
orden en que se ejecuta. `rag/ingesta.py` recorre las cinco etapas del pipeline en secuencia;
`rag/recuperacion.py` va del tokenizador hasta `RAGSystem`. La alternativa —un archivo por
etapa— daba 28 archivos para 1.000 líneas de código y obligaba a saltar entre cinco de ellos
para seguir un solo flujo.

`embeddings` y `pinecone` viven juntos a propósito: `DIMENSION` tiene que ser la misma para
el modelo y para el índice, y a la vista uno del otro no hay dos copias que puedan divergir.

Los parámetros de cada etapa viven junto al código que los aplica (`CHUNK_SIZE` con el
splitter, `PESOS_POR_DEFECTO` con el ensemble), no centralizados. `config.py` guarda solo lo
que necesita más de un módulo — incluido `TEXT_KEY`, que escriben la ingesta, el vector store
y la recuperación, y que tiene que coincidir en los tres.

La inicialización es **perezosa**: importar `rag` no contacta a Pinecone ni descarga el
corpus. `RAGSystem` paga ese trabajo en la primera consulta y lo cachea.

## Decisiones de diseño

### Por qué documentación de Pydantic y no un corpus genérico

El corpus decide si el recuperador híbrido tiene algo que fusionar o es decorativo. La
documentación técnica está llena de identificadores exactos —`TypeAdapter`,
`model_validator`, `AliasGenerator`, `json_schema_extra`— que en principio son el terreno de
BM25: son literales distintos, aunque hablen todos de "validar modelos". Sobre un corpus de
prosa general, la búsqueda léxica no tendría ninguna carta que jugar.

La hipótesis era que BM25 ganaría en esas preguntas y el vectorial en las reformulaciones.
**La medición la desmintió a medias**: BM25 sirve, pero el vectorial le gana también en las
léxicas. El corpus igual cumplió su función — permitió montar el experimento que muestra
dónde se rompe cada mitad. Ver [Qué dicen estos números](#qué-dicen-estos-números).

### Chunking: el parámetro no es el tamaño real

La consigna pide apuntar a 500-800 tokens. Lo que importa es el tamaño **realizado**: el
splitter corta en límites naturales del texto y nunca llena el cupo. Medido sobre este
corpus:

| separadores | `chunk_size` | fragmentos | mediana | en 500-800 | menos de 120 tokens |
|---|---|---|---|---|---|
| encabezados | 600 | 258 | 305 | 0 | 19 |
| encabezados | 800 | 200 | 396 | 52 | 12 |
| párrafos | 600 | 220 | 390 | 2 | 4 |
| **párrafos** | **800** | **162** | **528** | **95** | **0** |

Con `chunk_size=600` y separadores por encabezado —lo más intuitivo para Markdown— la
mediana real daba 305 tokens, por debajo del rango pedido, y 19 fragmentos quedaban por
debajo de 120 tokens: líneas sueltas que no responden nada y que igual ocupan un vector.
La configuración elegida (`\n\n` como primer separador, `chunk_size=800`, overlap 140) deja
la mediana en 528 y ningún fragmento minúsculo.

### Un solo número para la dimensión

`DIMENSION = 1536` vive en `rag/pinecone.py` y de ahí la leen el setup del índice y la
ingesta. El "mismatch de dimensiones" que la consigna marca como error típico solo puede
ocurrir si el número está escrito dos veces y las copias divergen. Además,
`crear_indice_si_falta()` compara contra el índice existente y aborta antes de generar el
primer embedding — el error aparece en el segundo 0, no a mitad de una ingesta ya facturada.

`gemini-embedding-001` produce 3072 dimensiones y acepta `output_dimensionality`. Recortar a
1536 no es truncar a ciegas: el modelo se entrena con Matryoshka Representation Learning,
que concentra la información más significativa en las primeras componentes.

Al recortar por debajo de 3072, Gemini devuelve los vectores **sin normalizar** (norma
medida ~0.69). Con métrica coseno es indistinto, porque el coseno divide por las normas;
si alguien cambiara el índice a `dotproduct`, la magnitud entraría directo en el score y
el ranking quedaría sesgado hacia los vectores más largos.

### Dos lotes, dos límites

La clase fija el punto dulce del batch upsert en 100-200 vectores. Pero en una ingesta hay
**dos** lotes, y los limita un servicio distinto cada uno:

| Lote | Lo acota | Tamaño |
|---|---|---|
| Embedding | Gemini, por tokens por minuto | 15 fragmentos (~8.000 tokens) |
| Upsert | Pinecone, por tamaño de request | 100 vectores |

Este proyecto lo descubrió chocándose: el primer intento embebía y subía de a 100 con
`PineconeVectorStore.add_documents()`, y Gemini devolvía `429 RESOURCE_EXHAUSTED` en el
primer lote — 100 fragmentos de este corpus son ~50.000 tokens de golpe. Midiendo contra la
API, ~8.000 tokens pasan y ~16.000 ya fallan.

Por eso la ingesta corre en dos fases explícitas (`embeber()` y `subir()` en
`rag/ingesta.py`) en lugar de delegar ambas en el vectorstore, que las hace juntas y
deja el tamaño del lote de embedding fuera de nuestro control. Entre lotes de embedding hay
una pausa de 20 segundos para no volver a agotar la cuota, y el backoff de los reintentos
arranca en 65 segundos: la cuota de Gemini se repone por ventana de un minuto, así que un
backoff que arranca en 4 segundos agota los tres intentos dentro de la misma ventana y falla
igual, solo que más tarde.

La contrapartida es que el texto original hay que escribirlo a mano en `metadata["text"]`
en vez de dejar que lo haga `PineconeVectorStore`. No es una pérdida: la consigna pide
guardar el contenido en la metadata, y así queda explícito en el pipeline en lugar de ser un
efecto secundario del vectorstore. El lado de la recuperación sigue usando
`PineconeVectorStore` y lee de la misma clave — la consigna admite las dos vías, y acá cada
una se usa donde encaja.

### `task_type`: el documento y la pregunta no se embeben igual

Gemini pide declarar para qué se usa cada vector. La ingesta usa `RETRIEVAL_DOCUMENT` y la
consulta `RETRIEVAL_QUERY`: el modelo proyecta distinto un pasaje que se va a indexar que
una pregunta que lo busca. Por eso `obtener_vectorstore()` se cachea **por `task_type`**.

### El corpus de BM25 sale de Pinecone, no del disco

BM25 no es un servicio: es un índice invertido en memoria que necesita el texto de todos
los fragmentos. Lo obvio sería releer `data/` y volver a fragmentar — y ahí aparecen dos
corpus, el indexado en la nube y el del disco de quien corre el script. Si alguien edita
`data/` sin reingestar, o cambia `CHUNK_SIZE`, el ensemble fusiona rankings de universos
distintos. No falla: devuelve métricas que no significan nada.

`rag/recuperacion.py` pagina el namespace y reconstruye los fragmentos desde
`metadata["text"]` — que es exactamente para lo que la consigna pide guardar el texto ahí.
Una sola fuente de verdad. `--corpus local` queda como salida de emergencia sin red.

### El tokenizador de BM25

`BM25Retriever` tokeniza por defecto con `str.split()`. Sobre esta consulta:

```
"¿Cómo uso TypeAdapter?".split()  ->  ["¿Cómo", "uso", "TypeAdapter?"]
```

`TypeAdapter?` con el signo pegado no es el mismo término que el `TypeAdapter` del
documento, así que la coincidencia exacta —el único aporte real de BM25 frente al
vectorial— no ocurre. `rag/recuperacion.py` normaliza ambos lados igual:
minúsculas, sin acentos, cortando por todo lo que no sea letra, dígito o guion bajo. El
guion bajo se conserva porque en Python es parte del identificador: `model_validator` es un
término, no dos.

### RRF, no suma de scores

`EnsembleRetriever` aplica Reciprocal Rank Fusion: puntúa cada documento por
`1/(60 + posición)` en cada ranking y suma esas contribuciones ponderadas. Fusiona
**posiciones, no puntajes**, que es la advertencia explícita de la clase: un score de BM25
puede valer 14.7 y un coseno 0.83, y sumarlos deja que la escala arbitraria de BM25 domine
el resultado.

El ensemble devuelve la unión de ambos rankings, hasta 2k documentos.
`RAGSystem.recuperar()` recorta a `k`: sin ese corte, "top-5" sería un top-10 disfrazado y
la Precision@5 estaría midiendo otra cosa.

### Metadatos: un esquema, no campos sueltos

`rag/ingesta.py` es el único lugar donde se construye la metadata, y `CAMPOS`
documenta el contrato. Es la defensa contra el **schema drift** de la clase: si un proceso
escribe `categoria` y otro `category`, ningún filtro falla — devuelven cero resultados y el
sistema parece andar. La categoría (`modelado`, `validacion`, `serializacion`, `tipos`,
`configuracion`) habilita el hard filter: `--categoria validacion` acota el espacio de
búsqueda antes de comparar similitud.

Los nueve campos: `text`, `source`, `titulo`, `categoria`, `doc_type`, `chunk_index`,
`total_chunks`, `n_caracteres` y `url`. La consigna menciona *página* entre los metadatos
esperados; este corpus es Markdown y no tiene paginación, así que el análogo posicional son
`chunk_index` / `total_chunks`, que ubican cada fragmento dentro de su documento. `url`
cumple la otra mitad de esa función: permite citar la fuente exacta y verificable.

`titulo` estuvo mal calculado hasta que se detectó auditando la metadata ingestada. MkDocs
saca el título de la navegación, no del cuerpo, y 14 de los 16 documentos no tienen ningún
H1: el campo caía al nombre de archivo y quedaba duplicando `source` — peso muerto en cada
uno de los 162 vectores. Peor todavía, la búsqueda del H1 recorría todo el documento y un
comentario de Python dentro de un bloque de código también empieza con `# `, así que
`models.md` terminó titulado *"normal copy gives the same object reference for bar:"*. Ahora
se mira solo la primera línea con contenido, con respaldo derivado del nombre de archivo
(`json_schema.md` → *JSON Schema*), y hay dos pruebas que lo cubren.

### IDs determinísticos

El id de cada vector es `archivo.md::n`, no un UUID. Con UUIDs, reingestar el mismo corpus
duplica cada fragmento y el top-5 se llena de copias del mismo texto: la precisión medida
sube sin que el sistema haya mejorado en nada. Con este esquema, reingestar sobrescribe —
que es lo que la palabra *upsert* promete.

### Namespace

Todo el corpus vive en `pydantic-concepts-v1`, configurable por entorno. El sufijo de
versión es deliberado: cambiar el chunking implica un corpus distinto, y levantarlo en un
namespace nuevo permite comparar ambos sin destruir el anterior.

## Evaluación

### Cómo está armado el golden set

10 preguntas, cada una con los documentos que **realmente** contienen la respuesta,
verificados a mano contra el corpus. Están clasificadas por tipo, y esa clasificación es el
experimento:

| tipo | n | qué mide |
|---|---|---|
| `lexica` | 5 | Nombra un identificador exacto y poco ambiguo (`AliasGenerator`: 8 de 8 apariciones en `alias.md`). Debería ganar BM25. |
| `semantica` | 3 | Describe el problema sin nombrar ninguna API. Debería ganar el vectorial. |
| `mixta` | 2 | Nombra un identificador repartido en varios documentos (`TypeAdapter`: 97 apariciones en 8 archivos). Ninguno debería resolverla solo. |

Dos casos declaran **más de un** documento relevante. El notebook de la clase señala que
con un único `documento_id_esperado` el Recall@5 solo puede dar 0 o 1; con dos documentos
relevantes puede dar 0.5, y la métrica distingue "recuperó la mitad" de "no recuperó nada".
`rag/evaluacion.py` acepta igual el formato de un solo documento de la consigna.

### Cómo se calculan las métricas

- **Precision@5** — fracción de las 5 posiciones ocupada por fragmentos de un documento
  esperado. Se divide por `k`, no por la cantidad devuelta: recuperar de menos no puede ser
  una ventaja.
- **Recall@5** — proporción de los documentos relevantes que aparece en el top-5, **a nivel
  documento**: cinco fragmentos del mismo archivo cuentan como un documento, no como cinco.
- **hit-rate@5** — 1 si al menos un documento relevante entró. Es el "¿está o no está?" de
  la consigna.
- **MRR@5** — inversa de la posición del primer acierto. Precision y Recall son ciegas al
  orden: el documento correcto en la posición 1 o en la 5 da la misma Precision@5, y no es
  lo mismo.

**El techo de Precision@5.** La definición de la consigna tiene un límite estructural: si el
documento correcto solo produjo 3 fragmentos, la Precision@5 máxima es 3/5 = 0.6 aunque el
sistema funcione perfecto. `techo_precision()` lo calcula y el reporte lo imprime en la
columna `techo`, para leer el valor medido contra lo máximo alcanzable en vez de contra un
1.0 imposible.

### Resultados

Medidos sobre el índice real: 162 vectores, 10 preguntas, `top_k=5`. Reproducibles con
`python evaluate.py`.

| configuración | P@5 | techo P@5 | R@5 | hit@5 | MRR@5 |
|---|---|---|---|---|---|
| BM25 (léxico) | 0.50 | 0.86 | 0.75 | 0.80 | 0.65 |
| **Pinecone (vectorial)** | **0.74** | 0.86 | **1.00** | **1.00** | **0.95** |
| Híbrido RRF (0.5 / 0.5) | 0.62 | 0.86 | 1.00 | 1.00 | 0.80 |

Recall@5 desglosado por tipo de pregunta:

| configuración | léxicas (5) | mixtas (2) | semánticas (3) |
|---|---|---|---|
| BM25 (léxico) | 1.00 | 1.00 | **0.17** |
| Pinecone (vectorial) | 1.00 | 1.00 | 1.00 |
| Híbrido RRF (0.5 / 0.5) | 1.00 | 1.00 | 1.00 |

Barrido de pesos del ensemble:

| pesos (bm25 / vectorial) | P@5 | R@5 | hit@5 | MRR@5 |
|---|---|---|---|---|
| 0.7 / 0.3 | 0.50 | 0.75 | 0.80 | 0.75 |
| 0.5 / 0.5 | 0.62 | 1.00 | 1.00 | 0.80 |
| 0.3 / 0.7 | 0.74 | 1.00 | 1.00 | 0.95 |

### Qué dicen estos números

**1. BM25 falla exactamente donde se predijo, y falla feo.** Recall 0.17 en las preguntas
semánticas: de las tres, dos no recuperaron el documento correcto en ninguna posición. A
"¿qué recomendaciones hay para que la validación sea más rápida?" devolvió cinco fragmentos
de `validators.md` y ninguno de `performance.md` — la palabra "validación" está en la
consulta y en el documento equivocado, y BM25 no tiene forma de saber que el tema es otro.

**2. El vectorial es mucho más fuerte de lo que anticipaba el diseño del benchmark.** Gana
incluso en las preguntas *léxicas*, donde BM25 supuestamente tenía la ventaja: recupera
`AliasGenerator`, `json_schema_extra` y `field_serializer` con recall 1.00 y MRR 1.00. Dos
razones. Una, `gemini-embedding-001` a 1536 dimensiones representa bien los identificadores
de código, que no son "ruido" para él. Dos, es multilingüe: las preguntas están en español y
el corpus en inglés, un desajuste que a BM25 lo deja sin nada que hacer salvo el
identificador suelto, y que al vectorial no lo afecta.

**3. El híbrido no mejora al vectorial: lo empata en recall y lo empeora en precisión.**
Recupera todo lo que BM25 pierde (semánticas de 0.17 a 1.00), pero contra el vectorial solo
pierde 12 puntos de precisión (0.74 → 0.62) y 15 de MRR (0.95 → 0.80). El mecanismo se ve en
el detalle: en la pregunta de coerción, el vectorial devuelve `strict_mode.md` en las
posiciones 1, 2 y 3; el híbrido lo baja a las posiciones 2 y 4 porque RRF le da a BM25 —que
ahí no acertó ninguno— el mismo voto que al vectorial, y sube documentos que solo BM25
rankeó alto.

**4. El barrido es monótono y no tiene punto óptimo interior.** Cada punto de peso que se le
saca a BM25 mejora *todas* las métricas, y en 0.3/0.7 el híbrido converge exactamente en los
números del vectorial puro (0.74 / 1.00 / 0.95). No hay una mezcla que supere a ninguna de
las dos partes: la mejor configuración del ensemble es la que más se parece a no tener
ensemble.

**Aclaración importante, porque el número se puede leer mal:** que el híbrido no le gane al
vectorial **no** indica que el ensemble esté mal implementado. La prueba de que la fusión
funciona es la fila de las semánticas: BM25 solo tiene 0.17 de recall ahí, y al fusionarlo
con el vectorial el híbrido sube a 1.00. El ensemble está recuperando exactamente lo que
tiene que recuperar. Lo que el experimento muestra es una propiedad *de este corpus con este
modelo de embeddings*, no un defecto del código.

**5. Conclusión honesta: acá el híbrido es un seguro, no una mejora.** Cuesta ~12 puntos de
precisión para cubrir un modo de falla que este recuperador vectorial no tiene sobre este
corpus. Eso no invalida la técnica, delimita cuándo paga: con un modelo de embeddings más
débil o monolingüe en inglés, o con identificadores sin ninguna carga semántica —códigos de
error tipo `ERR-4021`, SKUs, números de parte— donde BM25 es literalmente el único de los dos
que puede encontrar el término. Nada de eso se puede afirmar sin medirlo, que es justamente
por qué `evaluate.py` compara tres configuraciones en vez de reportar solo el híbrido.

**Los pesos por defecto quedan en 0.5/0.5** aunque el barrido favorezca 0.3/0.7. Mover el
default para ganar en un benchmark de 10 preguntas, donde cada una vale 10 puntos de recall,
es ajustar el sistema al examen: la diferencia no es distinguible del ruido con esta muestra.
El barrido queda documentado y a un flag de distancia (`--pesos 0.3 0.7`).

**Sobre la columna `techo`.** El máximo de Precision@5 alcanzable promedia 0.86, no 1.00,
porque tres documentos del corpus generan menos de 5 fragmentos: `type_adapter.md` produce 2
(techo 0.40), y `strict_mode.md` y `performance.md` producen 3 (techo 0.60). El 0.74 del
vectorial es **86% del máximo posible**, no 74% de un ideal inalcanzable.

## Errores de la consigna, y qué los evita acá

| Error | Mitigación |
|---|---|
| Mismatch de dimensiones | `DIMENSION` en un solo lugar + `verificar_compatibilidad()` antes del primer embedding |
| Ignorar el namespace | Namespace obligatorio y versionado, configurable por `.env`; filtros por `categoria` dentro de él |
| Subestimar el chunking | Tres configuraciones medidas; se eligió la de mediana 528 tokens y cero fragmentos minúsculos |
| Sumar scores heterogéneos | RRF sobre posiciones (`EnsembleRetriever`), nunca suma de score BM25 + coseno |

## Limitaciones conocidas

- **Sin capa de generación.** La consigna pide un módulo de *recuperación*: el sistema
  devuelve fragmentos, no respuestas. La generación fue el alcance de la pre-entrega 3.
- **BM25 vive en memoria.** Se reconstruye en cada proceso leyendo el índice completo. Con
  162 fragmentos tarda un segundo; con cientos de miles habría que persistir el índice
  invertido o mover la búsqueda léxica a un servicio (los índices *sparse* de Pinecone, o
  Elasticsearch).
- **El golden set es chico.** 10 preguntas alcanzan para ver la diferencia entre modos, no
  para medir mejoras de pocos puntos: cada pregunta pesa 10 puntos de Recall.
- **La relevancia se juzga a nivel documento.** Un fragmento del archivo correcto que no
  contenga la respuesta cuenta como acierto. Etiquetar fragmento por fragmento daría una
  Precision más honesta, a costa de rehacer el golden set con cada cambio de chunking.
- **Cuota gratuita de Gemini.** Es el factor que fija la duración de la ingesta: 11 lotes de
  embedding con 20 segundos de pausa entre cada uno, ~4 minutos para 162 fragmentos. Con
  cuota paga, los mismos lotes irían seguidos y bajaría a segundos. Las consultas gastan una
  llamada por pregunta, muy por debajo del límite.
