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

## Evaluación

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

**BM25 falla donde se predijo.** Recall 0.17 en las semánticas: dos de las tres preguntas no
recuperaron el documento correcto en ninguna posición. A *"¿qué recomendaciones hay para que
la validación sea más rápida?"* devolvió cinco fragmentos de `validators.md` y ninguno de
`performance.md` — la palabra "validación" está en el documento equivocado y BM25 no puede
saber que el tema es otro.

**El vectorial gana incluso en las preguntas léxicas**, donde BM25 tenía que tener la
ventaja. Dos razones: `gemini-embedding-001` representa bien los identificadores de código, y
es multilingüe — las preguntas están en español y el corpus en inglés, un desajuste que a
BM25 lo deja sin nada salvo el identificador suelto.

**El híbrido empata en recall y pierde precisión:** −12 puntos de P@5 (0.74 → 0.62) y −15 de
MRR (0.95 → 0.80) contra el vectorial solo. RRF le da a BM25 el mismo voto aunque ahí no haya
acertado, y eso empuja hacia abajo lo que el vectorial ya tenía primero. El barrido lo
confirma: es monótono a favor del vectorial, y en 0.3/0.7 el híbrido converge exactamente en
los números del vectorial puro.

**Conclusión: acá el híbrido es un seguro, no una mejora.** Paga ~12 puntos de precisión para
cubrir un modo de falla que este vectorial no tiene sobre este corpus. Dónde sí pagaría: con
un modelo más débil o monolingüe, o con identificadores sin carga semántica (`ERR-4021`,
SKUs) donde BM25 es el único que puede encontrar el término. Nada de eso se afirma sin
medirlo — por eso `evaluate.py` compara tres configuraciones y no solo el híbrido.

**La columna `techo`** promedia 0.86 y no 1.00 porque tres documentos generan menos de 5
fragmentos (`type_adapter.md` produce 2; `strict_mode.md` y `performance.md`, 3). El 0.74 del
vectorial es **86% del máximo posible**, no 74% de un ideal inalcanzable.

## Errores evitados

| Error | Mitigación |
|---|---|
| Mismatch de dimensiones | `DIMENSION` en un solo lugar + `verificar_compatibilidad()` antes del primer embedding |
| Ignorar el namespace | Namespace obligatorio y versionado, configurable por `.env`; filtros por `categoria` dentro de él |
| Subestimar el chunking | Tres configuraciones medidas; se eligió la de mediana 528 tokens y cero fragmentos minúsculos |
| Sumar scores heterogéneos | RRF sobre posiciones (`EnsembleRetriever`), nunca suma de score BM25 + coseno |
