"""Descarga el corpus: la documentación de conceptos de Pydantic.

Por qué documentación técnica real y no un corpus inventado: la consigna lo sugiere, pero
además es el escenario donde el recuperador híbrido se justifica. Estos documentos están
llenos de identificadores exactos —`TypeAdapter`, `model_validator`, `ConfigDict`,
`AliasGenerator`— que los embeddings confunden entre sí porque son semánticamente
parecidísimos (todos hablan de "validar modelos"), y que BM25 distingue sin esfuerzo
porque son literales distintos. Con un corpus de textos genéricos, BM25 no aportaría nada
y el ensemble sería decorativo.

La descarga está clavada a un tag (`REF`), no a la rama principal: el corpus tiene que ser
el mismo hoy y dentro de seis meses, o las métricas del README dejan de ser reproducibles.

Pydantic se distribuye bajo licencia MIT. `data/FUENTE.txt` deja registrada la atribución.
"""

import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import DATA_DIR
from rag.consola import correr
from rag.ingesta.metadata import CATEGORIAS

REPO = "pydantic/pydantic"
REF = "v2.13.5"
RUTA_EN_REPO = "docs/concepts"
URL_CRUDA = "https://raw.githubusercontent.com/{repo}/{ref}/{ruta}/{archivo}"

# Los 16 documentos son los de `CATEGORIAS`: un solo lugar define qué entra al corpus y a
# qué categoría pertenece cada archivo, así no puede haber un documento indexado sin
# categoría ni una categoría que apunte a un archivo inexistente.
ARCHIVOS = tuple(CATEGORIAS)

# Se guarda como .txt y no como .md a propósito: el loader levanta `*.md`, y un
# archivo de atribución indexado como si fuera documentación aparecería en los
# resultados de búsqueda.
ATRIBUCION = """# Fuente del corpus

Los archivos `.md` de esta carpeta son documentación de **Pydantic**, descargados sin
modificar desde el repositorio oficial:

- Repositorio: https://github.com/{repo}
- Ruta: `{ruta}/`
- Tag: `{ref}` (clavado para que el corpus sea reproducible)
- Documentación publicada: https://docs.pydantic.dev/latest/concepts/
- Licencia: MIT

Para regenerarlos: `python scripts/descargar_dataset.py`
"""


def descargar(archivo: str, destino: Path) -> int:
    url = URL_CRUDA.format(repo=REPO, ref=REF, ruta=RUTA_EN_REPO, archivo=archivo)
    with urllib.request.urlopen(url, timeout=30) as respuesta:
        contenido = respuesta.read().decode("utf-8")
    destino.write_text(contenido, encoding="utf-8")
    return len(contenido)


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📥 Descargando {len(ARCHIVOS)} documentos de {REPO}@{REF}\n")

    total = 0
    fallidos = []
    for archivo in ARCHIVOS:
        try:
            tamano = descargar(archivo, DATA_DIR / archivo)
        except (urllib.error.URLError, urllib.error.HTTPError) as error:
            fallidos.append(archivo)
            print(f"   ✗ {archivo:<28} {error}")
            continue
        total += tamano
        print(f"   ✓ {archivo:<28} {tamano:>7,} caracteres   [{CATEGORIAS[archivo]}]")

    (DATA_DIR / "FUENTE.txt").write_text(
        ATRIBUCION.format(repo=REPO, ruta=RUTA_EN_REPO, ref=REF), encoding="utf-8"
    )

    print(f"\n📦 {len(ARCHIVOS) - len(fallidos)}/{len(ARCHIVOS)} documentos en {DATA_DIR}")
    print(f"   {total:,} caracteres  (~{total // 4:,} tokens estimados)")

    if fallidos:
        print(f"\n⚠️  No se pudieron descargar: {', '.join(fallidos)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(correr(main))
