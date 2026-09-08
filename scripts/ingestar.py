"""Pipeline de ingesta completo: data/ → limpieza → chunking → metadata → Pinecone.

    python scripts/ingestar.py
    python scripts/ingestar.py --limpiar     # vacía el namespace antes de subir

Los IDs de los vectores son determinísticos (`validators.md::3`), así que correr esto dos
veces sobre el mismo corpus actualiza en vez de duplicar. `--limpiar` hace falta solo
cuando cambia el chunking: si el corpus nuevo produce menos fragmentos que el anterior,
los sobrantes quedan huérfanos en el índice y siguen apareciendo en las búsquedas.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import INDEX_NAME, NAMESPACE
from rag.consola import correr
from rag.infra.pinecone_setup import crear_indice_si_falta, estadisticas
from rag.ingesta.carga import cargar_documentos
from rag.ingesta.chunking import CHUNK_OVERLAP, CHUNK_SIZE, fragmentar
from rag.ingesta.metadata import aplicar_metadata
from rag.ingesta.upsert import (
    LOTE_EMBEDDING,
    LOTE_UPSERT,
    borrar_namespace,
    embeber,
    subir,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--namespace", default=NAMESPACE)
    parser.add_argument("--limpiar", action="store_true", help="vacía el namespace antes de subir")
    args = parser.parse_args()

    print(f"\n📥 Ingesta → índice '{INDEX_NAME}', namespace '{args.namespace}'\n")

    # Verificar el índice antes de generar un solo embedding: si la dimensión no coincide,
    # el upsert va a fallar igual, pero para entonces ya se pagaron los embeddings.
    info = crear_indice_si_falta()
    print("🆕 Índice creado." if info["creado"] else "♻️  Índice existente, compatible.")

    documentos = cargar_documentos()
    print(f"📄 {len(documentos)} documentos cargados y limpiados")

    chunks = aplicar_metadata(fragmentar(documentos))
    print(f"✂️  {len(chunks)} fragmentos (chunk_size={CHUNK_SIZE} tokens, overlap={CHUNK_OVERLAP})")

    por_categoria = Counter(c.metadata["categoria"] for c in chunks)
    print("   por categoría: " + ", ".join(f"{c}={n}" for c, n in sorted(por_categoria.items())))
    print(f"   metadata de ejemplo: {chunks[0].metadata}")

    if args.limpiar:
        borrar_namespace(args.namespace)
        print(f"\n🧹 Namespace '{args.namespace}' vaciado")

    # Dos fases con lotes distintos: el de embedding lo acota la cuota por minuto de
    # Gemini, el de upsert lo acota Pinecone. Ver rag/ingesta/upsert.py.
    print(f"\n🧮 Generando embeddings en lotes de {LOTE_EMBEDDING} (cuota por minuto)...")
    vectores = embeber(
        chunks,
        al_terminar_lote=lambda hechos, total: print(f"   {hechos}/{total} embeddings"),
    )

    print(f"\n☁️  Upsert en lotes de {LOTE_UPSERT}...")
    subir(
        chunks,
        vectores,
        namespace=args.namespace,
        al_terminar_lote=lambda subidos, total: print(f"   {subidos}/{total} vectores"),
    )

    stats = estadisticas()
    print(f"\n📦 Índice: {stats['vectores_totales']} vectores, dimensión {stats['dimension']}")
    for nombre, cantidad in sorted(stats["namespaces"].items()):
        print(f"   namespace '{nombre}': {cantidad}")

    print("\n✅ Listo. Siguiente paso: python evaluate.py\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(correr(main))
