"""Crea el índice Serverless de Pinecone si no existe, y lo valida si ya existe.

    python scripts/setup_indice.py

Es idempotente: correrlo diez veces deja el mismo estado que correrlo una. Y es el lugar
donde se detecta el "mismatch de dimensiones" antes de que cueste algo — validar acá
convierte un error a mitad de la ingesta (con parte del corpus ya subido y facturado) en
un mensaje que dice qué hacer, antes de generar el primer embedding.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import INDEX_NAME, NAMESPACE, PINECONE_CLOUD, PINECONE_REGION
from rag.consola import correr
from rag.embeddings import DIMENSION, METRICA, MODELO
from rag.infra.pinecone_setup import crear_indice_si_falta, estadisticas


def main() -> int:
    print("\n🔧 Setup del índice de Pinecone")
    print(f"   índice     : {INDEX_NAME}")
    print(f"   modelo     : {MODELO}")
    print(f"   dimensión  : {DIMENSION}   (fuente de verdad única: rag/embeddings.py)")
    print(f"   métrica    : {METRICA}")
    print(f"   spec       : serverless · {PINECONE_CLOUD} · {PINECONE_REGION}\n")

    info = crear_indice_si_falta()
    print("🆕 Índice creado." if info["creado"] else "♻️  El índice ya existía y es compatible.")
    print(f"   host: {info['host']}")

    stats = estadisticas()
    print(f"\n📦 Estado actual: {stats['vectores_totales']} vectores")
    for nombre, cantidad in sorted(stats["namespaces"].items()):
        marca = " ←" if nombre == NAMESPACE else ""
        print(f"   namespace '{nombre}': {cantidad}{marca}")
    if not stats["namespaces"]:
        print("   (sin namespaces todavía — corré: python scripts/ingestar.py)")

    return 0


if __name__ == "__main__":
    raise SystemExit(correr(main))
