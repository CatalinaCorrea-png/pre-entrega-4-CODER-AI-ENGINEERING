"""Consulta el recuperador desde la terminal.

    python scripts/consultar.py                                  # modo interactivo
    python scripts/consultar.py "como uso TypeAdapter"           # una consulta y sale
    python scripts/consultar.py --modo bm25 "AliasGenerator"     # solo el lexico
    python scripts/consultar.py --categoria validacion "alias"   # con filtro por metadata

Sirve para ver a ojo lo que `evaluate.py` mide en agregado: que fragmentos vuelven, de que
documento y en que orden. Correr la misma consulta con `--modo bm25` y con
`--modo vectorial` muestra el aporte de cada mitad mejor que cualquier explicacion.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import NAMESPACE
from rag.consola import correr
from rag.recuperacion import RAGSystem
from rag.recuperacion.rag_system import MODOS

RECORTE = 220
SALIDAS = ("salir", "exit", "quit", "")


def mostrar(sistema: RAGSystem, consulta: str) -> None:
    resultados = sistema.obtener_top_k(consulta)
    if not resultados:
        print("⚠️  No se recuperó ningún fragmento.\n")
        return

    print(f"\n📎 Top {len(resultados)} — {sistema!r}")
    for posicion, resultado in enumerate(resultados, start=1):
        texto = " ".join(resultado["contenido"].split())
        recorte = texto[:RECORTE] + ("..." if len(texto) > RECORTE else "")
        print(f"\n  {posicion}. [{resultado['fuente']} · {resultado['categoria']} · frag {resultado['chunk_index']}]")
        print(f"     {recorte}")
        print(f"     {resultado['url']}")
    print("-" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("consulta", nargs="*", help="si se omite, entra en modo interactivo")
    parser.add_argument("--modo", choices=list(MODOS), default="hibrido")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--namespace", default=NAMESPACE)
    parser.add_argument("--categoria", help="filtro por metadata: modelado, validacion, serializacion, tipos, configuracion")
    parser.add_argument("--corpus", choices=("pinecone", "local"), default="pinecone")
    args = parser.parse_args()

    filtro = {"categoria": {"$eq": args.categoria}} if args.categoria else None
    sistema = RAGSystem(
        k=args.k,
        modo=args.modo,
        namespace=args.namespace,
        origen_corpus=args.corpus,
        filtro=filtro,
    )

    if args.consulta:
        mostrar(sistema, " ".join(args.consulta))
        return 0

    print(f"\n💬 Recuperador {args.modo} — escribí tu consulta (o 'salir' para terminar)")
    while True:
        try:
            consulta = input("\n🧑 Vos: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Listo.")
            return 0
        if consulta.lower() in SALIDAS:
            print("👋 Listo.")
            return 0
        mostrar(sistema, consulta)


if __name__ == "__main__":
    raise SystemExit(correr(main))
