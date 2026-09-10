"""Consulta el recuperador desde la terminal.

    python scripts/consultar.py                                  # modo interactivo
    python scripts/consultar.py "como uso TypeAdapter"           # una consulta y sale
    python scripts/consultar.py --modo bm25 "AliasGenerator"     # solo el léxico
    python scripts/consultar.py --categoria validacion "alias"   # filtro por metadata

Sirve para ver a ojo lo que `evaluate.py` mide en agregado. Correr la misma consulta con
`--modo bm25` y con `--modo vectorial` muestra el aporte de cada mitad.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import correr, NAMESPACE
from rag.recuperacion import MODOS, RAGSystem

RECORTE = 200
SALIDAS = ("salir", "exit", "quit", "")


def mostrar(sistema: RAGSystem, consulta: str) -> None:
    resultados = sistema.obtener_top_k(consulta)
    if not resultados:
        print("Sin resultados.")
        return
    for posicion, resultado in enumerate(resultados, start=1):
        texto = " ".join(resultado["contenido"].split())[:RECORTE]
        print(f"{posicion}. [{resultado['fuente']} · {resultado['categoria']}] {texto}...")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("consulta", nargs="*", help="si se omite, entra en modo interactivo")
    parser.add_argument("--modo", choices=list(MODOS), default="hibrido")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--namespace", default=NAMESPACE)
    parser.add_argument("--categoria", help="modelado, validacion, serializacion, tipos, configuracion")
    parser.add_argument("--corpus", choices=("pinecone", "local"), default="pinecone")
    args = parser.parse_args()

    sistema = RAGSystem(
        k=args.k,
        modo=args.modo,
        namespace=args.namespace,
        origen_corpus=args.corpus,
        filtro={"categoria": {"$eq": args.categoria}} if args.categoria else None,
    )

    if args.consulta:
        mostrar(sistema, " ".join(args.consulta))
        return 0

    print(f"Recuperador {args.modo}. Escribí tu consulta, o 'salir' para terminar.")
    while True:
        try:
            consulta = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if consulta.lower() in SALIDAS:
            return 0
        mostrar(sistema, consulta)


if __name__ == "__main__":
    raise SystemExit(correr(main))
