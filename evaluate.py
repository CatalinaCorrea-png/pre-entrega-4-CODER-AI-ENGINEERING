"""Evalúa el recuperador contra el golden set e imprime el reporte en consola.

    python evaluate.py                 # los tres modos, k=5
    python evaluate.py --sweep         # + barrido de pesos del ensemble
    python evaluate.py --corpus local  # BM25 sin tocar la red

Mide las tres configuraciones y no solo el híbrido: sin comparar contra cada mitad por
separado no hay forma de saber si el ensemble aporta o si arrastra hacia abajo al
recuperador que sí funciona.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.config import correr, INDEX_NAME, NAMESPACE
from rag.recuperacion import obtener_corpus, PESOS_POR_DEFECTO, RAGSystem
from rag.evaluacion import cargar, evaluar_caso, Caso, Reporte

MODOS_POR_DEFECTO = ("bm25", "vectorial", "hibrido")

ETIQUETAS = {
    "bm25": "BM25 (léxico)",
    "vectorial": "Pinecone (vectorial)",
    "hibrido": "Híbrido (RRF)",
}

# Barrido de `weights` del EnsembleRetriever: cuánto pesa cada recuperador en la fusión.
PESOS_DEL_BARRIDO = ((0.7, 0.3), (0.5, 0.5), (0.3, 0.7))

# evalcuacion por caso
def evaluar(
    sistema: RAGSystem,
    casos: list[Caso],
    k: int,
    fragmentos_por_documento: dict[str, int],
    etiqueta: str,
) -> Reporte:
    resultados = [
        evaluar_caso(caso, sistema.fuentes(caso.pregunta), k, fragmentos_por_documento)
        for caso in casos
    ]
    return Reporte(etiqueta, resultados)


def tabla(titulo: str, columnas: list[str], filas: list[list[str]]) -> None:
    """Imprime una tabla alineada: primera columna a la izquierda, el resto a la derecha."""
    anchos = [max(len(col), *(len(f[i]) for f in filas)) for i, col in enumerate(columnas)]

    def armar(celdas):
        return "  ".join(
            celda.ljust(ancho) if i == 0 else celda.rjust(ancho)
            for i, (celda, ancho) in enumerate(zip(celdas, anchos))
        )

    print()
    print(titulo)
    print(armar(columnas))
    print("-" * (sum(anchos) + 2 * (len(anchos) - 1)))
    for fila in filas:
        print(armar(fila))


def imprimir_detalle(reporte: Reporte, k: int) -> None:
    """Una pregunta por bloque: qué se esperaba, qué volvió y con qué métricas."""
    print()
    print(f"DETALLE - {reporte.etiqueta}")
    for r in reporte.resultados:
        marca = "ok   " if r.hit else "FALLA"
        print()
        print(f"{marca} [{r.caso.tipo}] {r.caso.pregunta}")
        print(f"      esperado    : {', '.join(sorted(r.caso.documentos_esperados))}")
        print(f"      recuperados : {', '.join(r.recuperados)}")
        print(
            f"      P@{k} {r.precision:.2f} (techo {r.techo:.2f})   "
            f"R@{k} {r.recall:.2f}   MRR {r.mrr:.2f}"
        )


def imprimir_comparativa(reportes: list[Reporte], k: int, tipos: list[str]) -> None:
    """Las tres configuraciones sobre el mismo golden set, en agregado y por tipo."""
    tabla(
        f"COMPARATIVA ({len(reportes[0].resultados)} preguntas, top-{k})",
        ["configuración", f"P@{k}", "techo", f"R@{k}", "hit", "MRR"],
        [
            [r.etiqueta, f"{r.precision:.2f}", f"{r.techo:.2f}",
             f"{r.recall:.2f}", f"{r.hit_rate:.2f}", f"{r.mrr:.2f}"]
            for r in reportes
        ],
    )
    tabla(
        f"RECALL@{k} POR TIPO DE PREGUNTA",
        ["configuración", *(f"{t} ({len(reportes[0].por_tipo(t).resultados)})" for t in tipos)],
        [[r.etiqueta, *(f"{r.por_tipo(t).recall:.2f}" for t in tipos)] for r in reportes],
    )


def imprimir_barrido(reportes: list[Reporte], k: int) -> None:
    """El efecto de mover los pesos del ensemble, con todo lo demás igual."""
    tabla(
        "BARRIDO DE PESOS DEL ENSEMBLE",
        ["pesos (bm25 / vectorial)", f"P@{k}", f"R@{k}", "hit", "MRR"],
        [
            [r.etiqueta, f"{r.precision:.2f}", f"{r.recall:.2f}",
             f"{r.hit_rate:.2f}", f"{r.mrr:.2f}"]
            for r in reportes
        ],
    )


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=5, help="cuántos documentos recuperar (default: 5)")
    parser.add_argument("--modos", nargs="+", default=list(MODOS_POR_DEFECTO), choices=list(MODOS_POR_DEFECTO))
    parser.add_argument("--pesos", nargs=2, type=float, default=list(PESOS_POR_DEFECTO), metavar=("BM25", "VECTORIAL"))
    parser.add_argument("--corpus", choices=("pinecone", "local"), default="pinecone",
                        help="de dónde sale el corpus léxico de BM25 (default: pinecone)")
    parser.add_argument("--namespace", default=NAMESPACE)
    parser.add_argument("--sweep", action="store_true", help="barre los pesos del ensemble")
    parser.add_argument("--sin-detalle", action="store_true", help="solo la tabla comparativa")
    return parser.parse_args()


def main() -> int:
    args = parsear_argumentos()

    casos = cargar()
    tipos = sorted({caso.tipo for caso in casos})

    # El corpus se baja una sola vez y se comparte entre todas las configuraciones: son los
    # mismos fragmentos, y bajarlos por cada modo sería tiempo de red tirado. Se baja siempre,
    # incluso evaluando solo el modo vectorial que no usa BM25, porque de acá sale también el
    # conteo de fragmentos por documento y sin él la columna `techo` daría 0.00 en todo.
    corpus = obtener_corpus(args.corpus, args.namespace)
    fragmentos_por_documento = Counter(d.metadata.get("source", "") for d in corpus)

    print(
        f"Evaluación de {INDEX_NAME}/{args.namespace}: {len(casos)} preguntas, "
        f"top-{args.k}, corpus léxico desde {args.corpus}, "
        f"{len(corpus)} fragmentos de {len(fragmentos_por_documento)} documentos"
    )

    reportes = []
    for modo in args.modos:
        sistema = RAGSystem(
            k=args.k, modo=modo, pesos=tuple(args.pesos),
            namespace=args.namespace, corpus=corpus or None, origen_corpus=args.corpus,
        )
        etiqueta = ETIQUETAS[modo]
        if modo == "hibrido":
            etiqueta += f" {tuple(args.pesos)}"
        reportes.append(evaluar(sistema, casos, args.k, fragmentos_por_documento, etiqueta))

    if not args.sin_detalle:
        for reporte in reportes:
            imprimir_detalle(reporte, args.k)

    imprimir_comparativa(reportes, args.k, tipos)

    if args.sweep:
        barrido = [
            evaluar(
                RAGSystem(k=args.k, modo="hibrido", pesos=pesos, namespace=args.namespace, corpus=corpus),
                casos, args.k, fragmentos_por_documento, f"bm25={pesos[0]}  vect={pesos[1]}",
            )
            for pesos in PESOS_DEL_BARRIDO
        ]
        imprimir_barrido(barrido, args.k)

    mejor = max(reportes, key=lambda r: (r.recall, r.mrr))
    print()
    print(f"Mejor Recall@{args.k}: {mejor.etiqueta} {mejor.recall:.2f} (MRR {mejor.mrr:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(correr(main))
