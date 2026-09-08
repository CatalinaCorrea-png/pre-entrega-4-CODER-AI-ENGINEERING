"""Evalúa el recuperador contra el golden set e imprime el reporte en consola.

    python evaluate.py                      # los tres modos, k=5
    python evaluate.py --modos hibrido      # solo el híbrido
    python evaluate.py --corpus local       # BM25 sin tocar la red
    python evaluate.py --sweep              # barrido de pesos del ensemble

Por qué evalúa tres configuraciones y no solo el híbrido: El resultado es cuánto mejora 
sobre cada mitad por separado,medido sobre las mismas preguntas. 
Sin ese contrafáctico no hay forma de saber si el ensemble está aportando o si está 
arrastrando hacia abajo al recuperador que sí funciona.

El desglose por tipo de pregunta es donde se ve el mecanismo: BM25 debería ganar en las
léxicas (un identificador exacto), el vectorial en las semánticas (una reformulación sin
ninguna API nombrada), y el híbrido debería ser el único que no se hunde en ninguna de las
dos — que es exactamente para lo que existe.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluacion.golden_set import Caso, cargar
from evaluacion.metricas import Reporte, evaluar_caso
from rag.config import INDEX_NAME, NAMESPACE
from rag.consola import correr
from rag.recuperacion import RAGSystem, obtener_corpus
from rag.recuperacion.retrievers import PESOS_POR_DEFECTO

MODOS_POR_DEFECTO = ("bm25", "vectorial", "hibrido")

ETIQUETAS = {
    "bm25": "BM25 (léxico)",
    "vectorial": "Pinecone (vectorial)",
    "hibrido": "Híbrido (RRF)",
}

# Barrido de `weights` del EnsembleRetriever: cuánto pesa cada recuperador en la fusión.
PESOS_DEL_BARRIDO = ((0.7, 0.3), (0.5, 0.5), (0.3, 0.7))


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


def imprimir_detalle(reporte: Reporte, k: int) -> None:
    print(f"\n{'=' * 78}\n  DETALLE POR PREGUNTA — {reporte.etiqueta}\n{'=' * 78}")
    for resultado in reporte.resultados:
        marca = "✅" if resultado.hit else "❌"
        esperados = ", ".join(sorted(resultado.caso.documentos_esperados))
        print(f"\n{marca} [{resultado.caso.tipo}] {resultado.caso.pregunta}")
        print(f"   esperado    : {esperados}")
        print(f"   recuperados : {' | '.join(resultado.recuperados)}")
        print(
            f"   P@{k}={resultado.precision:.2f} (techo {resultado.techo:.2f})"
            f"   R@{k}={resultado.recall:.2f}"
            f"   MRR={resultado.mrr:.2f}"
        )


def imprimir_comparativa(reportes: list[Reporte], k: int, tipos: list[str]) -> None:
    print(f"\n{'=' * 78}\n  COMPARATIVA — {len(reportes[0].resultados)} preguntas, top-{k}\n{'=' * 78}")
    encabezado = f"{'configuración':<24}{'P@'+str(k):>8}{'techo':>8}{'R@'+str(k):>8}{'hit':>8}{'MRR':>8}"
    print(encabezado)
    print("-" * len(encabezado))
    for reporte in reportes:
        print(
            f"{reporte.etiqueta:<24}{reporte.precision:>8.2f}{reporte.techo:>8.2f}"
            f"{reporte.recall:>8.2f}{reporte.hit_rate:>8.2f}{reporte.mrr:>8.2f}"
        )

    print(f"\n  Recall@{k} por tipo de pregunta")
    encabezado_tipos = f"{'configuración':<24}" + "".join(f"{t:>14}" for t in tipos)
    print(encabezado_tipos)
    print("-" * len(encabezado_tipos))
    for reporte in reportes:
        celdas = ""
        for tipo in tipos:
            sub = reporte.por_tipo(tipo)
            celdas += f"{sub.recall:>10.2f} ({len(sub.resultados)})"
        print(f"{reporte.etiqueta:<24}{celdas}")


def imprimir_barrido(reportes: list[Reporte], k: int) -> None:
    print(f"\n{'=' * 78}\n  BARRIDO DE PESOS DEL ENSEMBLE (bm25, vectorial)\n{'=' * 78}")
    encabezado = f"{'pesos':<24}{'P@'+str(k):>8}{'R@'+str(k):>8}{'hit':>8}{'MRR':>8}"
    print(encabezado)
    print("-" * len(encabezado))
    for reporte in reportes:
        print(
            f"{reporte.etiqueta:<24}{reporte.precision:>8.2f}{reporte.recall:>8.2f}"
            f"{reporte.hit_rate:>8.2f}{reporte.mrr:>8.2f}"
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

    print(f"\n📊 Evaluación del recuperador — índice '{INDEX_NAME}', namespace '{args.namespace}'")
    print(f"   {len(casos)} preguntas · top-{args.k} · corpus léxico desde '{args.corpus}'")

    # El corpus se baja una sola vez y se comparte entre todas las configuraciones: son
    # los mismos fragmentos, y bajarlos por cada modo sería tiempo de red tirado.
    necesita_corpus = any(modo in ("bm25", "hibrido") for modo in args.modos) or args.sweep
    corpus = obtener_corpus(args.corpus, args.namespace) if necesita_corpus else []
    fragmentos_por_documento = Counter(d.metadata.get("source", "") for d in corpus)
    if corpus:
        print(f"   corpus léxico: {len(corpus)} fragmentos de {len(fragmentos_por_documento)} documentos")

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
    print(f"\n🏆 Mejor Recall@{args.k}: {mejor.etiqueta} — {mejor.recall:.2f} (MRR {mejor.mrr:.2f})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(correr(main))
