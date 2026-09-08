"""Precision@k, Recall@k y MRR@k sobre el golden set.

Las dos que pide la consigna son Precision@k y Recall@k. MRR se suma porque las otras dos
son ciegas al orden: recuperar el documento correcto en la posición 1 o en la 5 da
exactamente la misma Precision@5, y no es lo mismo — con `top_k` chico, lo que queda
arriba es lo que termina llegando al LLM.

Sobre el techo de Precision@k
-----------------------------
La consigna define Precision@5 como "qué porcentaje de los 5 recuperados son realmente
útiles", y acá un fragmento cuenta como útil si viene de un documento esperado. Esa
definición tiene un techo estructural que conviene tener presente al leer los números: si
el documento correcto solo produjo 3 fragmentos, la Precision@5 máxima alcanzable es
3/5 = 0.6 aunque el sistema haya funcionado perfecto. `techo_precision` lo calcula, para
comparar el valor medido contra lo máximo posible en vez de contra un 1.0 inalcanzable.
"""

from dataclasses import dataclass
from statistics import mean

from evaluacion.golden_set import Caso


def precision_at_k(recuperados: list[str], relevantes: frozenset[str], k: int) -> float:
    """Fracción de las k posiciones ocupada por fragmentos de un documento relevante.

    Se divide por `k` y no por `len(recuperados)`: si el sistema devuelve 3 documentos en
    vez de 5, dividir por 3 lo premiaría por haber recuperado de menos.
    """
    if k <= 0:
        return 0.0
    return sum(1 for fuente in recuperados[:k] if fuente in relevantes) / k


def recall_at_k(recuperados: list[str], relevantes: frozenset[str], k: int) -> float:
    """Proporción de los documentos relevantes que aparece en el top-k.

    A nivel documento, no de fragmento: cinco fragmentos del mismo archivo cuentan como un
    documento recuperado, no como cinco.
    """
    if not relevantes:
        return 0.0
    return len(set(recuperados[:k]) & relevantes) / len(relevantes)


def hit_at_k(recuperados: list[str], relevantes: frozenset[str], k: int) -> float:
    """1 si al menos un documento relevante entró en el top-k. El "¿está o no está?"."""
    return 1.0 if set(recuperados[:k]) & relevantes else 0.0


def mrr_at_k(recuperados: list[str], relevantes: frozenset[str], k: int) -> float:
    """Inversa de la posición del primer acierto: 1.0 si salió primero, 0.2 si salió quinto."""
    for posicion, fuente in enumerate(recuperados[:k], start=1):
        if fuente in relevantes:
            return 1.0 / posicion
    return 0.0


def techo_precision(
    relevantes: frozenset[str],
    fragmentos_por_documento: dict[str, int],
    k: int,
) -> float:
    """Precision@k máxima alcanzable dado cuántos fragmentos produjo cada documento."""
    if k <= 0:
        return 0.0
    disponibles = sum(fragmentos_por_documento.get(documento, 0) for documento in relevantes)
    return min(disponibles, k) / k


@dataclass(frozen=True)
class ResultadoCaso:
    """Métricas de una pregunta."""

    caso: Caso
    recuperados: list[str]
    precision: float
    recall: float
    hit: float
    mrr: float
    techo: float

    @property
    def aciertos(self) -> int:
        return sum(1 for f in self.recuperados if f in self.caso.documentos_esperados)


def evaluar_caso(
    caso: Caso,
    recuperados: list[str],
    k: int,
    fragmentos_por_documento: dict[str, int],
) -> ResultadoCaso:
    return ResultadoCaso(
        caso=caso,
        recuperados=recuperados[:k],
        precision=precision_at_k(recuperados, caso.documentos_esperados, k),
        recall=recall_at_k(recuperados, caso.documentos_esperados, k),
        hit=hit_at_k(recuperados, caso.documentos_esperados, k),
        mrr=mrr_at_k(recuperados, caso.documentos_esperados, k),
        techo=techo_precision(caso.documentos_esperados, fragmentos_por_documento, k),
    )


@dataclass(frozen=True)
class Reporte:
    """Resumen de una configuración evaluada sobre todo el golden set."""

    etiqueta: str
    resultados: list["ResultadoCaso"]

    def _promedio(self, atributo: str) -> float:
        return mean(getattr(r, atributo) for r in self.resultados) if self.resultados else 0.0

    @property
    def precision(self) -> float:
        return self._promedio("precision")

    @property
    def recall(self) -> float:
        return self._promedio("recall")

    @property
    def hit_rate(self) -> float:
        return self._promedio("hit")

    @property
    def mrr(self) -> float:
        return self._promedio("mrr")

    @property
    def techo(self) -> float:
        return self._promedio("techo")

    def por_tipo(self, tipo: str) -> "Reporte":
        return Reporte(self.etiqueta, [r for r in self.resultados if r.caso.tipo == tipo])
