"""Golden set y métricas de recuperación.

La consigna pide Precision@k y Recall@k. Se suman hit-rate y MRR porque las dos primeras
son ciegas al orden, y con un top-k chico lo que queda arriba es lo que llega al LLM.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from .config import DATA_DIR, GOLDEN_SET_PATH, ErrorDeUso


# --- El golden set: carga y validación --------------------------------------

class GoldenSetInvalido(ErrorDeUso):
    """El benchmark referencia documentos que no están en el corpus."""


@dataclass(frozen=True)
class Caso:
    """Una pregunta del benchmark con su verdad de referencia."""

    id: str
    pregunta: str
    documentos_esperados: frozenset[str]
    tipo: str
    nota: str = ""


def _documentos_de(caso: dict) -> frozenset[str]:
    """Acepta los dos formatos: lista o string único."""
    if "documentos_esperados" in caso:
        return frozenset(caso["documentos_esperados"])
    if "documento_id_esperado" in caso:
        return frozenset([caso["documento_id_esperado"]])
    raise GoldenSetInvalido(
        f"El caso {caso.get('id', caso)!r} no declara documentos esperados."
    )


def cargar(ruta: Path = GOLDEN_SET_PATH, validar_contra_corpus: bool = True) -> list[Caso]:
    """Lee el golden set y verifica que cada documento esperado exista en `data/`."""
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    casos = [
        Caso(
            id=caso.get("id", f"caso-{numero}"),
            pregunta=caso["pregunta"],
            documentos_esperados=_documentos_de(caso),
            tipo=caso.get("tipo", "sin_tipo"),
            nota=caso.get("nota", ""),
        )
        for numero, caso in enumerate(datos["casos"], start=1)
    ]

    if validar_contra_corpus:
        existentes = {archivo.name for archivo in DATA_DIR.glob("*.md")}
        faltantes = {
            documento
            for caso in casos
            for documento in caso.documentos_esperados
            if documento not in existentes
        }
        if faltantes:
            raise GoldenSetInvalido(
                f"El golden set espera documentos que no están en {DATA_DIR}: "
                f"{', '.join(sorted(faltantes))}.\n"
                "Corré: python scripts/descargar_dataset.py"
            )

    return casos


# --- Las métricas -----------------------------------------------------------

def precision_at_k(recuperados: list[str], relevantes: frozenset[str], k: int) -> float:
    """Fracción de las k posiciones ocupada por fragmentos de un documento relevante."""
    if k <= 0:
        return 0.0
    return sum(1 for fuente in recuperados[:k] if fuente in relevantes) / k


def recall_at_k(recuperados: list[str], relevantes: frozenset[str], k: int) -> float:
    """Proporción de los documentos relevantes que aparece en el top-k."""
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
