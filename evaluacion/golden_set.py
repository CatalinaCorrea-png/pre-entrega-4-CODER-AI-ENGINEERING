"""Carga y validación del golden set.

Un benchmark que referencia un documento que no existe no falla: baja el Recall y parece
un problema del recuperador. Por eso se valida contra `data/` al cargarlo — un error de
tipeo en un nombre de archivo tiene que ser un error ruidoso, no una métrica peor.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from rag.config import DATA_DIR, GOLDEN_SET_PATH
from rag.errores import ErrorDeUso


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
    """Acepta los dos formatos: lista o string único.

    El formato de la consigna es `documento_id_esperado` (uno solo). Se soporta, pero el
    formato preferido es la lista: con un único documento relevante, Recall@k solo puede
    valer 0 o 1 y deja de distinguir "recuperó la mitad" de "no recuperó nada".
    """
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
