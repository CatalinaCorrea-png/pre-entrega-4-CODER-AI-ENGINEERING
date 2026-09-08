"""Pruebas de las métricas, el tokenizador y el contrato de metadatos.

    python tests/test_metricas.py

Corren sin red y sin API keys. Es a propósito: son justamente las partes donde un error
no se nota. Un Precision@k mal calculado no lanza ninguna excepción — devuelve un número
plausible, se copia al README y nadie lo vuelve a mirar.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluacion.golden_set import cargar
from evaluacion.metricas import (
    hit_at_k,
    mrr_at_k,
    precision_at_k,
    recall_at_k,
    techo_precision,
)
from rag.consola import correr
from rag.ingesta.chunking import fragmentar
from rag.ingesta.metadata import CAMPOS, aplicar_metadata, id_vector
from rag.recuperacion.tokenizacion import tokenizar

from langchain_core.documents import Document

TOP5 = ["a.md", "b.md", "a.md", "c.md", "d.md"]


def test_precision_divide_por_k():
    # 2 de las 5 posiciones vienen del documento relevante.
    assert precision_at_k(TOP5, frozenset(["a.md"]), 5) == 0.4
    # Aunque devuelva menos de k, el denominador sigue siendo k: recuperar de menos no
    # puede ser una ventaja.
    assert precision_at_k(["a.md", "b.md"], frozenset(["a.md"]), 5) == 0.2


def test_recall_cuenta_documentos_no_fragmentos():
    # a.md aparece dos veces pero es un solo documento recuperado.
    assert recall_at_k(TOP5, frozenset(["a.md"]), 5) == 1.0
    # Con dos relevantes y uno hallado, el recall es parcial: eso es lo que un golden set
    # de un solo documento esperado nunca puede mostrar.
    assert recall_at_k(TOP5, frozenset(["a.md", "z.md"]), 5) == 0.5
    assert recall_at_k(TOP5, frozenset(["z.md"]), 5) == 0.0


def test_hit_y_mrr_miden_cosas_distintas():
    # Mismo hit, distinto MRR: la posición importa y Precision/Recall no la ven.
    assert hit_at_k(TOP5, frozenset(["a.md"]), 5) == hit_at_k(TOP5, frozenset(["d.md"]), 5) == 1.0
    assert mrr_at_k(TOP5, frozenset(["a.md"]), 5) == 1.0
    assert mrr_at_k(TOP5, frozenset(["c.md"]), 5) == 0.25
    assert mrr_at_k(TOP5, frozenset(["z.md"]), 5) == 0.0


def test_k_acota_la_ventana():
    assert precision_at_k(TOP5, frozenset(["c.md"]), 3) == 0.0
    assert recall_at_k(TOP5, frozenset(["c.md"]), 3) == 0.0
    assert recall_at_k(TOP5, frozenset(["c.md"]), 5) == 1.0


def test_techo_de_precision():
    # Un documento de 3 fragmentos no puede llenar 5 posiciones.
    assert techo_precision(frozenset(["a.md"]), {"a.md": 3}, 5) == 0.6
    assert techo_precision(frozenset(["a.md"]), {"a.md": 40}, 5) == 1.0
    assert techo_precision(frozenset(["a.md", "b.md"]), {"a.md": 3, "b.md": 4}, 5) == 1.0


def test_tokenizador_habilita_la_coincidencia_exacta():
    # El caso que el tokenizador por defecto de BM25 (str.split) no resuelve.
    assert tokenizar("¿Cómo uso TypeAdapter?") == ["como", "uso", "typeadapter"]
    assert tokenizar("TypeAdapter") == tokenizar("typeadapter,")
    # El guion bajo es parte del identificador: model_validator es un término, no dos.
    assert tokenizar("model_validator(mode='after')") == ["model_validator", "mode", "after"]
    # Y los acentos no pueden partir un término en dos.
    assert tokenizar("validación") == tokenizar("validacion")


def test_metadata_cumple_el_contrato():
    documentos = [
        Document(page_content="uno\n\n" + "texto de prueba. " * 200, metadata={"source": "alias.md", "titulo": "Alias"}),
        Document(page_content="dos\n\n" + "otro texto. " * 200, metadata={"source": "config.md", "titulo": "Config"}),
    ]
    chunks = aplicar_metadata(fragmentar(documentos))

    esperados = set(CAMPOS) - {"text"}  # `text` lo inyecta PineconeVectorStore al subir
    for chunk in chunks:
        assert set(chunk.metadata) == esperados, chunk.metadata

    # chunk_index se numera por documento, no globalmente.
    por_documento = {}
    for chunk in chunks:
        por_documento.setdefault(chunk.metadata["source"], []).append(chunk.metadata["chunk_index"])
    for source, indices in por_documento.items():
        assert indices == list(range(len(indices))), (source, indices)
        assert all(c.metadata["total_chunks"] == len(indices) for c in chunks if c.metadata["source"] == source)

    # Los IDs son determinísticos y únicos: reingestar actualiza, no duplica.
    ids = [id_vector(c.metadata["source"], c.metadata["chunk_index"]) for c in chunks]
    assert len(set(ids)) == len(ids)
    assert ids[0] == "alias.md::0"


def test_golden_set_es_valido():
    casos = cargar()
    assert len(casos) >= 5, "la consigna pide al menos 5 preguntas"
    assert len({c.id for c in casos}) == len(casos), "hay ids repetidos"
    assert all(c.documentos_esperados for c in casos)
    # El benchmark tiene que cubrir los dos extremos, o no mide lo que el híbrido resuelve.
    tipos = {c.tipo for c in casos}
    assert {"lexica", "semantica"} <= tipos, tipos


def main() -> int:
    pruebas = [valor for nombre, valor in sorted(globals().items()) if nombre.startswith("test_")]
    fallidas = 0
    for prueba in pruebas:
        try:
            prueba()
        except AssertionError as error:
            fallidas += 1
            print(f"❌ {prueba.__name__}: {error}")
        else:
            print(f"✅ {prueba.__name__}")

    print(f"\n{len(pruebas) - fallidas}/{len(pruebas)} pruebas pasaron")
    return 1 if fallidas else 0


if __name__ == "__main__":
    raise SystemExit(correr(main))
