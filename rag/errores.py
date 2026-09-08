"""Errores de uso, separados de los errores de programa.

`ErrorDeUso` marca las fallas que causa la configuración del entorno, no un bug: falta una
API key, el corpus no está descargado, el índice existe con otra dimensión. Todas tienen
en común que el usuario puede arreglarlas y que el mensaje ya explica cómo.

Los scripts las atrapan y muestran solo el mensaje. Un traceback de 30 líneas para decir
"falta PINECONE_API_KEY" esconde la única línea que importa, y encima sugiere que el
programa se rompió cuando en realidad está funcionando como debe.
"""


class ErrorDeUso(RuntimeError):
    """Falla que el usuario puede corregir; su mensaje dice cómo."""
