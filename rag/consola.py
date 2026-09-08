"""Salida por consola en UTF-8, también en Windows.

`cmd.exe` y PowerShell usan cp1252 por defecto. Cualquier `print()` con un emoji o una
vocal acentuada revienta con `UnicodeEncodeError` y mata el script a mitad de la ingesta —
un fallo de presentación que se lleva puesto el trabajo real.

Los scripts llaman a `configurar()` antes de imprimir. Vive en el paquete pero no se
ejecuta al importarlo: `rag` es una librería y no debe tocar el estado global del proceso
por el solo hecho de ser importada.
"""

import sys

from .errores import ErrorDeUso


def configurar() -> None:
    """Fuerza UTF-8 en stdout y stderr, sin fallar si el flujo no lo soporta."""
    for flujo in (sys.stdout, sys.stderr):
        reconfigurar = getattr(flujo, "reconfigure", None)
        if reconfigurar is None:
            continue
        try:
            reconfigurar(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def correr(main) -> int:
    """Ejecuta el `main` de un script con la consola lista y sin tracebacks inútiles.

    Un `ErrorDeUso` ya trae un mensaje que dice qué hacer: se imprime y se sale con
    código 1. Cualquier otra excepción sí es un bug y se deja propagar con su traceback
    completo, porque ahí el traceback es la información.
    """
    configurar()
    try:
        return main()
    except ErrorDeUso as error:
        print(f"\n❌ {error}\n", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n👋 Cancelado.", file=sys.stderr)
        return 130
