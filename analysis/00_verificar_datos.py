"""Verifica que el dataset este presente y sea el correcto (SHA-256).

Correr antes que los demas scripts. Si el archivo es otra version o se
descargo incompleto, los resultados cambian sin avisar.

Desarrollado por Arturo Rodriguez.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ARCHIVO = RAIZ / "data" / "raw" / "1-s2_0-S0098135421000041-mmc3.xlsx"
SHA256 = "d572a1aa4a74bbaecb10046e759364d1740dc8731de86661d7646ad14758e839"


def sha256(path: Path, bloque: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(bloque):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not ARCHIVO.exists():
        print(f"FALTA el dataset: {ARCHIVO}")
        print("Instrucciones para obtenerlo: data/raw/README.md")
        return 1
    real = sha256(ARCHIVO)
    if real != SHA256:
        print("El archivo existe pero NO coincide con el esperado.")
        print(f"  esperado: {SHA256}")
        print(f"  obtenido: {real}")
        print("Puede ser otra version del suplemento o una descarga incompleta.")
        return 1
    print(f"OK: dataset verificado ({ARCHIVO.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
