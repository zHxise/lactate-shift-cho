"""Ejecuta los bloques de codigo Python del README.

Existe porque un ejemplo del README usaba una variable que nunca definia y
fallaba al copiarlo. Se detecto al probar el repositorio como un usuario
nuevo, no con los demas tests.
"""

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"
BLOQUES = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), flags=re.S)


@pytest.mark.parametrize("codigo", BLOQUES, ids=[f"bloque_{i}" for i in range(len(BLOQUES))])
def test_ejemplo_del_readme_corre(codigo):
    exec(compile(codigo, "README.md", "exec"), {})


def test_hay_ejemplos():
    assert len(BLOQUES) >= 2
