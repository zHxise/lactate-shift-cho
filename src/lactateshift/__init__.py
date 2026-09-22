"""lactateshift — deteccion y prediccion temprana del lactate shift.

Dos piezas:

* :mod:`lactateshift.detect` — detecta el shift en cualquier serie de lactato.
* :mod:`lactateshift.features` — construye variables predictoras de una
  ventana temprana sin dejar entrar informacion posterior a ella.

Y dos apoyos: :mod:`lactateshift.datasets` (cultivos sinteticos con shift
conocido) y :mod:`lactateshift.validate` (controles contra la fuga de
informacion y el sobreajuste).
"""

from .detect import ShiftResult, detect_shift, detect_shift_batch, regularize, smooth
from .features import early_window_features, slope
from .datasets import make_culture, make_synthetic_cultures

__version__ = "0.1.0"
__all__ = [
    "ShiftResult", "detect_shift", "detect_shift_batch", "regularize", "smooth",
    "early_window_features", "slope",
    "make_culture", "make_synthetic_cultures",
]
