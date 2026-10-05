"""lactateshift: deteccion y prediccion temprana del lactate shift.

Modulos:
    detect    deteccion del shift en una serie de lactato
    features  variables de una ventana temprana
    datasets  cultivos sinteticos con shift conocido
    validate  permutacion, leave-one-group-out y SHAP fuera de fold
    qc        puntos bajos aislados

Desarrollado por Arturo Rodriguez.
"""

from .detect import ShiftResult, detect_shift, detect_shift_batch, regularize, smooth
from .features import early_window_features, slope
from .datasets import make_culture, make_synthetic_cultures
from .qc import isolated_low_points

__version__ = "0.1.0"
__author__ = "Arturo Rodriguez"
__all__ = [
    "ShiftResult", "detect_shift", "detect_shift_batch", "regularize", "smooth",
    "early_window_features", "slope",
    "make_culture", "make_synthetic_cultures",
    "isolated_low_points",
]
