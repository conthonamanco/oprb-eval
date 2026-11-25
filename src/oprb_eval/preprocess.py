"""Image preparation that a model's own documentation asks for.

Test pages are about 800 x 1030 pixels (roughly 95 DPI) and tilted by up to 5 degrees.
"""

from __future__ import annotations

MAX_SKEW = 6.0  # degrees


def skew_angle(image) -> float:
    """Tilt of the text lines in degrees, positive when they descend to the right.

    Projection profile: at the right angle, ink piles up in sharp rows.
    """
    import numpy as np

    gray = np.asarray(image.convert("L"))
    ys, xs = np.nonzero(gray < 128)
    if len(xs) < 100:
        return 0.0
    xs = xs - gray.shape[1] / 2
    offset = int(gray.shape[1] * np.tan(np.radians(MAX_SKEW + 1))) + 1

    def sharpness(degrees: float) -> float:
        rows = np.rint(ys - xs * np.tan(np.radians(degrees))).astype(np.int64) + offset
        profile = np.bincount(rows, minlength=gray.shape[0] + 2 * offset).astype(np.float64)
        return float(np.square(np.diff(profile)).sum())

    coarse = max(np.arange(-MAX_SKEW, MAX_SKEW + 0.01, 0.5), key=sharpness)
    return float(max(np.arange(coarse - 0.5, coarse + 0.51, 0.1), key=sharpness))


def deskew(image):
    from PIL import Image

    angle = skew_angle(image)
    if abs(angle) < 0.05:
        return image
    # PIL rotates counter-clockwise, which undoes lines that descend to the right.
    return image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor="white")


def upscale(image, factor: float):
    from PIL import Image

    if factor <= 1:
        return image
    return image.resize((round(image.width * factor), round(image.height * factor)), Image.LANCZOS)


def upscale_longest_edge(image, pixels: int):
    return upscale(image, pixels / max(image.size))
