from typing import Callable, Optional, Tuple, Union

import cv2
import numpy as np
from manim import config, logger

from .paths import *


def normalize(
    points: np.ndarray,
    return_factor: bool = False,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """Normalize the points to fit within the frame.

    Parameters
    ----------
    points : np.ndarray
        The points to normalize
    return_factor : bool, optional
        Whether or not the scale factor should be returned, by default False
    width : Optional[int], optional
        How wide the frame should be. When None, it is the width of the Manim frame. By default None
    height : Optional[int], optional
        How tall the frame should be. When None, it is the height of the Manim frame. By default None

    Returns
    -------
    Union[np.ndarray, Tuple[np.ndarray, float]]
        The normalized points. If return_factor is True, a tuple is returned with the scale factor as the second element.
    """
    if width is None:
        width = config.frame_width
    if height is None:
        height = config.frame_height

    scale = (
        max(
            (max(points.real) - min(points.real)) / width,
            (max(points.imag) - min(points.imag)) / height,
        )
        / 0.9
    )  # Determine scale factor such that all points fit within 90% of the frame

    points /= scale
    points -= (max(points.real) + min(points.real)) / 2 - (
        max(points.imag) + min(points.imag)
    ) / 2j

    if return_factor:
        return points, 1
    else:
        return points


def fft(points: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform a Fast Fourier Transform on the points.

    Parameters
    ----------
    points : np.ndarray
        The points to transform
    n : int
        The number of frequencies to keep

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        The amplitudes, frequencies and phases of the Fourier transform
    """
    coefficients = np.fft.fft(points, norm="forward")
    frequencies = np.fft.fftfreq(len(points), 1 / len(points))

    indices = np.argsort(-abs(coefficients))[:n]
    frequencies = frequencies[indices]
    coefficients = coefficients[indices]

    phases = np.angle(coefficients)
    amplitudes = abs(coefficients)

    return amplitudes, frequencies, phases


def extract_edges(
    image: np.ndarray,
    shortest_path: Callable[[np.ndarray], np.ndarray] = self_organising_maps,
    subsample=True,
    remove_internal=True,
    multiple_contours=False,
) -> np.ndarray:
    """Extract the edges from an image.

    Parameters
    ----------
    image : np.ndarray
        The image to extract the edges from
    shortest_path : Callable[[np.ndarray], np.ndarray], optional
        The algorithm to determine the path, by default self_organising_maps
    subsample : bool, optional
        Should points be sampled, by default True
    remove_internal : bool, optional
        Whether or not internal contours should be removed. In simple terms, when this
        is True, any paths that are within another path will not be rendered. For
        instance, an `o` would not have the interior line drawn, only the exterior.
        By default True
    multiple_contours : bool, optional
        Whether or not multiple contours are allowed. If this is False, only the contour
        occupying the largest area will be displayed. Note that this cannot be False
        when `remove_internal` is False. By default False

    Returns
    -------
    np.ndarray
        The edges of the image
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    all_contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if not remove_internal:
        if multiple_contours == False:
            logger.warning(
                "multiple_contours should be True if remove_internal is False. Value will be temporarily set to True"
            )
            multiple_contours = True  # Future-proofing
        contours = all_contours
    else:
        if len(contours) < len(all_contours):
            logger.info("Omitting internal contours")
        if not multiple_contours and len(contours) > 1:
            logger.info("Multiple exterior contours found, using the largest one")
            contours = [max(contours, key=cv2.contourArea)]

    points = np.concatenate(contours).reshape(
        -1, 2
    )  # Convert points to complex numbers
    points = points[:, 0] - 1j * points[:, 1]

    points, scale = normalize(points, True)

    if not subsample:
        scale = 0
    return shortest_path(points[:: max(int(scale / 10), 1)])
