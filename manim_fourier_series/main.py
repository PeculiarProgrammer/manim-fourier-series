from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from manim import *
from PIL import Image, ImageFont
from svgpathtools import svg2paths

from .mobjects import ArrayMobject, NestedPath
from .paths import greedy_shortest_path
from .utils import extract_edges, fft, normalize


class FourierSeries:
    """A class representing a Fourier series animation for Manim.

    This uses points generated from text, images, svgs, polygons, or
    manually to solve for circles and arrows that approximate the input.
    """

    def __init__(
        self,
        points: np.ndarray,
        number: int = 50,
        fade: float = 0.005,
        circle_opacity: float = 0.5,
        arrow_opacity: float = 0.8,
        width: Optional[int] = None,
        height: Optional[int] = None,
        shift: Optional[complex] = 0,
    ):
        """Initialize the FourierSeries object.

        Parameters
        ----------
        points : np.ndarray
            The points to generate the Fourier series from. This can be generated
            from the `text`, `image`, or `polygon` static methods on this class.
        number : int, optional
            How many circles should be generated. The more circles, the more precise the output, but the longer time to render. By default 50
        fade : float, optional
            The rate at which the path should fade. The path exponentially fades by this amount each frame. By default 0.005
        circle_opacity : float, optional
            The opacity of the circles, by default 0.5
        arrow_opacity : float, optional
            The opacity of the arrows, by default 0.8
        width : int, optional
            The width of the window. If None, then this will occupy the entire window. The points will only be scaled uniformly. By default None
        height : int, optional
            The height of the window. If None, then this will occupy the entire window. The points will only be scaled uniformly. By default None
        shift : Optional[complex], optional
            The amount to shift the points by. By default 0

        Examples
        --------
        The following code animates a Fourier series for a nonagon.

        ```py
        points = FourierSeries.polygon(9)
        fs = FourierSeries(points)
        self.add(fs.mobject)

        self.play(fs.revolve(1), run_time=5, rate_func=linear)
        ```
        """
        points = normalize(points, width=width, height=height) + shift

        self.points = points
        self.N = min(number, len(self.points))
        self.fade = fade

        self.amplitudes, self.frequencies, self.phases = fft(self.points, self.N)

        self.tracker = ValueTracker(0)

        self.arrows = [
            Arrow(
                ORIGIN,
                RIGHT,
                stroke_opacity=arrow_opacity,
                tip_style={
                    "stroke_opacity": arrow_opacity,
                    "fill_opacity": arrow_opacity,
                },
            )
            for _ in range(self.N)
        ]
        self.circles = [
            Circle(
                radius=self.amplitudes[i],
                color=TEAL,
                stroke_width=2,
                stroke_opacity=circle_opacity,
            )
            for i in range(self.N)
        ]

        self.path = NestedPath()

        self.values = ArrayMobject()
        self.cumulative = ArrayMobject()

        self.values.add_updater(
            lambda array, dt: array.set_data(
                np.array(
                    [0]
                    + [
                        a * np.exp(1j * (p + self.tracker.get_value() * f))
                        for a, f, p in zip(
                            self.amplitudes, self.frequencies, self.phases
                        )
                    ]
                )
            ),  # This lambda sets the value to e^i(a + wt)
            call_updater=True,
        )
        self.cumulative.add_updater(
            lambda array, dt: array.become(self.values.sum()), call_updater=True
        )

        for i, (arrow, ring) in enumerate(zip(self.arrows, self.circles)):
            arrow.idx = i
            ring.idx = i
            ring.add_updater(
                lambda ring: ring.move_to(complex_to_R3(self.cumulative[ring.idx]))
            )
            arrow.add_updater(
                lambda arrow: arrow.become(
                    Arrow(
                        complex_to_R3(self.cumulative[arrow.idx]),
                        complex_to_R3(self.cumulative[arrow.idx + 1]),
                        buff=0,
                        max_tip_length_to_length_ratio=0.2,
                        stroke_width=2,
                        stroke_opacity=arrow_opacity,
                        tip_style={
                            "stroke_opacity": arrow_opacity,
                            "fill_opacity": arrow_opacity,
                        },
                    )
                )
            )

        self.path.set_points_as_corners([complex_to_R3(self.cumulative[-1])] * 2)
        self.path.add_updater(
            lambda path: path.updater(complex_to_R3(self.cumulative[-1]), self.fade)
        )

        self.mobject = Group(
            *self.arrows, *self.circles, self.values, self.cumulative, self.path
        )

    def zoomed_display(
        self, scene: ZoomedScene, animate: bool = True, scale_factor: float = 2
    ) -> FourierSeries:
        """Add a window to the scene that follows the path.

        Parameters
        ----------
        scene : ZoomedScene
            The scene to add the window to. This must be a `ZoomedScene` otherwise the window will not work.
        animate : bool, optional
            Whether or not the window's entrance should be animated, by default True
        scale_factor : float, optional
            How much the zoomed camera should be scaled by. The smaller, the more zoomed in. By default 2

        Returns
        -------
        FourierSeries
            Self, for chaining purposes

        Raises
        ------
        AssertionError
            If the scene is not a `ZoomedScene`

        Examples
        --------
        The following code generates a set of points from the text "Guru" and then animates it through a Fourier series.

        ```py
        class BasicExample(ZoomedScene):
            def construct(self):
                points = FourierSeries.text("Guru", r"path/to/font.ttf")

                fs = FourierSeries(points)
                self.add(fs.mobject)

                fs.zoomed_display(self)

                self.play(fs.revolve(1), run_time=5, rate_func=linear)
        ```
        """
        assert isinstance(scene, ZoomedScene), "The scene must be a ZoomedScene"

        scene.zoomed_camera.frame.scale(scale_factor)
        scene.zoomed_camera.frame.move_to(complex_to_R3(self.cumulative[-1]))

        scene.activate_zooming(animate)

        scene.zoomed_camera.frame.add_updater(
            lambda frame: frame.move_to(complex_to_R3(self.cumulative[-1]))
        )

        return self

    def revolve(self, revolutions: float = 1) -> Animation:
        """Animate the Fourier series.

        Parameters
        ----------
        revolutions : float, optional
            How many times the image should be drawn, by default 1

        Returns
        -------
        Animation
            An animation that can be passed to `self.play`. A linear rate function is highly recommended.

        Examples
        --------
        The following code generates a set of points from the text "Guru" and then animates it through a Fourier series.

        ```py
        points = FourierSeries.text("Guru", r"path/to/font.ttf")
        fs = FourierSeries(points)
        self.add(fs.mobject)

        self.play(fs.revolve(1), run_time=5, rate_func=linear) # A linear rate function is highly recommended
        """
        return self.tracker.animate.set_value(revolutions * 2 * np.pi)

    def display_complete_path(self, opacity: float = 0.5) -> FourierSeries:
        """Displays the entire path that the Fourier series will go through.

        Parameters
        ----------
        opacity : float, optional
            The opacity of the path, by default 0.5

        Returns
        -------
        FourierSeries
            Self, for chaining purposes
        """
        self.path.clear_updaters()
        self.path.set_points_as_corners([complex_to_R3(point) for point in self.points])
        self.path.set_stroke(opacity=opacity)
        return self

    ## Static methods for point generation

    @staticmethod
    def polygon(n: int) -> np.ndarray:
        """Generate a set of points from a regular polygon.

        Parameters
        ----------
        n : int
            The number of sides the polygon should have.

        Returns
        -------
        np.ndarray
            The points generated from the polygon. Pass this to the FourierSeries constructor.

        Examples
        --------
        The following code generates a set of points from a regular pentagon and then animates it through a Fourier series.

        ```py
        points = FourierSeries.polygon(5)
        fs = FourierSeries(points)
        self.add(fs.mobject)

        self.play(fs.revolve(1), run_time=5, rate_func=linear)
        ```
        """
        points = np.array(
            [
                np.linspace(
                    np.exp(2j * k * np.pi / n), np.exp(2j * (k + 1) * np.pi / n), 1000
                )
                for k in range(n)
            ]
        ).reshape(-1)

        points *= 1j
        if not n % 2:
            points *= np.exp(1j * np.pi / n)
        return points

    @staticmethod
    def text(
        text: str,
        font: str,
        remove_internal: bool = True,
        multiple_contours: bool = False,
    ) -> np.ndarray:
        """Generate a set of points from text.

        Parameters
        ----------
        text : str
            The text to generate points from.
        font : str
            A path to a font file, to be passed to `PIL.ImageFont.truetype`. A cursive font
            is highly recommended for multiple letters. Try [Brush Script MT](https://github.com/PeculiarProgrammer/manim-fourier-series) or similar.
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
            The points generated from the text. Pass this to the FourierSeries constructor.
            Note that the points will be an array of complex numbers.

        Examples
        --------
        The following code generates a set of points from the text "Guru" and then animates it through a Fourier series.

        ```py
        points = FourierSeries.text("Guru", r"path/to/font.ttf")
        fs = FourierSeries(points)
        self.add(fs.mobject)
        ```
        """
        font_file = ImageFont.truetype(font, size=1000)

        mask = font_file.getmask(text, mode="1")

        image = Image.frombytes(mask.mode, mask.size, bytes(mask))
        image = np.array(image)

        return extract_edges(
            image, greedy_shortest_path, False, remove_internal, multiple_contours
        )

    @staticmethod
    def image(
        filename: str,
        greedy: bool = False,
        remove_internal: bool = True,
        multiple_contours: bool = False,
    ) -> np.ndarray:
        """Generate a set of points from the edges of an image.

        Parameters
        ----------
        filename : str
            Where the image is located.
        greedy : bool, optional
            This should normally be False, however if the image was rendered poorly, such as with
            peculiar lines, try setting this to True. By default False
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
            The points generated from the image. Pass this to the FourierSeries constructor.
            Note that the points will be an array of complex numbers.

        Examples
        --------

        The following code generates a set of points from an image of a bird and then animates it through a Fourier series.
        ```py
        points = FourierSeries.image(r"path/to/bird.jpg")
        fs = FourierSeries(points)
        self.add(fs.mobject)
        ```
        """
        image = cv2.imread(filename)

        scale = min(
            920 / image.shape[0], 1080 / image.shape[1]
        )  # Scale image to be no greater than 1080 x 920
        image = cv2.resize(
            image, (int(image.shape[1] * scale), int(image.shape[0] * scale))
        )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = 255 - cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )  # Determine different areas

        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )  # Find outlines of areas

        largest = max(contours, key=cv2.contourArea)  # Find the largest area

        image = np.zeros(
            image.shape, dtype=image.dtype
        )  # Shade everything that isn't the largest area

        cv2.drawContours(image, [largest], -1, (255, 255, 255), -1)

        if greedy:
            return extract_edges(
                image,
                greedy_shortest_path,
                remove_internal=remove_internal,
                multiple_contours=multiple_contours,
            )
        return extract_edges(
            image, remove_internal=remove_internal, multiple_contours=multiple_contours
        )

    @staticmethod
    def numpy_points(filename: str) -> np.ndarray:
        """Load a set of points from a numpy file.

        Parameters
        ----------
        filename : str
            Where the numpy file is located.

        Returns
        -------
        np.ndarray
            The points loaded from the numpy file. Pass this to the FourierSeries constructor.
            Note that the points will be an array of complex numbers.

        Examples
        --------
        The following code loads a set of points from a numpy file and then animates it through a Fourier series.

        ```py
        points = FourierSeries.numpy_points(r"path/to/points.npy")
        fs = FourierSeries(points)
        self.add(fs.mobject)
        ```
        """
        return np.load(filename)

    @staticmethod
    def svg(filename: str) -> np.ndarray:
        """Generate a set of points from an SVG file.

        Parameters
        ----------
        filename : str
            Where the SVG file is located.

        Returns
        -------
        np.ndarray
            The points generated from the SVG file. Pass this to the FourierSeries constructor.
            Note that the points will be an array of complex numbers.

        Examples
        --------
        The following code generates a set of points from an SVG file and then animates it through a Fourier series.

        ```py
        points = FourierSeries.svg(r"path/to/file.svg")
        fs = FourierSeries(points)
        self.add(fs.mobject)
        ```
        """
        paths, _ = svg2paths(filename)

        return np.concatenate(
            [
                shape.points(np.linspace(0, 1, 100 * int(shape.length())))
                for path in paths
                for shape in path
            ]
        ).conjugate()
