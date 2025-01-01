from __future__ import annotations

from manim import *


class ArrayMobject(Group):  # Supports both OpenGL and Cairo backends
    """A dummy Mobject to store an array of data that updates each frame."""

    def __init__(self, array: np.ndarray = None):
        super().__init__()

        self.set_data(array)

    def get_data(self) -> np.ndarray:
        """Get the data stored in the Mobject.

        Returns
        -------
        np.ndarray
            Returns the data stored in the Mobject.
        """
        return self.__data

    def set_data(self, data: np.ndarray) -> None:
        """Set the data stored in the Mobject.

        Parameters
        ----------
        data : np.ndarray
            The data to store in the Mobject.
        """
        self.__data = data

    def sum(self) -> ArrayMobject:
        """Accumulate the data and return a new Mobject.

        Returns
        -------
        ArrayMobject
            A new Mobject with the accumulated data.
        """
        return ArrayMobject(np.add.accumulate(self.get_data()))

    def __getitem__(self, idx: int) -> float:
        return self.get_data()[idx]

    def become(self, new_obj: ArrayMobject) -> ArrayMobject:
        """Set the data stored in the Mobject to the data stored in another
        Mobject.

        Parameters
        ----------
        new_obj : ArrayMobject
            The Mobject to copy the data from.

        Returns
        -------
        ArrayMobject
            Self, for chaining purposes.
        """
        self.set_data(new_obj.get_data())

        return self


class NestedPath(VGroup):  # Supports both OpenGL and Cairo backends
    """A Mobject that makes it easy to display points as a path.

    Examples
    --------

    ```py
    path = NestedPath()
    path.set_points_as_corners([LEFT, RIGHT, UP, DOWN])
    self.add(path)
    ```
    """

    def updater(self, point: np.ndarray, fade: float) -> NestedPath:
        previous_path = NestedPath()
        self.add(previous_path)

        previous_path.set_points_as_corners(self.points.copy())
        previous_path.add_updater(
            lambda path: (
                path.fade(fade)
                if path.get_stroke_opacity() > 2e-2
                else path.clear_updaters()
            )
        )

        self.add_points_as_corners([point])
        self.set_points_as_corners(self.points[-4:])

        return self
