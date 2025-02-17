# Manim Fourier Series (Epicycles)

<video controls autoplay muted>
    <source src="https://github.com/taibeled/manim-fourier-series/raw/refs/heads/master/examples/output/Handshake.mp4" type="video/mp4">
</video>


A Fourier series is a method of converting any closed path into a series of rotating vectors. This Manim tool makes generating these rotating vectors significantly easier.

Simply install:

```shell
pip3 install git+https://github.com/taibeled/manim-fourier-series.git
```

!!! warning

    `Manim` must be [installed](https://docs.manim.community/en/stable/installation.html). Only Python versions 3.11+ are supported.

And use:

```python
from manim import *

from manim_fourier_series import FourierSeries

class BasicExample(Scene):
    def construct(self):
        fs = FourierSeries(
            FourierSeries.text("G", font="arial.ttf"), # If you get the error "Cannot open resource", try changing the font to a path
            number=100,
        )

        self.add(fs.mobject)

        self.play(fs.revolve(2), run_time=20, rate_func=linear)
```

??? failure "OSError: cannot open resource"

    If the error `cannot open resource` is occurring, try changing `font=arial.ttf` to the path of a known font on your system.

??? note "Acknowledgments"

    Manim Fourier Series is an improved version of [George Ogden's Fourier-Transform](https://github.com/George-Ogden/Fourier-Transform) tool.

The package will be uploaded to PyPi as soon as the [Outlook domain limitation](https://blog.pypi.org/posts/2024-06-16-prohibiting-msn-emails/) has been lifted.
