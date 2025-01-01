# Manim Fourier Series (Epicycles)

 > An improved version of [George Ogden's Fourier-Transform](https://github.com/George-Ogden/Fourier-Transform) tool.

A Fourier series is a method of converting any closed path into a series of rotating vectors. This Manim tool makes generating these rotating vectors significantly easier.


Simply make sure `manim` is [installed](https://docs.manim.community/en/stable/installation.html) and install:

```shell
pip3 install git+https://github.com/PeculiarProgrammer/manim-fourier-series.git
```

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

If the error `cannot open resource` is occurring, try changing `font=arial.ttf` to the path of a known font on your system.

For further information, consult [the documentation](https://peculiarprogrammer.github.io/manim-fourier-series/).

The package will be uploaded to PyPi as soon as the [Outlook domain limitation](https://blog.pypi.org/posts/2024-06-16-prohibiting-msn-emails/) has been lifted.
