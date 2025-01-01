from manim import *

from manim_fourier_series import FourierSeries


class Pentagon(ZoomedScene):
    def construct(self):
        fs = FourierSeries(FourierSeries.polygon(5))

        self.add(fs.mobject)

        fs.zoomed_display(self, scale_factor=0.5)

        self.play(fs.revolve(2), run_time=20, rate_func=linear)


class Triangle(ZoomedScene):
    def construct(self):
        fs = FourierSeries(FourierSeries.polygon(3)).display_complete_path()

        self.add(fs.mobject)

        fs.zoomed_display(self, scale_factor=0.5)

        self.play(fs.revolve(2), run_time=20, rate_func=linear)


class Guru(ZoomedScene):
    def construct(self):
        fs = FourierSeries(
            FourierSeries.text("Guru", font="./examples/fonts/brush.ttf"),
            number=200,
        )

        self.add(fs.mobject)

        fs.zoomed_display(self)

        self.play(fs.revolve(2), run_time=20, rate_func=linear)


class Bird(ZoomedScene):
    def construct(self):
        fs = FourierSeries(
            FourierSeries.image("./examples/input/bird.jpg"),
            number=100,
        )

        self.add(fs.mobject)

        fs.zoomed_display(self)

        self.play(fs.revolve(2), run_time=20, rate_func=linear)


class G(Scene):
    def construct(self):
        fs = FourierSeries(
            FourierSeries.text(
                "G", font="arial.ttf"
            ),  # If you get the error "Cannot open resource", try changing the font to a path
            number=100,
        )

        self.add(fs.mobject)

        self.play(fs.revolve(2), run_time=20, rate_func=linear)


class Handshake(ZoomedScene):
    def construct(self):
        fs = FourierSeries(
            FourierSeries.image("./examples/input/handshake.png", greedy=True),
            number=300,
            circle_opacity=0.05,
            fade=0.00000005,
            width=7,
            shift=-3.5,
        )

        self.add(fs.mobject)

        self.zoomed_display.scale(2)
        self.zoomed_display.to_corner(UR)
        fs.zoomed_display(self, scale_factor=1)

        self.play(fs.revolve(1), run_time=60, rate_func=linear)


class Skyline(ZoomedScene):
    def construct(self):
        fs = FourierSeries(
            FourierSeries.image("./examples/input/skyline.png"),
            number=300,
            circle_opacity=0.05,
            fade=0.00000005,
            width=7,
            shift=-3.5,
        )

        self.add(fs.mobject)

        self.zoomed_display.scale(2)
        self.zoomed_display.to_corner(UR)
        fs.zoomed_display(self, scale_factor=1)

        self.play(fs.revolve(1), run_time=60, rate_func=linear)


class TrebleClef(ZoomedScene):
    def construct(self):
        fs = FourierSeries(
            FourierSeries.numpy_points("./examples/input/treble-clef.npy"),
            number=1000,
            circle_opacity=0.005,
            shift=-3.5,
        )

        self.add(fs.mobject)

        self.zoomed_display.scale_to_fit_height(config.frame_height * 0.9)
        self.zoomed_display.to_corner(UR)
        fs.zoomed_display(self)

        self.play(fs.revolve(2), run_time=20, rate_func=linear)


class Star(ZoomedScene):
    def construct(self):
        fs = FourierSeries(
            FourierSeries.svg("./examples/input/star.svg"),
            number=25,
        )

        self.add(fs.mobject)

        fs.zoomed_display(self, scale_factor=1)

        self.play(fs.revolve(2), run_time=20, rate_func=linear)


class GitHub(Scene):
    def construct(self):
        fs = FourierSeries(
            FourierSeries.svg("./examples/input/github.svg"),
            number=50,
        )

        self.add(fs.mobject)

        self.play(fs.revolve(2), run_time=20, rate_func=linear)


class Heart(ZoomedScene):
    def construct(self):
        fs = FourierSeries(
            FourierSeries.svg("./examples/input/heart.svg"),
            number=50,
            shift=-2,
        )

        self.add(fs.mobject)

        fs.zoomed_display(self, scale_factor=1)

        self.play(fs.revolve(2), run_time=20, rate_func=linear)


class Pi(ZoomedScene):
    def construct(self):
        fs = FourierSeries(
            FourierSeries.svg("./examples/input/pi.svg"),
            number=100,
        )

        self.add(fs.mobject)

        self.zoomed_display.to_corner(DL)
        fs.zoomed_display(self, scale_factor=1)

        self.play(fs.revolve(2), run_time=20, rate_func=linear)


class Runner(ZoomedScene):
    def construct(self):
        fs = FourierSeries(
            FourierSeries.image("./examples/input/runner.jpg", greedy=True),
            number=200,
        )

        self.add(fs.mobject)

        fs.zoomed_display(self, scale_factor=1)

        self.play(fs.revolve(2), run_time=20, rate_func=linear)
