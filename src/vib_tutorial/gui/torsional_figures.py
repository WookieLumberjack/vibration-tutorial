"""The Torsional vibration page's Theory figures (see `figures`): the motor-and-fan preset's one mode, its twist
along the shaft with the node near the heavier disc; a gear mesh, its spring along the line of action between the
base circles; and the fan's induction motor's speed–torque curve (Kloss's) against the load, from the page's own
functions."""

from __future__ import annotations

import math

import numpy as np

from ..core.torsional import FAN, modes
from ..core.torsional_startup import FAN_START, drag, steady_speed
from . import figures as fig
from .style import colors


def _along(x1: float, y1: float, x2: float, y2: float, color: str) -> str:
    """A spring from (x1, y1) to (x2, y2), at any angle."""
    length = math.hypot(x2 - x1, y2 - y1)
    deg = math.degrees(math.atan2(y2 - y1, x2 - x1))
    body = fig.spring(0.0, length, 0.0, color, coils=4, amplitude=4.0)
    return f"<g transform='translate({x1:.1f} {y1:.1f}) rotate({deg:.1f})'>{body}</g>"


def node_fraction() -> float:
    """Where the fan preset's mode stands still along the coupling, from the motor: J₂/(J₁ + J₂)."""
    a, b = modes(FAN)[0].shape
    return float(a / (a - b))


def node_svg() -> str:
    """The motor (J₁) and the fan (J₂ = 5J₁) on the coupling, and the mode's twist above it: the motor's swing 1,
    the fan's −0.2, a straight line between (the shaft's own inertia neglected), through zero at 5/6 of the way."""
    fg, muted, mass, strong = colors.foreground, colors.muted, colors.mass[0], colors.strong
    a, b = modes(FAN)[0].shape
    x1, x2, axis = 50.0, 250.0, 118.0
    parts = [
        fig.line(x1, axis, x2, axis, fg, 3.0),
        fig.rect(x1 - 6, axis - 16, 12, 32, mass, mass, 1.5, opacity=0.35),
        fig.rect(x2 - 10, axis - 30, 20, 60, mass, mass, 1.5, opacity=0.35),
        fig.math_label(x1, axis + 32, "J<sub>1</sub>", fg),
        fig.math_label(x2, axis + 46, "J<sub>2</sub>", fg),
        fig.math_label((x1 + x2) / 2, axis + 20, "k", fg),
    ]
    zero, unit = 60.0, 36.0
    xn = x1 + node_fraction() * (x2 - x1)
    parts += [
        fig.line(x1 - 14, zero, x2 + 14, zero, muted, 1.0, dash="3,3"),
        fig.line(x1, zero - unit * a, x2, zero - unit * b, strong, 2.0),
        fig.line(x1, zero, x1, zero - unit * a, muted, 1.0),
        fig.line(x2, zero, x2, zero - unit * b, muted, 1.0),
        fig.circle(xn, zero, 3.5, strong, strong),
        fig.text(xn, zero - 10, "node", strong),
        fig.text(x1 - 6, zero - unit * a + 4, "1", fg, "end"),
        fig.text(x2 - 8, zero - unit * b + 14, "−0.2", fg, "end"),
        fig.text(x2 + 14, 22, "twist", muted, "end"),
    ]
    return fig.svg(302, 180, "".join(parts))


def mesh_svg() -> str:
    """A bull gear and a pinion as their base circles, the teeth a spring k_m on the line of action, the common
    tangent between the circles; each gear turning its own way, θ_in and θ_out."""
    fg, muted, mass, force = colors.foreground, colors.muted, colors.mass[0], colors.force
    cy, x1, x2 = 92.0, 86.0, 170.0
    pressure = math.radians(20.0)
    pitch1, pitch2 = 0.7 * (x2 - x1), 0.3 * (x2 - x1)
    r1, r2 = pitch1 * math.cos(pressure), pitch2 * math.cos(pressure)
    ux, uy = math.cos(pressure), -math.sin(pressure)
    p1 = (x1 + r1 * ux, cy + r1 * uy)
    p2 = (x2 - r2 * ux, cy - r2 * uy)
    parts = [
        fig.circle(x1, cy, pitch1, muted, "none", 1.0, dash="3,3"),
        fig.circle(x2, cy, pitch2, muted, "none", 1.0, dash="3,3"),
        fig.circle(x1, cy, r1, mass, mass, 1.5).replace("fill=", "fill-opacity='0.2' fill="),
        fig.circle(x2, cy, r2, mass, mass, 1.5).replace("fill=", "fill-opacity='0.2' fill="),
        fig.line(x1, cy, *p1, fg, 1.2),
        fig.line(x2, cy, *p2, fg, 1.2),
        fig.line(2 * p1[0] - p2[0], 2 * p1[1] - p2[1], 2 * p2[0] - p1[0], 2 * p2[1] - p1[1], force, 1.0, "4,3"),
        _along(*p1, *p2, force),
        fig.circle(x1, cy, 2.5, fg, fg),
        fig.circle(x2, cy, 2.5, fg, fg),
        fig.math_label(x1 + 0.5 * (p1[0] - x1) - 2, cy + 0.5 * (p1[1] - cy) - 8, "r<sub>in</sub>", fg),
        fig.math_label(x2 + 6, cy + 18, "r<sub>out</sub>", fg, "start"),
        fig.math_label((p1[0] + p2[0]) / 2 + 2, (p1[1] + p2[1]) / 2 - 40, "k<sub>m</sub>", force, "start"),
        fig.arc(x1, cy, r1 + 10, 200, 250, fg, 1.4, arrow=True),
        fig.math_label(x1 - 50, cy + 58, "θ<sub>in</sub>", fg),
        fig.arc(x2, cy, r2 + 10, 280, 340, fg, 1.4),
        fig.head(*fig.point(x2, cy, r2 + 10, 280), -math.radians(280) + math.pi / 2, fg),
        fig.math_label(x2 + 30, cy - 22, "θ<sub>out</sub>", fg, "start"),
        fig.text(2 * p2[0] - p1[0] + 6, 2 * p2[1] - p1[1] + 14, "line of action", force, "start"),
    ]
    return fig.svg(302, 170, "".join(parts))


def kloss_curve() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The fan's start: motor speed (rpm, 0 to synchronous), Kloss's torque, and the train's drag (N·m)."""
    motor = FAN_START.motor
    ws = motor.sync_speed
    w = np.linspace(0.0, ws, 151)
    torque = np.array([motor.cage(1.0 - v / ws)[0] for v in w])
    load = np.array([drag(FAN, FAN_START, v) for v in w])
    return 60.0 * w / (2 * math.pi), torque, load


def kloss_svg() -> str:
    """Torque against the motor's speed: Kloss's curve (615 N·m at rest, the breakdown 1600 N·m at s_b = 0.2,
    nothing at synchronism) and the fan's drag rising as the speed squared; they meet where it runs, 1444 rpm."""
    fg, muted, force, strong = colors.foreground, colors.muted, colors.force, colors.strong
    rpm, torque, load = kloss_curve()
    x0, y0, w, h = 44.0, 140.0, 236.0, 112.0
    top = 1800.0

    def at(r: float, t: float) -> tuple[float, float]:
        return x0 + w * r / rpm[-1], y0 - h * t / top

    ws = steady_speed(FAN, FAN_START)
    run_rpm = 60.0 * ws / (2 * math.pi)
    run_t = drag(FAN, FAN_START, ws)
    motor = FAN_START.motor
    bx, by = at(motor.sync_rpm * (1 - motor.breakdown_slip), motor.breakdown)
    rx, ry = at(run_rpm, run_t)
    parts = [
        fig.arrow(x0, y0, x0 + w + 12, y0, muted, 1.0),
        fig.arrow(x0, y0, x0, y0 - h - 14, muted, 1.0),
        fig.curve([at(r, t) for r, t in zip(rpm, torque, strict=True)], strong, 2.0),
        fig.curve([at(r, t) for r, t in zip(rpm, load, strict=True)], force, 2.0),
        fig.circle(bx, by, 3.0, strong, strong),
        fig.circle(rx, ry, 3.5, fg, fg),
        fig.text(bx - 6, by - 8, "breakdown", strong, "end"),
        fig.text(x0 + 6, at(0, torque[0])[1] - 8, "motor", strong, "start"),
        fig.text(at(900, 0)[0], at(0, float(np.interp(900, rpm, load)))[1] + 18, "load", force),
        fig.text(rx - 8, ry - 8, "runs", fg, "end"),
        fig.math_label(x0 - 6, y0 - h - 4, "T", fg, "end"),
        fig.math_label(x0 + w + 12, y0 - 8, "Ω", fg, "end"),
        fig.math_label(x0 + w, y0 + 18, "Ω<sub>s</sub>", muted),
        fig.text(x0, y0 + 16, "0", muted),
    ]
    return fig.svg(302, 166, "".join(parts))


FIGURES = {"fig:node": node_svg, "fig:mesh": mesh_svg, "fig:kloss": kloss_svg}
