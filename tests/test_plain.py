import enum
import json
from dataclasses import dataclass

import numpy as np
import pytest

from vib_tutorial.core.plain import from_plain, impure, to_plain
from vib_tutorial.core.practice import Placement, Prediction


class Kind(enum.Enum):
    SOFT = (1, "a tuple value: stored by name")
    HARD = 2


@dataclass(frozen=True)
class Inner:
    a: float
    b: tuple[int, ...] = ()


@dataclass(frozen=True)
class State:
    name: str
    x: float = 1.0
    kind: Kind = Kind.SOFT
    pair: tuple[float, str] = (0.5, "half")
    inner: Inner | None = None
    ask: Prediction | Placement | None = None
    options: dict[str, float] | None = None
    n: int = 3


def _json(obj):
    return json.loads(json.dumps(to_plain(obj)))


@pytest.mark.parametrize(
    "state",
    [
        State("bare"),
        State("all", 2.5, Kind.HARD, (1.5, "x"), Inner(3.0, (1, 2)), options={"k": 400.0}, n=7),
        State("predict", ask=Prediction("Which?", ("a", "b"), 1, why=("because", ""))),
        State("place", ask=Placement("Where?", "plot", 2.0, (0.0, 5.0), "low", "high", log=True)),
    ],
)
def test_a_state_survives_a_json_round_trip(state):
    assert from_plain(State, _json(state)) == state


def test_a_whole_number_comes_back_as_a_float_where_the_field_is_one():
    back = from_plain(State, {"name": "n", "x": 5})
    assert back.x == 5.0 and isinstance(back.x, float)


def test_unknown_fields_and_wrong_types_are_refused():
    with pytest.raises(ValueError, match="no fields"):
        from_plain(State, {"name": "n", "colour": "red"})
    with pytest.raises(ValueError):
        from_plain(State, {"name": 3})


def test_what_isnt_data_has_no_plain_form():
    with pytest.raises(TypeError):
        to_plain(State("f", options={"k": len}))  # a function
    with pytest.raises(TypeError):
        to_plain({1: "a non-string key"})


def test_impure_names_what_isnt_plain():
    @dataclass
    class View:
        curve: tuple[np.ndarray, np.ndarray]
        note: str
        colour: object

    good = View((np.arange(3.0), np.zeros(3)), "a readout", "mass")
    assert impure(good) == []
    bad = View((np.arange(3.0), np.array(["a"], dtype=object)), "x", lambda: 0)
    found = impure(bad)
    assert len(found) == 2
    assert found[0].startswith("view.curve[1]: an array of object")
    assert found[1].startswith("view.colour: ") and "function" in found[1]
