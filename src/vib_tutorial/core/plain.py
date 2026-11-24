"""A page's state as plain data, and the check that a view's output is plain (#478, page standard item 9).

A page rebuilt to #478's standard keeps its state, trials and problems in plain data, so a front end other than
Qt (a Rust UI, WebAssembly) could store and send them, and its view model outputs only what any front end can
draw. This module is how both are kept so:

- `to_plain(obj)` turns a state dataclass into JSON's types: a dataclass a dict of its fields, an enum its name,
  a tuple a list, an array a list. `from_plain(cls, data)` builds it back, guided by the class's type hints: an
  `X | Y | None` field (a trial's `Prediction | Placement | None`) takes the first dataclass whose fields the dict
  has. So `from_plain(type(s), json.loads(json.dumps(to_plain(s)))) == s` for a page's state `s`.
- `impure(obj)` lists the paths in a view model's output that aren't plain data: numbers, strings, bools, None,
  numeric arrays, enums, and tuples, lists, dicts and dataclasses of them. A colour is a theme role's name (a
  string), never a Qt colour; nothing callable.

A callable can't be data: a problem's `make` and a spec's `goal` are code, found again by the problem's key.

Working on it: `tests/test_plain.py` checks the conversions; `tests/test_readiness.py` holds each rebuilt page to
them (its `READY` list is where a page joins).
"""

from __future__ import annotations

import dataclasses
import enum
import types
import typing
from typing import Any

import numpy as np

_PRIMITIVE = (bool, int, float, str, type(None))


def to_plain(obj: Any) -> Any:
    """`obj` in JSON's types (see the module); raises TypeError on anything that isn't data."""
    if isinstance(obj, enum.Enum):
        return obj.name
    if isinstance(obj, _PRIMITIVE):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (tuple, list)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, dict):
        if not all(isinstance(k, str) for k in obj):
            raise TypeError(f"a dict's keys must be strings to be plain: {list(obj)}")
        return {k: to_plain(v) for k, v in obj.items()}
    raise TypeError(f"not plain data: {type(obj).__name__}")


def from_plain(hint: Any, data: Any) -> Any:
    """Build a value of type `hint` (a class or a type hint) back from `to_plain`'s output."""
    if hint is Any or hint is typing.Any:
        return data
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        return _from_union(args, data)
    if hint is type(None) or hint is None:
        if data is not None:
            raise ValueError(f"expected None, got {data!r}")
        return None
    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        return hint[data]
    if dataclasses.is_dataclass(hint) and isinstance(hint, type):
        return _from_dataclass(hint, data)
    if hint is np.ndarray or origin is np.ndarray:
        return np.asarray(data, dtype=float)
    if origin is tuple or hint is tuple:
        if not args:
            return tuple(data)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(from_plain(args[0], v) for v in data)
        return tuple(from_plain(a, v) for a, v in zip(args, data, strict=True))
    if origin is list or hint is list:
        return [from_plain(args[0], v) for v in data] if args else list(data)
    if origin is dict or hint is dict:
        return {k: from_plain(args[1], v) for k, v in data.items()} if args else dict(data)
    if hint is float:
        return float(data)  # JSON writes 5.0 as 5.0, but a hand-written 5 is an int
    if hint in (int, str, bool):
        if not isinstance(data, hint) or (hint is int and isinstance(data, bool)):
            raise ValueError(f"expected {hint.__name__}, got {data!r}")
        return data
    raise TypeError(f"no plain form for {hint!r}")


def _from_union(args: tuple, data: Any) -> Any:
    if data is None and type(None) in args:
        return None
    for a in args:
        if isinstance(a, type) and dataclasses.is_dataclass(a) and isinstance(data, dict):
            names = {f.name for f in dataclasses.fields(a)}
            required = {
                f.name
                for f in dataclasses.fields(a)
                if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
            }
            if required <= data.keys() <= names:
                return _from_dataclass(a, data)
    for a in args:
        if a is type(None) or dataclasses.is_dataclass(a):
            continue
        try:
            return from_plain(a, data)
        except (ValueError, TypeError, KeyError):
            continue
    raise ValueError(f"{data!r} fits none of {args}")


def _from_dataclass(cls: type, data: dict) -> Any:
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            kwargs[f.name] = from_plain(hints[f.name], data[f.name])
    if unknown := data.keys() - kwargs.keys():
        raise ValueError(f"{cls.__name__} has no fields {sorted(unknown)}")
    return cls(**kwargs)


def impure(obj: Any, path: str = "view") -> list[str]:
    """The paths in `obj` that aren't plain data (see the module), each with what it is; [] when all of it is."""
    if isinstance(obj, (*_PRIMITIVE, enum.Enum, np.number, np.bool_)):
        return []
    if isinstance(obj, np.ndarray):
        return [] if obj.dtype.kind in "biufc" else [f"{path}: an array of {obj.dtype}"]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return [p for f in dataclasses.fields(obj) for p in impure(getattr(obj, f.name), f"{path}.{f.name}")]
    if isinstance(obj, (tuple, list)):
        return [p for i, v in enumerate(obj) for p in impure(v, f"{path}[{i}]")]
    if isinstance(obj, dict):
        return [p for k, v in obj.items() for p in impure(v, f"{path}[{k!r}]")]
    return [f"{path}: {type(obj).__module__}.{type(obj).__qualname__}"]
