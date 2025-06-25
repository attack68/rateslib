from __future__ import annotations

from enum import Enum
from json import dumps
from typing import TYPE_CHECKING

from rateslib.default import NoInput
from rateslib.dual import Dual, Dual2
from rateslib.dual.utils import _to_number

if TYPE_CHECKING:
    from rateslib.typing import Any, DualTypes, Number  # pragma: no cover


# Dualtypes handles case of rust wrapped Dual/Dual2 datatype intermixed with float.


def _dualtypes_to_json(val: DualTypes) -> str:
    val_: Number = _to_number(val)
    if isinstance(val_, Dual | Dual2):
        return val_.to_json()
    else:
        return dumps(val_)


def _enum_to_json(val: Enum) -> str:
    return f'{{"PyNative":{{"{type(val).__name__}":{val.value}}}}}'


def _obj_to_json(val: Any) -> str:
    if isinstance(val, NoInput):
        return _enum_to_json(val)
    else:
        try:
            return val.to_json()  # type: ignore[no-any-return]
        except AttributeError:
            return dumps(val)
