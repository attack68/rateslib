# SPDX-License-Identifier: LicenseRef-Rateslib-Dual
#
# Copyright (c) 2026 Siffrorna Technology Limited
#
# Dual-licensed: Free Educational Licence or Paid Commercial Licence (commercial/professional use)
# Source-available, not open source.
#
# See LICENSE and https://rateslib.com/py/en/latest/i_licence.html for details,
# and/or contact info (at) rateslib (dot) com
####################################################################################################

from __future__ import annotations

from enum import Enum
from json import dumps
from typing import TYPE_CHECKING

from rateslib.dual import Dual, Dual2
from rateslib.dual.utils import _to_number
from rateslib.enums.generics import NoInput

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
