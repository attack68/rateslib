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

from datetime import datetime as dt

import pytest
from rateslib.rs import Imm


@pytest.mark.parametrize(
    ("date", "expected"),
    [
        (dt(2022, 3, 16), True),
        (dt(2022, 6, 15), True),
        (dt(2022, 9, 25), False),
        (dt(2022, 8, 17), False),
    ],
)
def test_is_imm(date, expected) -> None:
    result = Imm.Wed3_HMUZ.validate(date)
    assert result is expected


def test_is_imm_serial() -> None:
    result = Imm.Wed3.validate(dt(2022, 8, 17))  # imm in Aug
    assert result


@pytest.mark.parametrize(
    ("month", "year", "expected"),
    [
        (3, 2022, dt(2022, 3, 16)),
        (6, 2022, dt(2022, 6, 15)),
        (9, 2022, dt(2022, 9, 21)),
        (12, 2022, dt(2022, 12, 21)),
    ],
)
def test_get_imm(month, year, expected) -> None:
    result = Imm.Wed3.get(year, month)
    assert result == expected


def test_get_imm_namespace():
    from rateslib import get_imm as f

    f(code="h24")


@pytest.mark.parametrize(
    ("month", "year", "expected"),
    [
        (2, 2022, dt(2022, 2, 28)),
        (2, 2024, dt(2024, 2, 29)),
        (8, 2022, dt(2022, 8, 31)),
    ],
)
def test_get_eom(month, year, expected) -> None:
    result = Imm.Eom.get(year, month)
    assert result == expected
