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

import numpy as np
import pytest
from pandas import DataFrame, DatetimeIndex, date_range
from pandas.testing import assert_index_equal
from pandas.tseries.holiday import Holiday
from rateslib import defaults
from rateslib.default import NoInput
from rateslib.rs import Adjuster, Frequency, RollDay
from rateslib.scheduling import Cal
from rateslib.scheduling.schedule import Schedule


@pytest.fixture
def cal_():
    return Cal([dt(_, 1, 3) for _ in range(1970, 2200)], [5, 6])


@pytest.mark.parametrize(
    ("dt1", "dt2", "fm", "expected"),
    [
        (dt(2022, 3, 16), dt(2022, 6, 30), 3, False),
        (dt(2022, 3, 16), dt(2024, 9, 16), 3, True),
        (dt(2022, 3, 16), dt(2028, 9, 16), 6, True),
        (dt(2022, 3, 16), dt(2029, 3, 16), 12, True),
        (dt(2022, 3, 16), dt(2022, 10, 16), 3, False),
        (dt(2022, 3, 31), dt(2024, 4, 1), 12, False),
    ],
)
def test_is_divisible_months(dt1, dt2, fm, expected) -> None:
    f = Frequency.Months(fm, RollDay.Day(16))
    try:
        f.uregular(dt1, dt2)
    except ValueError:
        assert not expected
    else:
        assert expected


@pytest.mark.parametrize(
    ("effective", "termination", "expected", "expected2"),
    [
        (dt(2022, 2, 22), dt(2024, 2, 22), 22, 22),
        (dt(2022, 2, 22), dt(2024, 2, 15), 15, 15),
        (dt(2022, 2, 28), dt(2024, 2, 29), 29, 31),
        (dt(2022, 6, 30), dt(2024, 9, 30), 30, 31),
        (dt(2022, 6, 30), dt(2024, 12, 30), 30, 30),
        (dt(2022, 2, 28), dt(2024, 9, 30), 30, 31),
        (dt(2024, 3, 31), dt(2024, 9, 30), 31, 31),
    ],
)
def test_get_unspecified_roll(effective, termination, expected, expected2) -> None:
    result = Schedule(
        effective,
        termination,
        Frequency.Months(1, None),
        eom=False,
    )
    assert result.frequency_obj.roll == RollDay.Day(expected)

    result = Schedule(
        effective,
        termination,
        Frequency.Months(1, None),
        eom=True,
    )
    assert result.frequency_obj.roll == RollDay.Day(expected2)


@pytest.mark.parametrize(
    ("e", "t", "stub", "exp_roll", "exp_stub"),
    [
        (dt(2022, 2, 26), dt(2024, 4, 22), "SHORTFRONT", 22, dt(2022, 4, 22)),
        (dt(2022, 2, 26), dt(2024, 4, 22), "LONGFRONT", 22, dt(2022, 7, 22)),
        (dt(2022, 2, 26), dt(2024, 4, 22), "SHORTBACK", 26, dt(2024, 2, 26)),
        (dt(2022, 2, 26), dt(2024, 4, 22), "LONGBACK", 26, dt(2023, 11, 26)),
    ],
)
def test_infer_stub_date(e, t, stub, exp_roll, exp_stub, cal_) -> None:
    result = Schedule(
        e,
        t,
        "Q",
        eom=False,
        stub=stub,
        calendar=cal_,
    )
    if "FRONT" in stub:
        assert result.ufront_stub == exp_stub
        assert result.roll == exp_roll
    else:
        assert result.uback_stub == exp_stub
        assert result.roll == exp_roll


@pytest.mark.parametrize(
    ("e", "t", "stub", "exp_roll", "exp_stub"),
    [
        (dt(2022, 2, 26), dt(2024, 2, 26), "SHORTFRONT", 26, NoInput(0)),
        (dt(2022, 2, 26), dt(2024, 2, 26), "LONGFRONT", 26, NoInput(0)),
        (dt(2022, 2, 26), dt(2024, 2, 26), "SHORTBACK", 26, NoInput(0)),
        (dt(2022, 2, 26), dt(2024, 2, 26), "LONGBACK", 26, NoInput(0)),
    ],
)
def test_infer_stub_date_no_inference_on_regular(e, t, stub, exp_roll, exp_stub, cal_) -> None:
    result = Schedule(
        e,
        t,
        "Q",
        stub=stub,
        eom=False,
        calendar=cal_,
    )
    assert result.is_regular()


def test_infer_stub_date_no_inference_on_regular_dual(cal_) -> None:
    result = Schedule(
        dt(2022, 2, 26),
        dt(2024, 4, 26),
        "Q",
        stub="SHORTFRONT",
        front_stub=NoInput(0),
        back_stub=dt(2024, 2, 26),
        calendar=cal_,
    )
    assert result.ufront_stub is None
    assert result.roll == 26

    result = Schedule(
        dt(2022, 2, 26),
        dt(2024, 4, 26),
        "Q",
        stub="SHORTBACK",
        front_stub=dt(2022, 4, 26),
        back_stub=NoInput(0),
        calendar=cal_,
    )
    assert result.uback_stub is None
    assert result.roll == 26


@pytest.mark.parametrize(
    ("e", "t", "stub"),
    [
        (dt(2022, 2, 26), dt(2024, 4, 22), "SHORTFRONT"),
        (dt(2022, 2, 26), dt(2024, 4, 22), "LONGFRONT"),
        (dt(2022, 2, 26), dt(2024, 4, 22), "SHORTBACK"),
        (dt(2022, 2, 26), dt(2024, 4, 22), "LONGBACK"),
    ],
)
def test_infer_stub_date_invalid_roll(e, t, stub, cal_) -> None:
    with pytest.raises(ValueError, match="A Schedule could not be generated from"):
        Schedule(e, t, "Q", stub=stub, roll=14, calendar=cal_)


@pytest.mark.parametrize(
    ("e", "fs", "t", "stub", "exp_roll", "exp_stub"),
    [
        (dt(2022, 1, 1), dt(2022, 2, 26), dt(2024, 4, 26), "FRONTSHORTBACK", 26, dt(2024, 2, 26)),
        (dt(2022, 1, 1), dt(2022, 2, 26), dt(2024, 4, 26), "FRONTLONGBACK", 26, dt(2023, 11, 26)),
    ],
)
def test_infer_stub_date_dual_sided(e, fs, t, stub, exp_roll, exp_stub, cal_) -> None:
    result = Schedule(e, t, "Q", stub=stub, front_stub=fs, calendar=cal_)
    assert result.ueffective == e
    assert result.uback_stub == exp_stub
    assert result.utermination == t
    assert result.roll == exp_roll


@pytest.mark.parametrize(
    ("e", "bs", "t", "stub", "exp_roll", "exp_stub"),
    [
        (dt(2022, 1, 1), dt(2024, 2, 26), dt(2024, 4, 26), "SHORTFRONT", 26, dt(2022, 2, 26)),
        (dt(2022, 1, 1), dt(2024, 2, 26), dt(2024, 4, 26), "LONGFRONT", 26, dt(2022, 5, 26)),
    ],
)
def test_infer_stub_date_dual_sided2(e, bs, t, stub, exp_roll, exp_stub, cal_) -> None:
    result = Schedule(e, t, "Q", stub=stub, back_stub=bs, calendar=cal_)
    assert result.ueffective == e
    assert result.ufront_stub == exp_stub
    assert result.uback_stub == bs
    assert result.utermination == t
    assert result.roll == exp_roll


def test_infer_stub_date_dual_sided_invalid(cal_) -> None:
    with pytest.raises(ValueError, match="A Schedule could not be generated from"):
        Schedule(
            dt(2022, 1, 1),
            dt(2022, 12, 31),
            "Q",
            stub="FRONTSHORT",
            front_stub=dt(2022, 2, 13),
            calendar=cal_,
        )


def test_infer_stub_date_eom(cal_) -> None:
    result = Schedule(
        dt(2022, 1, 1),
        dt(2023, 2, 28),
        "Q",
        stub="LONGFRONT",
        eom=True,  # <- the EOM parameter forces the stub to be 31 May and not 28 May
        calendar=cal_,
    )
    assert result.ufront_stub == dt(2022, 5, 31)


def test_repr():
    schedule = Schedule(
        dt(2022, 1, 1),
        "2M",
        "M",
    )
    expected = f"<rl.Schedule at {hex(id(schedule))}>"
    assert expected == schedule.__repr__()


def test_schedule_str(cal_) -> None:
    schedule = Schedule(dt(2022, 1, 1), "2M", "M", eom=False, calendar=cal_, roll=1, payment_lag=1)
    expected = "freq: 1M (roll: 1), accrual adjuster: MF, payment adjuster: 1B,\n"
    df = DataFrame(
        {
            defaults.headers["stub_type"]: ["Regular", "Regular"],
            defaults.headers["u_acc_start"]: [dt(2022, 1, 1), dt(2022, 2, 1)],
            defaults.headers["u_acc_end"]: [dt(2022, 2, 1), dt(2022, 3, 1)],
            defaults.headers["a_acc_start"]: [dt(2022, 1, 4), dt(2022, 2, 1)],
            defaults.headers["a_acc_end"]: [dt(2022, 2, 1), dt(2022, 3, 1)],
            defaults.headers["payment"]: [dt(2022, 2, 2), dt(2022, 3, 2)],
        },
    )
    result = schedule.__str__()
    assert result == expected + df.__repr__()


def test_schedule_raises(cal_) -> None:
    with pytest.raises(ValueError, match="Frequency can not be determined from `frequency` input."):
        _ = Schedule(dt(2022, 1, 1), dt(2022, 12, 31), "Unknown")

    with pytest.raises(ValueError, match="`termination` must be after"):
        _ = Schedule(dt(2022, 1, 1), dt(2021, 12, 31), "Q")

    with pytest.raises(ValueError):
        _ = Schedule(
            dt(2022, 1, 1),
            dt(2022, 12, 31),
            "Q",
            stub="SHORTFRONT",
            front_stub=None,
            back_stub=dt(2022, 11, 15),
            eom=False,
            modifier="MF",
            calendar=cal_,
            roll=1,
        )

    with pytest.raises(ValueError):
        _ = Schedule(
            dt(2022, 1, 1),
            dt(2022, 12, 31),
            "Q",
            stub="SHORTBACK",
            front_stub=dt(2022, 3, 15),
            eom=False,
            calendar=cal_,
            roll=1,
        )

    with pytest.raises(ValueError):
        _ = Schedule(
            dt(2022, 1, 1),
            dt(2022, 12, 31),
            "Q",
            stub="SBLB",
            front_stub=dt(2022, 3, 15),
            eom=False,
            calendar=cal_,
            roll=1,
        )


@pytest.mark.parametrize(
    ("eff", "term", "f", "roll"),
    [
        (dt(2022, 3, 16), dt(2024, 9, 10), "Q", "imm"),  # non-imm term
        (dt(2022, 3, 31), dt(2023, 3, 30), "A", "eom"),  # non-eom term
        (dt(2022, 3, 1), dt(2023, 3, 2), "A", "som"),  # non-som term
        (dt(2022, 2, 20), dt(2025, 8, 21), "S", 20),  # roll
        (dt(2022, 2, 28), dt(2024, 2, 28), "S", 30),  # is leap
    ],
)
def test_unadjusted_regular_swap_dead_stubs(eff, term, f, roll) -> None:
    with pytest.raises(ValueError, match="A Schedule could not be generated from the parameter c"):
        Schedule(eff, term, f, eom=False, roll=roll)


@pytest.mark.parametrize(
    ("eff", "term", "f", "roll", "exp"),
    [
        (dt(2022, 3, 16), dt(2022, 6, 30), "S", NoInput(0), False),  # frequency
        (dt(2022, 3, 15), dt(2022, 9, 21), "Q", "imm", False),  # non-imm eff
        (dt(2022, 3, 30), dt(2029, 3, 31), "A", "eom", False),  # non-eom eff
        (dt(2022, 3, 2), dt(2029, 3, 1), "A", "som", False),  # non-som eff
        (dt(2022, 3, 30), dt(2023, 9, 30), "S", 31, False),  # non-eom
        (dt(2024, 2, 28), dt(2025, 8, 30), "S", 30, False),  # is leap
        (dt(2024, 2, 29), dt(2025, 8, 30), "S", 30, True),  # is leap
        (dt(2022, 2, 28), dt(2025, 8, 29), "S", 29, True),  # is end feb
        (dt(2022, 2, 20), dt(2025, 8, 20), "S", 20, True),  # OK
        (dt(2022, 2, 21), dt(2025, 8, 20), "S", 20, False),  # roll
        (dt(2022, 2, 22), dt(2024, 2, 15), "S", NoInput(0), False),  # no valid roll
        (dt(2022, 2, 28), dt(2024, 2, 29), "S", NoInput(0), True),  # 29 or eom
        (dt(2022, 6, 30), dt(2024, 12, 30), "S", NoInput(0), True),  # 30
    ],
)
def test_unadjusted_regular_swap(eff, term, f, roll, exp) -> None:
    result = Schedule(eff, term, f, eom=False, roll=roll)
    assert result.is_regular() is exp


# 12th and 13th of Feb and March are Saturday and Sunday
@pytest.mark.parametrize(
    ("eff", "term", "roll", "e_bool", "e_ueff", "e_uterm", "e_roll"),
    [
        (dt(2022, 2, 11), dt(2022, 3, 11), 11, True, dt(2022, 2, 11), dt(2022, 3, 11), 11),
        (dt(2022, 2, 14), dt(2022, 3, 14), 14, True, dt(2022, 2, 14), dt(2022, 3, 14), 14),
        (dt(2022, 2, 14), dt(2022, 3, 14), NoInput(0), True, dt(2022, 2, 14), dt(2022, 3, 14), 14),
        (dt(2022, 2, 13), dt(2022, 3, 14), NoInput(0), True, dt(2022, 2, 13), dt(2022, 3, 13), 13),
        (dt(2022, 2, 12), dt(2022, 3, 14), NoInput(0), True, dt(2022, 2, 12), dt(2022, 3, 12), 12),
        (dt(2022, 2, 12), dt(2022, 3, 13), NoInput(0), False, None, None, None),
        (dt(2022, 2, 14), dt(2022, 3, 12), NoInput(0), True, dt(2022, 2, 12), dt(2022, 3, 12), 12),
        (dt(2022, 2, 14), dt(2022, 3, 14), 12, True, dt(2022, 2, 12), dt(2022, 3, 12), 12),
        (dt(2022, 2, 14), dt(2022, 3, 14), 11, False, None, None, None),
        (dt(2022, 2, 28), dt(2022, 3, 31), NoInput(0), True, dt(2022, 2, 28), dt(2022, 3, 31), 31),
        (dt(2022, 2, 28), dt(2022, 3, 31), 28, False, None, None, None),
        (dt(2022, 2, 28), dt(2022, 3, 31), "eom", True, dt(2022, 2, 28), dt(2022, 3, 31), 31),
    ],
)
def test_check_regular_swap_mf(eff, term, roll, e_bool, e_ueff, e_uterm, e_roll, cal_) -> None:
    try:
        result = Schedule(eff, term, "M", modifier="MF", eom=False, roll=roll, calendar=cal_)
    except ValueError:
        assert not e_bool
    else:
        assert result.ueffective == e_ueff
        assert result.utermination == e_uterm
        assert result.roll == e_roll


@pytest.mark.parametrize(
    ("effective", "termination", "uf", "ub", "roll", "expected"),
    [
        (
            dt(2023, 2, 4),
            dt(2023, 9, 4),
            dt(2023, 3, 4),
            NoInput(0),
            4,
            [dt(2023, 2, 4), dt(2023, 3, 4), dt(2023, 6, 4), dt(2023, 9, 4)],
        ),
        (
            dt(2023, 2, 4),
            dt(2023, 9, 4),
            NoInput(0),
            dt(2023, 8, 4),
            4,
            [dt(2023, 2, 4), dt(2023, 5, 4), dt(2023, 8, 4), dt(2023, 9, 4)],
        ),
        (
            dt(2023, 3, 4),
            dt(2023, 9, 4),
            NoInput(0),
            NoInput(0),
            4,
            [dt(2023, 3, 4), dt(2023, 6, 4), dt(2023, 9, 4)],
        ),
        (
            dt(2023, 2, 4),
            dt(2023, 10, 4),
            dt(2023, 3, 4),
            dt(2023, 9, 4),
            4,
            [dt(2023, 2, 4), dt(2023, 3, 4), dt(2023, 6, 4), dt(2023, 9, 4), dt(2023, 10, 4)],
        ),
    ],
)
def test_generate_irregular_uschedule(effective, termination, uf, ub, roll, expected) -> None:
    result = Schedule(effective, termination, "Q", roll=roll, front_stub=uf, back_stub=ub)
    assert result.uschedule == expected


@pytest.mark.parametrize(
    ("effective", "termination", "roll", "expected"),
    [
        (dt(2023, 3, 4), dt(2023, 9, 4), 4, [dt(2023, 3, 4), dt(2023, 6, 4), dt(2023, 9, 4)]),
        (dt(2023, 3, 6), dt(2023, 9, 6), 6, [dt(2023, 3, 6), dt(2023, 6, 6), dt(2023, 9, 6)]),
        (
            dt(2023, 4, 30),
            dt(2023, 10, 31),
            31,
            [dt(2023, 4, 30), dt(2023, 7, 31), dt(2023, 10, 31)],
        ),
        (
            dt(2022, 2, 28),
            dt(2022, 8, 31),
            "eom",
            [dt(2022, 2, 28), dt(2022, 5, 31), dt(2022, 8, 31)],
        ),
        (
            dt(2021, 11, 30),
            dt(2022, 5, 31),
            31,
            [dt(2021, 11, 30), dt(2022, 2, 28), dt(2022, 5, 31)],
        ),
        (
            dt(2023, 4, 30),
            dt(2023, 10, 30),
            30,
            [dt(2023, 4, 30), dt(2023, 7, 30), dt(2023, 10, 30)],
        ),
        (
            dt(2022, 3, 16),
            dt(2022, 9, 21),
            "imm",
            [dt(2022, 3, 16), dt(2022, 6, 15), dt(2022, 9, 21)],
        ),
        (dt(2022, 12, 1), dt(2023, 6, 1), "som", [dt(2022, 12, 1), dt(2023, 3, 1), dt(2023, 6, 1)]),
    ],
)
def test_generate_regular_uschedule(effective, termination, roll, expected) -> None:
    result = Schedule(effective, termination, "Q", roll=roll)
    assert result.uschedule == expected


@pytest.mark.parametrize(
    ("effective", "termination", "frequency", "expected"),
    [
        (dt(2022, 2, 15), dt(2022, 8, 15), "M", 6),
        (dt(2022, 2, 15), dt(2022, 8, 15), "Q", 2),
        (dt(2022, 2, 15), dt(2032, 2, 15), "Q", 40),
        (dt(2022, 2, 15), dt(2032, 2, 15), "Z", 1),
    ],
)
def test_regular_n_periods(effective, termination, frequency, expected) -> None:
    result = Schedule(effective, termination, frequency)
    assert result.n_periods == expected


@pytest.mark.parametrize(
    ("eff", "term", "freq", "ss", "eom", "roll", "expected"),
    [
        (dt(2022, 1, 1), dt(2023, 2, 15), "M", "SHORTFRONT", False, NoInput(0), dt(2022, 1, 15)),
        (dt(2022, 1, 1), dt(2023, 2, 15), "Q", "SHORTFRONT", False, NoInput(0), dt(2022, 2, 15)),
        (dt(2022, 1, 1), dt(2023, 2, 15), "S", "SHORTFRONT", False, NoInput(0), dt(2022, 2, 15)),
        (dt(2022, 2, 15), dt(2023, 2, 1), "S", "SHORTFRONT", False, NoInput(0), dt(2022, 8, 1)),
        (dt(2022, 1, 1), dt(2023, 2, 15), "M", "SHORTBACK", False, NoInput(0), dt(2023, 2, 1)),
        (dt(2022, 1, 1), dt(2023, 2, 15), "Q", "SHORTBACK", False, NoInput(0), dt(2023, 1, 1)),
        (dt(2022, 1, 1), dt(2023, 2, 15), "S", "SHORTBACK", False, NoInput(0), dt(2023, 1, 1)),
        (dt(2022, 2, 15), dt(2023, 2, 1), "S", "SHORTBACK", False, NoInput(0), dt(2022, 8, 15)),
        (dt(2022, 1, 1), dt(2023, 2, 28), "M", "SHORTFRONT", True, NoInput(0), dt(2022, 1, 31)),
        (dt(2022, 3, 1), dt(2023, 2, 28), "Q", "SHORTFRONT", True, NoInput(0), dt(2022, 5, 31)),
        (dt(2022, 3, 1), dt(2023, 2, 17), "Q", "SHORTFRONT", False, 17, dt(2022, 5, 17)),
    ],
)
def test_get_unadjusted_short_stub_date(eff, term, freq, ss, eom, roll, expected) -> None:
    result = Schedule(eff, term, freq, stub=ss, eom=eom, roll=roll)
    if ss == "SHORTFRONT":
        assert result.ufront_stub == expected
    else:
        assert result.uback_stub == expected


@pytest.mark.parametrize(
    ("eff", "term", "freq", "stub", "eom", "roll", "expected"),
    [
        (dt(2022, 1, 1), dt(2023, 2, 15), "M", "LONGFRONT", False, NoInput(0), dt(2022, 2, 15)),
        (dt(2022, 1, 1), dt(2023, 2, 15), "Q", "LONGFRONT", False, NoInput(0), dt(2022, 5, 15)),
        (dt(2022, 1, 1), dt(2023, 2, 15), "S", "LONGFRONT", False, NoInput(0), dt(2022, 8, 15)),
        (dt(2022, 2, 15), dt(2024, 2, 1), "S", "LONGFRONT", False, NoInput(0), dt(2023, 2, 1)),
        (dt(2022, 1, 1), dt(2023, 2, 15), "M", "LONGBACK", False, NoInput(0), dt(2023, 1, 1)),
        (dt(2022, 1, 1), dt(2023, 2, 15), "Q", "LONGBACK", False, NoInput(0), dt(2022, 10, 1)),
        (dt(2022, 1, 1), dt(2023, 2, 15), "S", "LONGBACK", False, NoInput(0), dt(2022, 7, 1)),
        (dt(2022, 2, 15), dt(2024, 2, 1), "S", "LONGBACK", False, NoInput(0), dt(2023, 2, 15)),
        (dt(2022, 1, 1), dt(2023, 2, 28), "M", "LONGFRONT", True, NoInput(0), dt(2022, 2, 28)),
        (dt(2022, 3, 1), dt(2023, 2, 28), "Q", "LONGFRONT", True, NoInput(0), dt(2022, 8, 31)),
        (dt(2022, 3, 1), dt(2023, 2, 17), "Q", "LONGFRONT", False, 17, dt(2022, 8, 17)),
        (dt(2022, 4, 30), dt(2023, 2, 18), "Q", "LONGBACK", True, NoInput(0), dt(2022, 10, 31)),
    ],
)
def test_get_unadjusted_stub_date_long(eff, term, freq, stub, eom, roll, expected) -> None:
    result = Schedule(eff, term, freq, stub=stub, eom=eom, roll=roll)
    if stub == "LONGFRONT":
        assert result.ufront_stub == expected
    else:
        assert result.uback_stub == expected


@pytest.mark.parametrize(
    ("e", "t", "r", "exp_roll", "exp_ue", "exp_ut"),
    [
        (
            dt(2020, 8, 31),
            dt(2021, 2, 26),
            NoInput(0),
            31,
            dt(2020, 8, 31),
            dt(2021, 2, 28),
        ),
        (
            dt(2021, 2, 26),
            dt(2021, 8, 31),
            NoInput(0),
            31,
            dt(2021, 2, 28),
            dt(2021, 8, 31),
        ),
        (dt(2021, 2, 26), dt(2021, 8, 30), 29, 29, dt(2021, 2, 28), dt(2021, 8, 29)),
    ],
)
def test_schedule_eom(e, t, r, exp_roll, exp_ue, exp_ut, cal_) -> None:
    sched = Schedule(e, t, "S", roll=r, modifier="MF", calendar=cal_)
    assert sched.ueffective == exp_ue
    assert sched.utermination == exp_ut
    assert sched.roll == exp_roll


def test_payment_lag_is_business_days() -> None:
    sched = Schedule(dt(2022, 11, 16), "1M", "M", modifier="MF", calendar="ldn")
    assert sched.pschedule[1] == dt(2022, 12, 20)
    # not 19th Dec which is adjusted(16th Dec + 2 days)


def test_schedule_bad_stub_combinations_raise() -> None:
    with pytest.raises(ValueError, match="Must supply at least one stub date"):
        _ = Schedule(
            effective=dt(2022, 1, 1),
            termination=dt(2023, 1, 1),
            frequency="S",
            stub="SHORTFRONTSHORTBACK",
        )


@pytest.mark.skip(reason="StubInference enum behaves differently to versions <= 2.0")
def test_schedule_bad_stub_combinations_raise2() -> None:
    with pytest.raises(ValueError, match="`stub` is only front sided but `back_stub` given"):
        _ = Schedule(
            effective=dt(2022, 1, 1),
            termination=dt(2023, 1, 1),
            frequency="S",
            stub="FRONT",
            front_stub=dt(2022, 2, 1),
            back_stub=dt(2022, 12, 1),
        )


@pytest.mark.parametrize(
    ("st", "fs", "bs"),
    [
        ("SHORTFRONTSHORTBACK", NoInput(0), dt(2023, 1, 1)),
        ("SHORTFRONTLONGBACK", dt(2022, 2, 1), NoInput(0)),
        ("SHORTFRONTSHORTBACK", dt(2022, 4, 15), dt(2022, 10, 15)),
        ("SHORTFRONT", NoInput(0), NoInput(0)),
        ("SHORTFRONT", dt(2022, 2, 1), NoInput(0)),
        ("SHORTBACK", NoInput(0), dt(2023, 1, 1)),
        ("SHORTBACK", NoInput(0), NoInput(0)),
    ],
)
def test_schedule_combinations_valid(st, fs, bs) -> None:
    Schedule(
        effective=dt(2022, 1, 1),
        termination=dt(2023, 2, 1),
        frequency="S",
        stub=st,
        back_stub=bs,
        front_stub=fs,
    )


@pytest.mark.parametrize(
    ("st", "fs", "bs", "roll"),
    [
        ("FRONTBACK", NoInput(0), dt(2023, 1, 15), 20),
        ("FRONTBACK", dt(2022, 2, 1), NoInput(0), 20),
        ("FRONTBACK", dt(2022, 4, 15), dt(2023, 11, 25), NoInput(0)),
        ("FRONT", NoInput(0), NoInput(0), 20),
        ("FRONT", dt(2022, 3, 12), NoInput(0), 20),
        ("BACK", NoInput(0), dt(2022, 12, 5), 20),
        ("BACK", NoInput(0), NoInput(0), 20),
    ],
)
def test_schedule_combinations_invalid(st, fs, bs, roll) -> None:
    with pytest.raises(ValueError, match="A Schedule could not be generated from the parameter co"):
        Schedule(
            effective=dt(2022, 1, 1),
            termination=dt(2023, 2, 1),
            frequency="S",
            stub=st,
            back_stub=bs,
            front_stub=fs,
            roll=roll,
        )


def test_schedule_n_periods() -> None:
    result = Schedule(
        effective=dt(2022, 1, 1),
        termination=dt(2023, 2, 1),
        frequency="S",
        stub="SHORTFRONT",
    )
    assert result.n_periods == 3


@pytest.mark.parametrize(
    ("ue", "ut", "exp"),
    [
        (dt(2023, 3, 17), dt(2023, 12, 20), dt(2023, 9, 20)),
        (dt(2022, 12, 19), dt(2023, 12, 20), dt(2023, 3, 15)),
    ],  # PR #9
)
def test_get_unadjusted_long_stub_imm(ue, ut, exp) -> None:
    result = Schedule(ue, ut, "Q", stub="LONGFRONT", eom=False, roll="imm")
    assert result.ufront_stub == exp


@pytest.mark.parametrize(
    ("ue", "ut"),
    [
        (dt(2023, 3, 15), dt(2023, 12, 20)),
    ],
)
def test_get_unadjusted_short_stub_imm(ue, ut) -> None:
    result = Schedule(ue, ut, "Q", stub="SHORTFRONT", eom=False)
    assert result.is_regular()
    assert result.roll == "IMM"


def test_dead_stubs() -> None:
    # this was a bug detected in performance testing which generated a 1d invalid stub.
    # this failed originally because a 1D stub between Sun 2nd May 27 and Mon 3rd May 27
    # was invalid since the adjusted accrual schedule modified the sunday to be
    # equal to the Monday giving a 0 day period.
    s = Schedule(
        dt(2027, 5, 2),
        dt(2046, 5, 3),
        "A",
        stub="LONGFRONT",
        calendar="bus",
    )
    assert s.uschedule[0:2] == [dt(2027, 5, 2), dt(2028, 5, 3)]
    assert s.aschedule[0:2] == [dt(2027, 5, 3), dt(2028, 5, 3)]

    # manipulate this test to cover the case for dual sided stubs
    s = Schedule(
        dt(2027, 5, 2),
        dt(2046, 6, 3),
        "A",
        stub="LONGFRONTSHORTBACK",
        back_stub=dt(2046, 5, 3),  # back stub means front stub is inferred
        calendar="bus",
    )
    assert s.uschedule[0:2] == [dt(2027, 5, 2), dt(2028, 5, 3)]
    assert s.aschedule[0:2] == [dt(2027, 5, 3), dt(2028, 5, 3)]

    # this was a bug detected in performance testing which generated a 1d invalid stub.
    # this failed originally because the ueffective date of Sat 20-dec-25 and the
    # inferred front stub of Sun 21-dec-25 both adjusted forwards to 22-dec-25
    # giving a 0 day period.
    s = Schedule(
        dt(2025, 12, 20),
        dt(2069, 12, 21),
        "A",
        stub="LONGFRONT",
        calendar="bus",
    )
    assert s.uschedule[0:2] == [dt(2025, 12, 20), dt(2026, 12, 21)]
    assert s.aschedule[0:2] == [dt(2025, 12, 22), dt(2026, 12, 21)]

    # this was a bug detected in performance testing which generated a 1d invalid stub.
    # this failed originally because the utermination date of Sat 20-dec-25 and the
    # inferred front stub of Sun 21-dec-25 both adjusted forwards to 22-dec-25
    # giving a 0 day period.
    s = Schedule(
        dt(2027, 10, 19),
        dt(2047, 10, 20),
        "A",
        stub="LONGBACK",
        calendar="bus",
    )
    assert s.uschedule[-2:] == [dt(2046, 10, 19), dt(2047, 10, 20)]
    assert s.aschedule[-2:] == [dt(2046, 10, 19), dt(2047, 10, 21)]

    # manipulate this test for dual sided stubs
    s = Schedule(
        dt(2027, 8, 19),
        dt(2047, 10, 20),
        "A",
        stub="SHORTFRONTLONGBACK",
        front_stub=dt(2027, 10, 19),
        calendar="bus",
    )
    assert s.uschedule[-2:] == [dt(2046, 10, 19), dt(2047, 10, 20)]
    assert s.aschedule[-2:] == [dt(2046, 10, 19), dt(2047, 10, 21)]


@pytest.mark.parametrize(
    ("mode", "end", "roll"),
    [
        (NoInput(0), dt(2025, 8, 17), 17),
        ("swaps_align", dt(2025, 8, 17), 17),
        ("swaptions_align", dt(2025, 8, 19), 19),
    ],
)
def test_eval_mode(mode, end, roll) -> None:
    sch = Schedule(
        effective="1Y",
        termination="1Y",
        frequency="S",
        calendar="tgt",
        eval_date=dt(2023, 8, 17),
        eval_mode=mode,
    )
    assert sch.roll == roll
    assert sch.utermination == end


def test_eval_date_raises() -> None:
    with pytest.raises(ValueError, match="For `effective` given as string tenor, must"):
        Schedule(
            effective="1Y",
            termination="1Y",
            frequency="S",
        )


def test_single_period_imm_roll():
    s = Schedule(
        effective=dt(2024, 12, 18),
        termination=dt(2025, 3, 19),
        roll="imm",
        frequency="a",
        calendar="stk",
    )
    assert len(s.aschedule) == 2


def test_deviate_from_effective_in_inference() -> None:
    # 28th and 30th are both valid rolls for this schedule
    # test that 30th is inferred since it deviates the least from effective input.
    s = Schedule(
        effective=dt(2024, 12, 30),
        termination=dt(2025, 11, 28),
        frequency="m",
        eom=False,
        calendar="bus",
    )
    assert s.ueffective == dt(2024, 12, 30)
    assert s.utermination == dt(2025, 11, 30)
    assert s.roll == 30


@pytest.mark.parametrize(
    ("f", "expected"),
    [
        (Frequency.CalDays(10), NoInput(0)),
        (Frequency.Months(1, None), 16),
        (Frequency.Months(1, RollDay.Day(16)), 16),
    ],
)
def test_roll_property(f, expected) -> None:
    s = Schedule(dt(2000, 1, 16), dt(2001, 1, 16), f)
    result = s.roll
    assert result == expected


def test_day_type_tenor() -> None:
    # should convert MF to Following only
    s = Schedule(
        dt(2024, 12, 30),
        "1d",
        "A",
        modifier="mf",
        calendar="stk",
    )
    assert s.utermination == dt(2025, 1, 2)


def test_cds_standard_example() -> None:
    # https://www.cdsmodel.com/documentation.html?# standard example
    # use Adjuster.FollowingExLast to avoid adjusting the final accrual date.
    s = Schedule(
        dt(2008, 12, 20),
        dt(2010, 3, 20),
        "Q",
        modifier="fex",
        calendar="bus",
        payment_lag=0,
    )
    expected = [
        dt(2008, 12, 22),
        dt(2009, 3, 20),
        dt(2009, 6, 22),
        dt(2009, 9, 21),
        dt(2009, 12, 21),
        dt(2010, 3, 20),
    ]
    assert s.aschedule == expected

    expected = [
        dt(2008, 12, 22),
        dt(2009, 3, 20),
        dt(2009, 6, 22),
        dt(2009, 9, 21),
        dt(2009, 12, 21),
        dt(2010, 3, 22),
    ]
    assert s.pschedule == expected


@pytest.mark.parametrize(
    "frequency",
    [
        "M",  # monthly,
        "Q",  # quarterly,
        "S",  # semi-annually,
        "A",  # annually,
        "10D",  # 10-cal-days
        "10B",  # 10-bus-days
        "2W",  # 14-cal-days
        "8M",  # 8-months
        "1Y",  # 1-year
    ],
)
def test_all_frequency_as_str(frequency):
    s = Schedule(
        dt(2000, 1, 1),
        dt(2010, 1, 1),
        frequency=frequency,
        stub="ShortFront",
        calendar="bus",
    )
    s.__str__()


def test_inference_busdays():
    # the effective is given adjusted whilst termination is unadjusted
    s = Schedule(
        effective=dt(2000, 1, 6),
        termination=dt(2000, 3, 1),
        frequency=Frequency.Months(1, None),
        modifier=Adjuster.BusDaysLagSettle(5),
    )
    assert s.uschedule == [dt(2000, 1, 1), dt(2000, 2, 1), dt(2000, 3, 1)]
    assert s.aschedule == [dt(2000, 1, 6), dt(2000, 2, 6), dt(2000, 3, 6)]


def test_payment_adjuster_2_and_3():
    s = Schedule(
        dt(2000, 1, 1),
        dt(2000, 3, 1),
        "M",
        calendar="all",
        modifier="none",
        payment_lag=1,
        payment_lag_exchange=2,
        extra_lag=-2,
    )
    assert s.pschedule == [dt(2000, 1, 2), dt(2000, 2, 2), dt(2000, 3, 2)]
    assert s.pschedule2 == [dt(2000, 1, 3), dt(2000, 2, 3), dt(2000, 3, 3)]
    assert s.pschedule3 == [dt(1999, 12, 30), dt(2000, 1, 30), dt(2000, 2, 28)]


@pytest.mark.parametrize(
    ("eff", "front", "back", "term"),
    [
        # All unadjusted
        (dt(2025, 1, 15), NoInput(0), NoInput(0), dt(2025, 4, 15)),
        (dt(2025, 1, 15), dt(2025, 2, 15), NoInput(0), dt(2025, 4, 15)),
        (dt(2025, 1, 15), NoInput(0), dt(2025, 3, 15), dt(2025, 4, 15)),
        (dt(2025, 1, 15), dt(2025, 2, 15), dt(2025, 3, 15), dt(2025, 4, 15)),
        # Stub given as adjusted
        (dt(2025, 1, 15), dt(2025, 2, 17), NoInput(0), dt(2025, 4, 15)),
        (dt(2025, 1, 15), NoInput(0), dt(2025, 3, 17), dt(2025, 4, 15)),
        (dt(2025, 1, 15), dt(2025, 2, 17), dt(2025, 3, 17), dt(2025, 4, 15)),
        # Stub given as mixed
        (dt(2025, 1, 15), dt(2025, 2, 17), dt(2025, 3, 15), dt(2025, 4, 15)),
    ],
)
def test_schedule_when_stub_input_is_regular(eff, front, back, term):
    # GH-dev 142
    s_base = Schedule(
        effective=dt(2025, 1, 15),
        termination=dt(2025, 3, 17),
        calendar="bus",
        frequency="M",
        modifier="mf",
    )
    assert s_base.uschedule == [dt(2025, 1, 15), dt(2025, 2, 15), dt(2025, 3, 15)]
    assert s_base.aschedule == [dt(2025, 1, 15), dt(2025, 2, 17), dt(2025, 3, 17)]

    s = Schedule(
        effective=eff,
        termination=term,
        front_stub=front,
        back_stub=back,
        calendar="bus",
        frequency="M",
        modifier="mf",
    )
    assert s._stubs == [False, False, False]


@pytest.mark.skip(reason="multiple stubs, where one may be a genuine stub is not implemented.")
@pytest.mark.parametrize("fs", [dt(2025, 2, 15), dt(2025, 2, 17)])
def test_schedule_when_one_front_stub_of_two_is_regular(fs):
    # GH-dev 142
    # this tests that one stub might be genuine whilst the other is a regular period and
    # the schedule still generates correctly.

    # this requires additional branching in the Rust scheduling code in the pre-check which has
    # not been developed. The most common use case for this pre-check is when only a front stub,
    # i.e. the first coupon date of a bond is provided.

    s = Schedule(
        effective=dt(2025, 1, 15),
        termination=dt(2025, 4, 25),
        front_stub=fs,
        back_stub=dt(2025, 4, 15),
        calendar="bus",
        frequency="M",
        modifier="mf",
    )
    assert s._stubs == [False, False, False, True]


def test_schedule_in_advance_payment():
    # used by FRA constructor
    from rateslib.scheduling import Adjuster

    s = Schedule(
        effective=dt(2024, 3, 20),
        termination=dt(2024, 12, 18),
        calendar="bus",
        frequency="Q",
        modifier="mf",
        payment_lag=Adjuster.BusDaysLagSettleInAdvance(1),
    )
    assert s.aschedule == [dt(2024, 3, 20), dt(2024, 6, 19), dt(2024, 9, 18), dt(2024, 12, 18)]
    assert s.pschedule == [dt(2024, 3, 21), dt(2024, 3, 21), dt(2024, 6, 20), dt(2024, 9, 19)]
    assert s.pschedule3 == s.pschedule


@pytest.mark.parametrize("tenor", ["3b", "3d", "7d", "14d", "2w", "1m", "6m", "12m", "18m", "2y"])
def test_single_period_from_str_matching_frequency(tenor):
    # test was introduced for a Bill that derives a termination from a string tenor.
    # When the frequency matches the tenor it should generate only a single period.
    s = Schedule(effective=dt(2025, 1, 15), termination=tenor, frequency=tenor)
    assert s.n_periods == 1
