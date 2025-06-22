import pytest
import numpy as np
from pandas import DatetimeIndex, date_range, DataFrame
from pandas.tseries.holiday import Holiday
from pandas.testing import assert_index_equal
from datetime import datetime as dt

import context
from rateslib.default import Defaults, NoInput
from rateslib.calendars import create_calendar
from rateslib.scheduling import (
    _check_unadjusted_regular_swap,
    _check_regular_swap,
    _is_divisible_months,
    _get_date_category,
    _get_default_stub,
    _get_unadjusted_roll,
    _get_unadjusted_date_alternatives,
    _get_unadjusted_short_stub_date,
    _get_unadjusted_stub_date,
    _get_n_periods_in_regular,
    _get_roll,
    _generate_regular_schedule_unadjusted,
    _generate_irregular_schedule_unadjusted,
    _infer_stub_date,
    Schedule,
)


@pytest.fixture
def cal_():
    rules = [Holiday("New Year", month=1, day=3)]
    return create_calendar(rules=rules)


@pytest.mark.parametrize(
    "dt1, dt2, fm, expected",
    [
        (dt(2022, 3, 16), dt(2022, 6, 30), 3, True),
        (dt(2022, 3, 16), dt(2024, 9, 10), 3, True),
        (dt(2022, 3, 16), dt(2028, 9, 16), 6, True),
        (dt(2022, 3, 16), dt(2029, 3, 16), 12, True),
        (dt(2022, 3, 16), dt(2022, 10, 16), 3, False),
        (dt(2022, 3, 31), dt(2024, 4, 1), 12, False),
    ],
)
def test_is_divisible_months(dt1, dt2, fm, expected):
    result = _is_divisible_months(dt1, dt2, fm)
    assert result == expected


@pytest.mark.parametrize(
    "date, expected",
    [
        (dt(2022, 2, 28), 0),
        (dt(2024, 2, 28), 1),
        (dt(2024, 2, 29), 2),
        (dt(2024, 4, 30), 3),
        (dt(2022, 1, 31), 4),
        (dt(2022, 1, 30), 5),
        (dt(2022, 9, 29), 6),
        (dt(2022, 9, 28), 7),
    ],
)
def test_get_date_category(date, expected):
    result = _get_date_category(date)
    assert result == expected


def test_get_date_category_raises():
    with pytest.raises(ValueError):
        _ = _get_date_category(dt(2022, 2, 26))


@pytest.mark.parametrize(
    "effective, termination, expected, expected2",
    [
        (dt(2022, 2, 22), dt(2024, 2, 22), 22, 22),
        (dt(2022, 2, 22), dt(2024, 2, 15), 0, 0),
        (dt(2022, 2, 28), dt(2024, 2, 29), 29, "eom"),
        (dt(2022, 6, 30), dt(2024, 9, 30), 30, "eom"),
        (dt(2022, 6, 30), dt(2024, 12, 30), 30, 30),
        (dt(2022, 2, 28), dt(2024, 9, 30), 30, "eom"),
        (dt(2024, 3, 31), dt(2024, 9, 30), 31, "eom"),
    ],
)
def test_get_unadjusted_roll(effective, termination, expected, expected2):
    result = _get_unadjusted_roll(effective, termination, eom=False)
    assert result == expected

    result = _get_unadjusted_roll(effective, termination, eom=True)
    assert result == expected2


def test_get_default_stub():
    assert "SHORTFRONT" == _get_default_stub("FRONT", "SHORTFRONTLONGBACK")
    assert "LONGBACK" == _get_default_stub("BACK", "SHORTFRONTLONGBACK")
    assert f"{Defaults.stub_length}FRONT" == _get_default_stub("FRONT", "FRONTBACK")


@pytest.mark.parametrize(
    "e, t, stub, exp_roll, exp_stub",
    [
        (dt(2022, 2, 26), dt(2024, 4, 22), "SHORTFRONT", 22, dt(2022, 4, 22)),
        (dt(2022, 2, 26), dt(2024, 4, 22), "LONGFRONT", 22, dt(2022, 7, 22)),
        (dt(2022, 2, 26), dt(2024, 4, 22), "SHORTBACK", 26, dt(2024, 2, 26)),
        (dt(2022, 2, 26), dt(2024, 4, 22), "LONGBACK", 26, dt(2023, 11, 26)),
    ],
)
def test_infer_stub_date(e, t, stub, exp_roll, exp_stub, cal_):
    result = _infer_stub_date(e, t, "Q", stub, NoInput(0), NoInput(0), "MF", False, NoInput(0), cal_)
    assert result[0]
    if "FRONT" in stub:
        assert result[1]["front_stub"] == exp_stub
        assert result[1]["roll"] == exp_roll
    else:
        assert result[1]["back_stub"] == exp_stub
        assert result[1]["roll"] == exp_roll


@pytest.mark.parametrize(
    "e, t, stub, exp_roll, exp_stub",
    [
        (dt(2022, 2, 26), dt(2024, 2, 26), "SHORTFRONT", 26, NoInput(0)),
        (dt(2022, 2, 26), dt(2024, 2, 26), "LONGFRONT", 26, NoInput(0)),
        (dt(2022, 2, 26), dt(2024, 2, 26), "SHORTBACK", 26, NoInput(0)),
        (dt(2022, 2, 26), dt(2024, 2, 26), "LONGBACK", 26, NoInput(0)),
    ],
)
def test_infer_stub_date_no_inference_on_regular(e, t, stub, exp_roll, exp_stub, cal_):
    result = _infer_stub_date(e, t, "Q", stub, NoInput(0), NoInput(0), "MF", False, NoInput(0), cal_)
    assert result[0]
    if "FRONT" in stub:
        assert result[1]["front_stub"] == exp_stub
        assert result[1]["roll"] == exp_roll
    else:
        assert result[1]["back_stub"] == exp_stub
        assert result[1]["roll"] == exp_roll


def test_infer_stub_date_no_inference_on_regular_dual(cal_):
    result = _infer_stub_date(
        dt(2022, 2, 26),
        dt(2024, 4, 26),
        "Q",
        "SHORTFRONTBACK",
        NoInput(0),
        dt(2024, 2, 26),
        "MF",
        NoInput(0),
        NoInput(0),
        cal_,
    )
    assert result[0]
    assert result[1]["front_stub"] is NoInput(0)
    assert result[1]["roll"] == 26

    result = _infer_stub_date(
        dt(2022, 2, 26),
        dt(2024, 4, 26),
        "Q",
        "FRONTSHORTBACK",
        dt(2022, 4, 26),
        NoInput(0),
        "MF",
        NoInput(0),
        NoInput(0),
        cal_,
    )
    assert result[0]
    assert result[1]["back_stub"] is NoInput(0)
    assert result[1]["roll"] == 26


@pytest.mark.parametrize(
    "e, t, stub",
    [
        (dt(2022, 2, 26), dt(2024, 4, 22), "SHORTFRONT"),
        (dt(2022, 2, 26), dt(2024, 4, 22), "LONGFRONT"),
        (dt(2022, 2, 26), dt(2024, 4, 22), "SHORTBACK"),
        (dt(2022, 2, 26), dt(2024, 4, 22), "LONGBACK"),
    ],
)
def test_infer_stub_date_invalid_roll(e, t, stub, cal_):
    result = _infer_stub_date(e, t, "Q", stub, NoInput(0), NoInput(0), "MF", NoInput(0), 14, cal_)
    assert result[0] is False


@pytest.mark.parametrize(
    "e, fs, t, stub, exp_roll, exp_stub",
    [
        (dt(2022, 1, 1), dt(2022, 2, 26), dt(2024, 4, 26), "FRONTSHORTBACK", 26, dt(2024, 2, 26)),
        (dt(2022, 1, 1), dt(2022, 2, 26), dt(2024, 4, 26), "FRONTLONGBACK", 26, dt(2023, 11, 26)),
    ],
)
def test_infer_stub_date_dual_sided(e, fs, t, stub, exp_roll, exp_stub, cal_):
    result = _infer_stub_date(e, t, "Q", stub, fs, NoInput(0), "MF", NoInput(0), NoInput(0), cal_)
    assert result[0]
    assert result[1]["ueffective"] == e
    assert result[1]["front_stub"] == fs
    assert result[1]["back_stub"] == exp_stub
    assert result[1]["utermination"] == t
    assert result[1]["roll"] == exp_roll


@pytest.mark.parametrize(
    "e, bs, t, stub, exp_roll, exp_stub",
    [
        (dt(2022, 1, 1), dt(2024, 2, 26), dt(2024, 4, 26), "SHORTFRONTBACK", 26, dt(2022, 2, 26)),
        (dt(2022, 1, 1), dt(2024, 2, 26), dt(2024, 4, 26), "LONGFRONTBACK", 26, dt(2022, 5, 26)),
    ],
)
def test_infer_stub_date_dual_sided2(e, bs, t, stub, exp_roll, exp_stub, cal_):
    result = _infer_stub_date(e, t, "Q", stub, NoInput(0), bs, "MF", False, NoInput(0), cal_)
    assert result[0]
    assert result[1]["ueffective"] == e
    assert result[1]["front_stub"] == exp_stub
    assert result[1]["back_stub"] == bs
    assert result[1]["utermination"] == t
    assert result[1]["roll"] == exp_roll


def test_infer_stub_date_dual_sided_invalid(cal_):
    result = _infer_stub_date(
        dt(2022, 1, 1),
        dt(2022, 12, 31),
        "Q",
        "FRONTSHORTBACK",
        dt(2022, 2, 13),
        None,
        "MF",
        False,
        9,
        cal_,
    )
    assert not result[0]


def test_infer_stub_date_eom(cal_):
    result = _infer_stub_date(
        dt(2022, 1, 1),
        dt(2023, 2, 28),
        "Q",
        "LONGFRONT",
        NoInput(0),
        NoInput(0),
        "MF",
        True,  # <- the EOM parameter forces the stub to be 31 May and not 28 May
        NoInput(0),
        cal_,
    )
    assert result[1]["front_stub"] == dt(2022, 5, 31)


def test_schedule_repr(cal_):
    schedule = Schedule(dt(2022, 1, 1), "2M", "M", NoInput(0), NoInput(0), NoInput(0), NoInput(0), False, "MF", cal_, 1)
    expected = "freq: M,  stub: SHORTFRONT,  roll: 1,  pay lag: 1,  modifier: MF\n"
    df = DataFrame(
        {
            Defaults.headers["stub_type"]: ["Regular", "Regular"],
            Defaults.headers["u_acc_start"]: [dt(2022, 1, 1), dt(2022, 2, 1)],
            Defaults.headers["u_acc_end"]: [dt(2022, 2, 1), dt(2022, 3, 1)],
            Defaults.headers["a_acc_start"]: [dt(2022, 1, 4), dt(2022, 2, 1)],
            Defaults.headers["a_acc_end"]: [dt(2022, 2, 1), dt(2022, 3, 1)],
            Defaults.headers["payment"]: [dt(2022, 2, 2), dt(2022, 3, 2)],
        }
    )
    assert schedule.__repr__() == expected + df.__repr__()


def test_schedule_raises(cal_):
    with pytest.raises(ValueError, match="`frequency` must be in"):
        _ = Schedule(dt(2022, 1, 1), dt(2022, 12, 31), "Bad")

    with pytest.raises(ValueError, match="`termination` must be after"):
        _ = Schedule(dt(2022, 1, 1), dt(2021, 12, 31), "Q")

    with pytest.raises(ValueError):
        _ = Schedule(
            dt(2022, 1, 1),
            dt(2022, 12, 31),
            "Q",
            "SHORTFRONT",
            None,
            dt(2022, 11, 15),
            None,
            False,
            "MF",
            cal_,
            1,
        )

    with pytest.raises(ValueError):
        _ = Schedule(
            dt(2022, 1, 1),
            dt(2022, 12, 31),
            "Q",
            "SHORTBACK",
            dt(2022, 3, 15),
            None,
            None,
            False,
            "MF",
            cal_,
            1,
        )

    with pytest.raises(ValueError):
        _ = Schedule(
            dt(2022, 1, 1),
            dt(2022, 12, 31),
            "Q",
            "SBLB",
            dt(2022, 3, 15),
            None,
            None,
            False,
            "MF",
            cal_,
            1,
        )


@pytest.mark.parametrize(
    "eff, term, f, roll, exp",
    [
        (dt(2022, 3, 16), dt(2022, 6, 30), "S", NoInput(0), False),  # frequency
        (dt(2022, 3, 15), dt(2022, 9, 21), "Q", "imm", False),  # non-imm eff
        (dt(2022, 3, 16), dt(2024, 9, 10), "Q", "imm", False),  # non-imm term
        (dt(2022, 3, 30), dt(2029, 3, 31), "A", "eom", False),  # non-eom eff
        (dt(2022, 3, 31), dt(2023, 3, 30), "A", "eom", False),  # non-eom term
        (dt(2022, 3, 2), dt(2029, 3, 1), "A", "som", False),  # non-som eff
        (dt(2022, 3, 1), dt(2023, 3, 2), "A", "som", False),  # non-som term
        (dt(2022, 3, 30), dt(2023, 9, 30), "S", 31, False),  # non-eom
        (dt(2024, 2, 28), dt(2025, 8, 30), "S", 30, False),  # is leap
        (dt(2022, 2, 28), dt(2024, 2, 28), "S", 30, False),  # is leap
        (dt(2024, 2, 29), dt(2025, 8, 30), "S", 30, True),  # is leap
        (dt(2022, 2, 28), dt(2025, 8, 29), "S", 29, True),  # is end feb
        (dt(2022, 2, 20), dt(2025, 8, 20), "S", 20, True),  # OK
        (dt(2022, 2, 20), dt(2025, 8, 21), "S", 20, False),  # roll
        (dt(2022, 2, 21), dt(2025, 8, 20), "S", 20, False),  # roll
        (dt(2022, 2, 22), dt(2024, 2, 15), "S", NoInput(0), False),  # no valid roll
        (dt(2022, 2, 28), dt(2024, 2, 29), "S", NoInput(0), True),  # 29 or eom
        (dt(2022, 6, 30), dt(2024, 12, 30), "S", NoInput(0), True),  # 30
    ],
)
def test_unadjusted_regular_swap(eff, term, f, roll, exp):
    result = _check_unadjusted_regular_swap(eff, term, f, False, roll)[0]
    assert result == exp


@pytest.mark.parametrize(
    "eff, term, f, m, roll, exp",
    [
        (dt(2022, 3, 16), dt(2022, 6, 30), "S", "NONE", NoInput(0), False),  # frequency
        (dt(2022, 3, 15), dt(2022, 9, 21), "Q", "NONE", "imm", False),  # non-imm eff
        (dt(2022, 3, 16), dt(2024, 9, 10), "Q", "NONE", "imm", False),  # non-imm term
        (dt(2022, 3, 30), dt(2029, 3, 31), "A", "NONE", "eom", False),  # non-eom eff
        (dt(2022, 3, 31), dt(2023, 3, 30), "A", "NONE", "eom", False),  # non-eom term
        (dt(2022, 3, 2), dt(2029, 3, 1), "A", "NONE", "som", False),  # non-som eff
        (dt(2022, 3, 1), dt(2023, 3, 2), "A", "NONE", "som", False),  # non-som term
        (dt(2022, 3, 30), dt(2023, 9, 30), "S", "NONE", 31, False),  # non-eom
        (dt(2024, 2, 28), dt(2025, 8, 30), "S", "NONE", 30, False),  # is leap
        (dt(2022, 2, 28), dt(2024, 2, 28), "S", "NONE", 30, False),  # is leap
        (dt(2024, 2, 29), dt(2025, 8, 30), "S", "NONE", 30, True),  # is leap
        (dt(2022, 2, 28), dt(2025, 8, 29), "S", "NONE", 29, True),  # is end feb
        (dt(2022, 2, 20), dt(2025, 8, 20), "S", "NONE", 20, True),  # OK
        (dt(2022, 2, 20), dt(2025, 8, 21), "S", "NONE", 20, False),  # roll
        (dt(2022, 2, 21), dt(2025, 8, 20), "S", "NONE", 20, False),  # roll
        (dt(2022, 2, 22), dt(2024, 2, 15), "S", "NONE", NoInput(0), False),  # no valid roll
        (dt(2022, 2, 28), dt(2024, 2, 29), "S", "NONE", NoInput(0), True),  # 29 or eom
        (dt(2022, 6, 30), dt(2024, 12, 30), "S", "NONE", NoInput(0), True),  # 30
    ],
)
def test_check_regular_swap(eff, term, f, m, roll, exp, cal_):
    # modifier is unadjusted: should mirror test_unadjusted_regular_swap
    result = _check_regular_swap(eff, term, f, m, False, roll, cal_)
    assert result[0] == exp


# 12th and 13th of Feb and March are Saturday and Sunday
@pytest.mark.parametrize(
    "eff, term, roll, e_bool, e_ueff, e_uterm, e_roll",
    [
        (dt(2022, 2, 11), dt(2022, 3, 11), 11, True, dt(2022, 2, 11), dt(2022, 3, 11), 11),
        (dt(2022, 2, 14), dt(2022, 3, 14), 14, True, dt(2022, 2, 14), dt(2022, 3, 14), 14),
        (dt(2022, 2, 14), dt(2022, 3, 14), NoInput(0), True, dt(2022, 2, 14), dt(2022, 3, 14), 14),
        (dt(2022, 2, 13), dt(2022, 3, 14), NoInput(0), True, dt(2022, 2, 13), dt(2022, 3, 13), 13),
        (dt(2022, 2, 13), dt(2022, 3, 12), NoInput(0), False, None, None, None),
        (dt(2022, 2, 12), dt(2022, 3, 14), NoInput(0), True, dt(2022, 2, 12), dt(2022, 3, 12), 12),
        (dt(2022, 2, 12), dt(2022, 3, 13), NoInput(0), False, None, None, None),
        (dt(2022, 2, 14), dt(2022, 3, 12), NoInput(0), True, dt(2022, 2, 12), dt(2022, 3, 12), 12),
        (dt(2022, 2, 14), dt(2022, 3, 14), 12, True, dt(2022, 2, 12), dt(2022, 3, 12), 12),
        (dt(2022, 2, 14), dt(2022, 3, 14), 11, False, None, None, None),
        (dt(2022, 2, 28), dt(2022, 3, 31), NoInput(0), True, dt(2022, 2, 28), dt(2022, 3, 31), 31),
        (dt(2022, 2, 28), dt(2022, 3, 31), 28, False, None, None, None),
        (dt(2022, 2, 28), dt(2022, 3, 31), "eom", True, dt(2022, 2, 28), dt(2022, 3, 31), "eom"),
    ],
)
def test_check_regular_swap_mf(eff, term, roll, e_bool, e_ueff, e_uterm, e_roll, cal_):
    result = _check_regular_swap(eff, term, "M", "MF", False, roll, cal_)
    assert result[0] == e_bool
    if e_bool:
        assert result[1]["ueffective"] == e_ueff
        assert result[1]["utermination"] == e_uterm
        assert result[1]["roll"] == e_roll


@pytest.mark.parametrize(
    "date, modifier, expected",
    [
        (dt(2022, 1, 3), "F", [dt(2022, 1, 3)]),
        (dt(2022, 1, 4), "F", [dt(2022, 1, 4), dt(2022, 1, 3), dt(2022, 1, 2), dt(2022, 1, 1)]),
        (dt(2022, 8, 1), "F", [dt(2022, 8, 1), dt(2022, 7, 31), dt(2022, 7, 30)]),
        (dt(2022, 7, 29), "F", [dt(2022, 7, 29)]),
        (dt(2022, 7, 29), "MF", [dt(2022, 7, 29), dt(2022, 7, 30), dt(2022, 7, 31)]),
        (dt(2022, 7, 29), "P", [dt(2022, 7, 29), dt(2022, 7, 30), dt(2022, 7, 31)]),
        (dt(2021, 12, 31), "P", [dt(2021, 12, 31), dt(2022, 1, 1), dt(2022, 1, 2), dt(2022, 1, 3)]),
        (dt(2022, 1, 4), "MP", [dt(2022, 1, 4), dt(2022, 1, 3), dt(2022, 1, 2), dt(2022, 1, 1)]),
    ],
)
def test_unadjusted_date_alternatives(date, modifier, cal_, expected):
    result = _get_unadjusted_date_alternatives(date, modifier, cal_)
    assert result == expected


@pytest.mark.parametrize(
    "effective, termination, uf, ub, roll, expected",
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
def test_generate_irregular_uschedule(effective, termination, uf, ub, roll, expected):
    result = list(
        _generate_irregular_schedule_unadjusted(effective, termination, "Q", roll, uf, ub)
    )
    assert result == expected


@pytest.mark.parametrize(
    "effective, termination, roll, expected",
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
def test_generate_regular_uschedule(effective, termination, roll, expected):
    result = list(_generate_regular_schedule_unadjusted(effective, termination, "Q", roll))
    assert result == expected


@pytest.mark.parametrize(
    "month, year, roll, expected",
    [
        (2, 2022, "eom", dt(2022, 2, 28)),
        (2, 2024, "eom", dt(2024, 2, 29)),
        (2, 2024, 31, dt(2024, 2, 29)),
        (4, 2024, 31, dt(2024, 4, 30)),
        (9, 2022, "imm", dt(2022, 9, 21)),
        (9, 2022, "som", dt(2022, 9, 1)),
        (9, 2022, 11, dt(2022, 9, 11)),
    ],
)
def test_get_roll(month, year, roll, expected):
    result = _get_roll(month, year, roll)
    assert result == expected


@pytest.mark.parametrize(
    "effective, termination, frequency, expected",
    [
        (dt(2022, 2, 15), dt(2022, 8, 28), "M", 6),
        (dt(2022, 2, 15), dt(2022, 8, 28), "Q", 2),
        (dt(2022, 2, 15), dt(2032, 2, 28), "Q", 40),
        (dt(2022, 2, 15), dt(2032, 2, 28), "Z", 1),
    ],
)
def test_regular_n_periods(effective, termination, frequency, expected):
    result = _get_n_periods_in_regular(effective, termination, frequency)
    assert result == expected


def test_regular_n_periods_raises():
    # this raise is superfluous by the design principles of private methods
    with pytest.raises(ValueError):
        _get_n_periods_in_regular(dt(2020, 1, 1), dt(2020, 3, 31), "Q")


@pytest.mark.parametrize(
    "eff, term, freq, ss, eom, roll, expected",
    [
        (dt(2022, 1, 1), dt(2023, 2, 15), "M", "FRONT", False, NoInput(0), dt(2022, 1, 15)),
        (dt(2022, 1, 1), dt(2023, 2, 15), "Q", "FRONT", False, NoInput(0), dt(2022, 2, 15)),
        (dt(2022, 1, 1), dt(2023, 2, 15), "S", "FRONT", False, NoInput(0), dt(2022, 2, 15)),
        (dt(2022, 2, 15), dt(2023, 2, 1), "S", "FRONT", False, NoInput(0), dt(2022, 8, 1)),
        (dt(2022, 1, 1), dt(2023, 2, 15), "M", "BACK", False, NoInput(0), dt(2023, 2, 1)),
        (dt(2022, 1, 1), dt(2023, 2, 15), "Q", "BACK", False, NoInput(0), dt(2023, 1, 1)),
        (dt(2022, 1, 1), dt(2023, 2, 15), "S", "BACK", False, NoInput(0), dt(2023, 1, 1)),
        (dt(2022, 2, 15), dt(2023, 2, 1), "S", "BACK", False, NoInput(0), dt(2022, 8, 15)),
        (dt(2022, 1, 1), dt(2023, 2, 28), "M", "FRONT", True, NoInput(0), dt(2022, 1, 31)),
        (dt(2022, 3, 1), dt(2023, 2, 28), "Q", "FRONT", True, NoInput(0), dt(2022, 5, 31)),
        (dt(2022, 3, 1), dt(2023, 2, 18), "Q", "FRONT", False, 17, dt(2022, 5, 17)),
    ],
)
def test_get_unadjusted_short_stub_date(eff, term, freq, ss, eom, roll, expected):
    result = _get_unadjusted_short_stub_date(eff, term, freq, ss, eom, roll)
    assert result == expected


@pytest.mark.parametrize(
    "eff, term, freq, stub, eom, roll, expected",
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
        (dt(2022, 3, 1), dt(2023, 2, 18), "Q", "SHORTFRONT", False, 17, dt(2022, 5, 17)),
    ],
)
def test_get_unadjusted_stub_date_mirror(eff, term, freq, stub, eom, roll, expected):
    # this should mirror the short stub date test
    result = _get_unadjusted_stub_date(eff, term, freq, stub, eom, roll)
    assert result == expected


@pytest.mark.parametrize(
    "eff, term, freq, stub, eom, roll, expected",
    [
        (dt(2022, 1, 1), dt(2023, 2, 15), "M", "LONGFRONT", False, NoInput(0), dt(2022, 2, 15)),
        (dt(2022, 1, 1), dt(2023, 2, 15), "Q", "LONGFRONT", False, NoInput(0), dt(2022, 5, 15)),
        (dt(2022, 1, 1), dt(2023, 2, 15), "S", "LONGFRONT", False, NoInput(0), dt(2022, 8, 15)),
        (dt(2022, 2, 15), dt(2023, 2, 1), "S", "LONGFRONT", False, NoInput(0), dt(2023, 2, 1)),
        (dt(2022, 1, 1), dt(2023, 2, 15), "M", "LONGBACK", False, NoInput(0), dt(2023, 1, 1)),
        (dt(2022, 1, 1), dt(2023, 2, 15), "Q", "LONGBACK", False, NoInput(0), dt(2022, 10, 1)),
        (dt(2022, 1, 1), dt(2023, 2, 15), "S", "LONGBACK", False, NoInput(0), dt(2022, 7, 1)),
        (dt(2022, 2, 15), dt(2023, 2, 1), "S", "LONGBACK", False, NoInput(0), dt(2022, 2, 15)),
        (dt(2022, 1, 1), dt(2023, 2, 28), "M", "LONGFRONT", True, NoInput(0), dt(2022, 2, 28)),
        (dt(2022, 3, 1), dt(2023, 2, 28), "Q", "LONGFRONT", True, NoInput(0), dt(2022, 8, 31)),
        (dt(2022, 3, 1), dt(2023, 2, 18), "Q", "LONGFRONT", False, 17, dt(2022, 8, 17)),
        (dt(2022, 4, 30), dt(2023, 2, 18), "Q", "LONGBACK", True, NoInput(0), dt(2022, 10, 31)),
    ],
)
def test_get_unadjusted_stub_date_long(eff, term, freq, stub, eom, roll, expected):
    result = _get_unadjusted_stub_date(eff, term, freq, stub, eom, roll)
    assert result == expected


@pytest.mark.parametrize(
    "e, t, r, exp_roll, exp_ue, exp_ut",
    [
        (dt(2020, 8, 31), dt(2021, 2, 26), NoInput(0), 31, dt(2020, 8, 31), dt(2021, 2, 28)),
        (dt(2021, 2, 26), dt(2021, 8, 31), NoInput(0), 31, dt(2021, 2, 28), dt(2021, 8, 31)),
        (dt(2021, 2, 26), dt(2021, 8, 30), 29, 29, dt(2021, 2, 28), dt(2021, 8, 29)),
    ],
)
def test_schedule_eom(e, t, r, exp_roll, exp_ue, exp_ut, cal_):
    sched = Schedule(e, t, "S", roll=r, modifier="MF", calendar=cal_)
    assert sched.ueffective == exp_ue
    assert sched.utermination == exp_ut
    assert sched.roll == exp_roll


@pytest.mark.parametrize(
    "e, t, r, exp_roll, exp_ue, exp_ut",
    [
        (dt(2020, 8, 31), dt(2021, 2, 26), NoInput(0), 31, dt(2020, 8, 31), dt(2021, 2, 28)),
        (dt(2021, 2, 26), dt(2021, 8, 31), NoInput(0), 31, dt(2021, 2, 28), dt(2021, 8, 31)),
        (dt(2021, 2, 26), dt(2021, 8, 30), 29, 29, dt(2021, 2, 28), dt(2021, 8, 29)),
    ],
)
def test_schedule_stub_inference(e, t, r, exp_roll, exp_ue, exp_ut, cal_):
    sched = Schedule(e, t, "S", roll=r, modifier="MF", calendar=cal_)
    assert sched.ueffective == exp_ue
    assert sched.utermination == exp_ut
    assert sched.roll == exp_roll


def test_payment_lag_is_business_days():
    sched = Schedule(dt(2022, 11, 16), "1M", "M", modifier="MF", calendar="ldn")
    assert sched.pschedule[1] == dt(2022, 12, 20)
    # not 19th Dec which is adjusted(16th Dec + 2 days)


def test_schedule_bad_stub_combinations_raise():
    with pytest.raises(ValueError, match="Must supply at least one stub date"):
        _ = Schedule(
            effective=dt(2022, 1, 1), termination=dt(2023, 1, 1), frequency="S", stub="FRONTBACK"
        )


def test_schedule_bad_stub_combinations_raise2():
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
    "st, fs, bs",
    [
        ("FRONTBACK", NoInput(0), dt(2023, 1, 1)),
        ("FRONTBACK", dt(2022, 2, 1), NoInput(0)),
        ("FRONTBACK", dt(2022, 4, 15), dt(2023, 10, 15)),
        ("FRONT", NoInput(0), NoInput(0)),
        ("FRONT", dt(2022, 2, 1), NoInput(0)),
        ("BACK", NoInput(0), dt(2023, 1, 1)),
        ("BACK", NoInput(0), NoInput(0)),
    ],
)
def test_schedule_combinations_valid(st, fs, bs):
    Schedule(
        effective=dt(2022, 1, 1),
        termination=dt(2023, 2, 1),
        frequency="S",
        stub=st,
        back_stub=bs,
        front_stub=fs,
    )


@pytest.mark.parametrize(
    "st, fs, bs, roll",
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
def test_schedule_combinations_invalid(st, fs, bs, roll):
    with pytest.raises(ValueError, match="date, stub and roll inputs are invalid"):
        Schedule(
            effective=dt(2022, 1, 1),
            termination=dt(2023, 2, 1),
            frequency="S",
            stub=st,
            back_stub=bs,
            front_stub=fs,
            roll=roll,
        )


def test_schedule_n_periods():
    result = Schedule(
        effective=dt(2022, 1, 1),
        termination=dt(2023, 2, 1),
        frequency="S",
        stub="FRONT",
    )
    assert result.n_periods == 3


@pytest.mark.parametrize(
    "ue, ut, exp",
    [
        (dt(2023, 3, 17), dt(2023, 12, 20), dt(2023, 9, 20)),
        (dt(2022, 12, 19), dt(2023, 12, 20), dt(2023, 3, 15)),
    ],  # PR #9
)
def test_get_unadjusted_long_stub_imm(ue, ut, exp):
    result = _get_unadjusted_stub_date(ue, ut, "Q", "LONGFRONT", False, "imm")
    assert result == exp


@pytest.mark.parametrize(
    "ue, ut, exp",
    [
        (dt(2023, 3, 17), dt(2023, 12, 20), dt(2023, 6, 21)),  # PR #9
    ],
)
def test_get_unadjusted_short_stub_imm(ue, ut, exp):
    result = _get_unadjusted_short_stub_date(ue, ut, "Q", "FRONT", False, "imm")
    assert result == exp


def test_dead_stubs():
    # this was a bug detected in performance testing which generated a 1d invalid stub.
    # this failed originally because a 1D stub between Sun 2nd May 27 and Mon 3rd May 27
    # was invalid since the adjusted accrual schedule modified the sunday to be
    # equal to the Monday giving a 0 day period.
    s = Schedule(
        dt(2027, 5, 2),
        dt(2046, 5, 3),
        "A",
        "SHORTFRONT",
        NoInput(0),
        NoInput(0),
        NoInput(0),
        NoInput(0),
        NoInput(0),
        "bus",
        NoInput(0),
    )
    assert s.uschedule[0:2] == [dt(2027, 5, 3), dt(2028, 5, 3)]
    assert s.aschedule[0:2] == [dt(2027, 5, 3), dt(2028, 5, 3)]

    # manipulate this test to cover the case for dual sided stubs
    s = Schedule(
        dt(2027, 5, 2),
        dt(2046, 6, 3),
        "A",
        "SHORTFRONTSHORTBACK",
        NoInput(0),
        dt(2046, 5, 3),  # back stub means front stub is inferred
        NoInput(0),
        NoInput(0),
        NoInput(0),
        "bus",
        NoInput(0),
    )
    assert s.uschedule[0:2] == [dt(2027, 5, 3), dt(2028, 5, 3)]
    assert s.aschedule[0:2] == [dt(2027, 5, 3), dt(2028, 5, 3)]

    # this was a bug detected in performance testing which generated a 1d invalid stub.
    # this failed originally because the ueffective date of Sat 20-dec-25 and the
    # inferred front stub of Sun 21-dec-25 both adjusted forwards to 22-dec-25
    # giving a 0 day period.
    s = Schedule(
        dt(2025, 12, 20),
        dt(2069, 12, 21),
        "A",
        "SHORTFRONT",
        NoInput(0),
        NoInput(0),
        NoInput(0),
        NoInput(0),
        NoInput(0),
        "bus",
        NoInput(0),
    )
    assert s.uschedule[0:2] == [dt(2025, 12, 21), dt(2026, 12, 21)]
    assert s.aschedule[0:2] == [dt(2025, 12, 22), dt(2026, 12, 21)]

    # this was a bug detected in performance testing which generated a 1d invalid stub.
    # this failed originally because the utermination date of Sat 20-dec-25 and the
    # inferred front stub of Sun 21-dec-25 both adjusted forwards to 22-dec-25
    # giving a 0 day period.
    s = Schedule(
        dt(2027, 10, 19),
        dt(2047, 10, 20),
        "A",
        "SHORTBACK",
        NoInput(0),
        NoInput(0),
        NoInput(0),
        NoInput(0),
        NoInput(0),
        "bus",
        NoInput(0),
    )
    assert s.uschedule[-2:] == [dt(2046, 10, 19), dt(2047, 10, 19)]
    assert s.aschedule[-2:] == [dt(2046, 10, 19), dt(2047, 10, 21)]

    # manipulate this test for dual sided stubs
    s = Schedule(
        dt(2027, 8, 19),
        dt(2047, 10, 20),
        "A",
        "SHORTFRONTSHORTBACK",
        dt(2027, 10, 19),
        NoInput(0),
        NoInput(0),
        NoInput(0),
        NoInput(0),
        "bus",
        NoInput(0),
    )
    assert s.uschedule[-2:] == [dt(2046, 10, 19), dt(2047, 10, 19)]
    assert s.aschedule[-2:] == [dt(2046, 10, 19), dt(2047, 10, 21)]


@pytest.mark.parametrize("mode, end, roll", [
    (NoInput(0), dt(2025, 8, 17), 17),
    ("swaps_align", dt(2025, 8, 17), 17),
    ("swaptions_align", dt(2025, 8, 19), 19),
])
def test_eval_mode(mode, end, roll):
    sch = Schedule(
        effective="1Y",
        termination="1Y",
        frequency="S",
        calendar="tgt",
        eval_date=dt(2023, 8, 17),
        eval_mode=mode,
    )
    assert sch.roll == roll
    assert sch.termination == end


def test_eval_date_raises():
    with pytest.raises(ValueError, match="For `effective` given as string tenor, must"):
        Schedule(
            effective="1Y",
            termination="1Y",
            frequency="S",
        )
