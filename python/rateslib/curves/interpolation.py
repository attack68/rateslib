from __future__ import annotations

from math import floor
from typing import TYPE_CHECKING, Protocol

from pytz import UTC

from rateslib.calendars import dcf
from rateslib.dual import dual_exp, dual_log
from rateslib.rs import index_left_f64

if TYPE_CHECKING:
    from rateslib.typing import Any, Curve, DualTypes, datetime  # pragma: no cover


class InterpolationFunction(Protocol):
    # Callable type for Interpolation Functions
    def __call__(self, date: datetime, curve: Curve) -> DualTypes: ...


def _linear(date: datetime, curve: Curve) -> DualTypes:
    x, x_1, x_2, i = _get_posix(date, curve)
    node_values = list(curve.nodes.nodes.values())
    y_1, y_2 = node_values[i], node_values[i + 1]
    return y_1 + (y_2 - y_1) * (x - x_1) / (x_2 - x_1)


def _linear_bus(date: datetime, curve: Curve) -> DualTypes:
    i = index_left(curve.nodes.keys, curve.nodes.n, date)
    x_1, x_2 = curve.nodes.keys[i], curve.nodes.keys[i + 1]
    d_n = dcf(x_1, x_2, "bus252", calendar=curve.meta.calendar)
    d_m = dcf(x_1, date, "bus252", calendar=curve.meta.calendar)
    node_values = list(curve.nodes.nodes.values())
    y_1, y_2 = node_values[i], node_values[i + 1]
    return y_1 + (y_2 - y_1) * d_m / d_n


def _log_linear(date: datetime, curve: Curve) -> DualTypes:
    x, x_1, x_2, i = _get_posix(date, curve)
    node_values = list(curve.nodes.nodes.values())
    y_1, y_2 = dual_log(node_values[i]), dual_log(node_values[i + 1])
    return dual_exp(y_1 + (y_2 - y_1) * (x - x_1) / (x_2 - x_1))


def _log_linear_bus(date: datetime, curve: Curve) -> DualTypes:
    i = index_left(curve.nodes.keys, curve.nodes.n, date)
    x_1, x_2 = curve.nodes.keys[i], curve.nodes.keys[i + 1]
    d_n = dcf(x_1, x_2, "bus252", calendar=curve.meta.calendar)
    d_m = dcf(x_1, date, "bus252", calendar=curve.meta.calendar)
    node_values = list(curve.nodes.nodes.values())
    y_1, y_2 = dual_log(node_values[i]), dual_log(node_values[i + 1])
    return dual_exp(y_1 + (y_2 - y_1) * d_m / d_n)


def _flat_forward(date: datetime, curve: Curve) -> DualTypes:
    x, x_1, x_2, i = _get_posix(date, curve)
    node_values = list(curve.nodes.nodes.values())
    y_1, y_2 = node_values[i], node_values[i + 1]
    if x >= x_2:
        return y_2
    return y_1


def _flat_backward(date: datetime, curve: Curve) -> DualTypes:
    x, x_1, x_2, i = _get_posix(date, curve)
    node_values = list(curve.nodes.nodes.values())
    y_1, y_2 = node_values[i], node_values[i + 1]
    if x <= x_1:
        return y_1
    return y_2


def _linear_zero_rate(date: datetime, curve: Curve) -> DualTypes:
    # base time on DCF, which depends on the curve convention.
    i = index_left(curve.nodes.keys, curve.nodes.n, date)
    nvs = list(curve.nodes.nodes.values())
    nds = curve.nodes.keys

    d_2 = dcf(nds[0], nds[i + 1], curve.meta.convention, calendar=curve.meta.calendar)
    r_2 = -dual_log(nvs[i + 1]) / dcf(
        nds[0], nds[i + 1], curve.meta.convention, calendar=curve.meta.calendar
    )
    if i == 0:
        # first period must use flat backwards zero rate
        d_m = dcf(nds[0], date, curve.meta.convention, calendar=curve.meta.calendar)
        r_m = r_2
    else:
        d_1 = dcf(nds[0], nds[i], curve.meta.convention, calendar=curve.meta.calendar)
        r_1 = -dual_log(nvs[i]) / d_1
        d_m = dcf(nds[0], date, curve.meta.convention, calendar=curve.meta.calendar)
        r_m = r_1 + (r_2 - r_1) * (d_m - d_1) / (d_2 - d_1)

    return dual_exp(-r_m * d_m)


def _linear_index(date: datetime, curve: Curve) -> DualTypes:
    x, x_1, x_2, i = _get_posix(date, curve)
    node_values = list(curve.nodes.nodes.values())
    y_1, y_2 = node_values[i], node_values[i + 1]
    return (1 / y_1 + (1 / y_2 - 1 / y_1) * (x - x_1) / (x_2 - x_1)) ** -1.0


def _runtime_error(date: datetime, curve: Curve) -> DualTypes:
    """Spline interpolation is performed by a PPSpline over the whole nodes domain."""
    raise RuntimeError(  # pragma: no cover
        "An `interpolation` mode of 'spline' should never call this function.\n"
        "The configured knot sequence `t` for the PPSpline should cover the entire `nodes` domain."
    )


INTERPOLATION: dict[str, InterpolationFunction] = {
    "linear": _linear,
    "linear_bus252": _linear_bus,
    "log_linear": _log_linear,
    "log_linear_bus252": _log_linear_bus,
    "linear_zero_rate": _linear_zero_rate,
    "linear_index": _linear_index,
    "flat_forward": _flat_forward,
    "flat_backward": _flat_backward,
    "spline": _runtime_error,
}


def _get_posix(date: datetime, curve: Curve) -> tuple[float, float, float, int]:
    """
    Convert a datetime and curve_nodes to posix timestamps and return the index_left.
    """
    date_posix: float = date.replace(tzinfo=UTC).timestamp()
    l_index = index_left_f64(curve.nodes.posix_keys, date_posix, None)
    node_left_posix, node_right_posix = (
        curve.nodes.posix_keys[l_index],
        curve.nodes.posix_keys[l_index + 1],
    )
    return date_posix, node_left_posix, node_right_posix, l_index


def index_left(
    list_input: list[Any],
    list_length: int,
    value: Any,
    left_count: int = 0,
) -> int:
    """
    Return the interval index of a value from an ordered input list on the left side.

    Parameters
    ----------
    input : list
        Ordered list (lowest to highest) containing datatypes the same as value.
    length : int
        The length of ``input``.
    value : Any
        The value for which to determine the list index of.
    left_count : int
        The counter to pass recursively to determine the output. Users should not
        directly specify, it is used in internal calculation only.

    Returns
    -------
    int : The left index of the interval within which value is found (or extrapolated
          from)

    Notes
    -----
    Uses a binary search method which operates with time :math:`O(log_2 n)`.

    Examples
    --------
    .. ipython:: python

       from rateslib.curves import index_left

    Out of domain values return the left-side index of the closest matching interval.
    100 is attributed to the interval (1, 2].

    .. ipython:: python

       list = [0, 1, 2]
       index_left(list, 3, 100)

    -100 is attributed to the interval (0, 1].

    .. ipython:: python

       index_left(list, 3, -100)

    Interior values return the left-side index of the interval.
    1.45 is attributed to the interval (1, 2].

    .. ipython:: python

       index_left(list, 3, 1.45)

    1 is attributed to the interval (0, 1].

    .. ipython:: python

       index_left(list, 3, 1)

    """
    if list_length == 1:
        raise ValueError("`index_left` designed for intervals. Cannot index list of length 1.")

    if list_length == 2:
        return left_count

    split: int = floor((list_length - 1) / 2)
    if list_length == 3 and value == list_input[split]:
        return left_count

    if value <= list_input[split]:
        return index_left(list_input[: split + 1], split + 1, value, left_count)
    else:
        return index_left(list_input[split:], list_length - split, value, left_count + split)


# # ALTERNATIVE index_left: exhaustive search which is inferior to binary search
# def index_left_exhaustive(list_input, value, left_count=0):
#     if left_count == 0:
#         if value > list_input[-1]:
#             return len(list_input)-2
#         if value <= list_input[0]:
#             return 0
#
#     if list_input[0] < value <= list_input[1]:
#         return left_count
#     else:
#         return index_left_exhaustive(list_input[1:], value, left_count + 1)
