"""
.. ipython:: python
   :suppress:

   from rateslib.curves import *
   from datetime import datetime as dt
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pytz import UTC
from typing import Optional, Union, Callable, Any
import numpy as np
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import Holiday
from uuid import uuid4
import warnings
import json
from math import floor, comb
from rateslib import defaults
from rateslib.dual import Dual, dual_log, dual_exp, set_order_convert, DualTypes, Dual2
from rateslib.splines import PPSplineF64, PPSplineDual, PPSplineDual2
from rateslib.default import plot, NoInput
from rateslib.calendars import (
    create_calendar,
    get_calendar,
    add_tenor,
    dcf,
    CalInput,
    _DCF1d,
)
from rateslib.rateslibrs import index_left_f64

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rateslib.fx import FXForwards  # pragma: no cover


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class _Serialize:
    """
    Methods mixin for serializing and solving :class:`Curve` or :class:`LineCurve` s.
    """

    def to_json(self):
        """
        Convert the parameters of the curve to JSON format.

        Returns
        -------
        str
        """
        if self.t is NoInput.blank:
            t = None
        else:
            t = [t.strftime("%Y-%m-%d") for t in self.t]

        container = {
            "nodes": {dt.strftime("%Y-%m-%d"): v.real for dt, v in self.nodes.items()},
            "interpolation": self.interpolation if isinstance(self.interpolation, str) else None,
            "t": t,
            "c": self.spline.c if self.c_init else None,
            "id": self.id,
            "convention": self.convention,
            "endpoints": self.spline_endpoints,
            "modifier": self.modifier,
            "calendar_type": self.calendar_type,
            "ad": self.ad,
        }
        if type(self) is IndexCurve:
            container.update({"index_base": self.index_base, "index_lag": self.index_lag})

        if self.calendar_type == "null":
            container.update({"calendar": None})
        elif "named: " in self.calendar_type:
            container.update({"calendar": self.calendar_type[7:]})
        else:  # calendar type is custom
            container.update(
                {
                    "calendar": {
                        "weekmask": self.calendar.weekmask,
                        "holidays": [
                            d.item().strftime("%Y-%m-%d")
                            for d in self.calendar.holidays  # numpy/pandas timestamp to py
                        ],
                    }
                }
            )

        return json.dumps(container, default=str)

    @classmethod
    def from_json(cls, curve, **kwargs):
        """
        Reconstitute a curve from JSON.

        Parameters
        ----------
        curve : str
            The JSON string representation of the curve.

        Returns
        -------
        Curve or LineCurve
        """
        serial = json.loads(curve)

        serial["nodes"] = {
            datetime.strptime(dt, "%Y-%m-%d"): v for dt, v in serial["nodes"].items()
        }

        if serial["calendar_type"] == "custom":
            # must load and construct a custom holiday calendar from serial dates
            def parse(d: datetime):
                return Holiday("", year=d.year, month=d.month, day=d.day)

            dates = [
                parse(datetime.strptime(d, "%Y-%m-%d")) for d in serial["calendar"]["holidays"]
            ]
            serial["calendar"] = create_calendar(
                rules=dates, weekmask=serial["calendar"]["weekmask"]
            )

        if serial["t"] is not None:
            serial["t"] = [datetime.strptime(t, "%Y-%m-%d") for t in serial["t"]]

        serial = {k: v for k, v in serial.items() if v is not None}
        return cls(**{**serial, **kwargs})

    def copy(self):
        """
        Create an identical copy of the curve object.

        Returns
        -------
        Curve or LineCurve
        """
        return self.from_json(self.to_json())

    def __eq__(self, other):
        """Test two curves are identical"""
        if type(self) is not type(other):
            return False
        attrs = [attr for attr in dir(self) if attr[:1] != "_"]
        for attr in attrs:
            if callable(getattr(self, attr, None)):
                continue
            elif getattr(self, attr, None) != getattr(other, attr, None):
                return False
        return True

    def _set_ad_order(self, order):
        """
        Change the node values to float, Dual or Dual2 based on input parameter.
        """
        if order == getattr(self, "ad", None):
            return None
        elif order not in [0, 1, 2]:
            raise ValueError("`order` can only be in {0, 1, 2} for auto diff calcs.")

        self.ad = order
        self.nodes = {
            k: set_order_convert(v, order, [f"{self.id}{i}"])
            for i, (k, v) in enumerate(self.nodes.items())
        }
        self.csolve()
        return None


class Curve(_Serialize):
    """
    Curve based on DF parametrisation at given node dates with interpolation.

    Parameters
    ----------
    nodes : dict[datetime: float]
        Parameters of the curve denoted by a node date and a corresponding
        DF at that point.
    interpolation : str or callable
        The interpolation used in the non-spline section of the curve. That is the part
        of the curve between the first node in ``nodes`` and the first knot in ``t``.
        If a callable, this allows a user-defined interpolation scheme, and this must
        have the signature ``method(date, nodes)``, where ``date`` is the datetime
        whose DF will be returned and ``nodes`` is as above and is passed to the
        callable.
    t : list[datetime], optional
        The knot locations for the B-spline log-cubic interpolation section of the
        curve. If *None* all interpolation will be done by the local method specified in
        ``interpolation``.
    c : list[float], optional
        The B-spline coefficients used to define the log-cubic spline. If not given,
        which is the expected case, uses :meth:`csolve` to calculate these
        automatically.
    endpoints : str or list, optional
        The left and right endpoint constraint for the spline solution. Valid values are
        in {"natural", "not_a_knot"}. If a list, supply the left endpoint then the
        right endpoint.
    id : str, optional, set by Default
        The unique identifier to distinguish between curves in a multicurve framework.
    convention : str, optional, set by Default
        The convention of the curve for determining rates. Please see :meth:`dcf()<rateslib.calendars.dcf>` for all available options.
    modifier : str, optional
        The modification rule, in {"F", "MF", "P", "MP"}, for determining rates.
    calendar : calendar or str, optional
        The holiday calendar object to use. If str, looks up named calendar from
        static data. Used for determining rates.
    ad : int in {0, 1, 2}, optional
        Sets the automatic differentiation order. Defines whether to convert node
        values to float, :class:`~rateslib.dual.Dual` or
        :class:`~rateslib.dual.Dual2`. It is advised against
        using this setting directly. It is mainly used internally.
    Notes
    -----

    This curve type is **discount factor (DF)** based and is parametrised by a set of
    (date, DF) pairs set as ``nodes``. The initial node date of the curve is defined
    to be today and should **always** have a DF of precisely 1.0. The initial DF
    will **not** be affected by a :class:`~rateslib.solver.Solver`.

    Intermediate DFs are determined through ``interpolation``. If local interpolation
    is adopted a DF for an arbitrary date is dependent only on its immediately
    neighbouring nodes via the interpolation routine. Available options are:

    - *"log_linear"* (default for this curve type)
    - *"linear_index"*

    And also the following which are not recommended for this curve type:

    - *"linear"*,
    - *"linear_zero_rate"*,
    - *"flat_forward"*,
    - *"flat_backward"*,

    Global interpolation in the form of a **log-cubic** spline is also configurable
    with the parameters ``t``, ``c`` and ``endpoints``. See
    :ref:`splines<splines-doc>` for instruction of knot sequence calibration.
    Values before the first knot in ``t`` will be determined through the local
    interpolation method.

    For defining rates by a given tenor, the ``modifier`` and ``calendar`` arguments
    will be used. For correct scaling of the rate a ``convention`` is attached to the
    curve, which is usually one of "Act360" or "Act365F".

    Examples
    --------

    .. ipython:: python

       curve = Curve(
           nodes={
               dt(2022,1,1): 1.0,  # <- initial DF should always be 1.0
               dt(2023,1,1): 0.99,
               dt(2024,1,1): 0.979,
               dt(2025,1,1): 0.967,
               dt(2026,1,1): 0.956,
               dt(2027,1,1): 0.946,
           },
           interpolation="log_linear",
       )
       curve.plot("1d")

    .. plot::

       from rateslib.curves import *
       import matplotlib.pyplot as plt
       from datetime import datetime as dt
       import numpy as np
       curve = Curve(
           nodes={
               dt(2022,1,1): 1.0,
               dt(2023,1,1): 0.99,
               dt(2024,1,1): 0.979,
               dt(2025,1,1): 0.967,
               dt(2026,1,1): 0.956,
               dt(2027,1,1): 0.946,
           },
           interpolation="log_linear",
       )
       fig, ax, line = curve.plot("1D")
       plt.show()
    """

    _op_exp = staticmethod(dual_exp)  # Curve is DF based: log-cubic spline is exp'ed
    _op_log = staticmethod(dual_log)  # Curve is DF based: spline is applied over log
    _ini_solve = 1  # Curve is assumed to have initial DF node at 1.0 as constraint
    _base_type = "dfs"
    collateral = None

    def __init__(
        self,
        nodes: dict,
        *,
        interpolation: Union[str, Callable, NoInput] = NoInput(0),
        t: Union[list[datetime], NoInput] = NoInput(0),
        c: Union[list[float], NoInput] = NoInput(0),
        endpoints: Union[str, NoInput] = NoInput(0),
        id: Union[str, NoInput] = NoInput(0),
        convention: Union[str, NoInput] = NoInput(0),
        modifier: Union[str, NoInput] = NoInput(0),
        calendar: Union[CustomBusinessDay, str, NoInput] = NoInput(0),
        ad: int = 0,
        **kwargs,
    ):
        self.id = uuid4().hex[:5] + "_" if id is NoInput.blank else id  # 1 in a million clash
        self.nodes = nodes  # nodes.copy()
        self.node_keys = list(self.nodes.keys())
        self.node_dates = self.node_keys
        self.node_dates_posix = [_.replace(tzinfo=UTC).timestamp() for _ in self.node_dates]
        self.n = len(self.node_dates)
        for idx in range(1, self.n):
            if self.node_dates[idx - 1] >= self.node_dates[idx]:
                raise ValueError(
                    "Curve node dates are not sorted or contain duplicates. To sort directly use: `dict(sorted(nodes.items()))`"
                )
        self.interpolation = (
            defaults.interpolation[type(self).__name__]
            if interpolation is NoInput.blank
            else interpolation
        )
        if isinstance(self.interpolation, str):
            self.interpolation = self.interpolation.lower()

        # Parameters for the rate derivation
        self.convention = defaults.convention if convention is NoInput.blank else convention
        self.modifier = defaults.modifier if modifier is NoInput.blank else modifier
        self.calendar, self.calendar_type = get_calendar(calendar, kind=True)
        if self.calendar_type == "named":
            self.calendar_type = f"named: {calendar.lower()}"

        # Parameters for PPSpline
        if endpoints is NoInput.blank:
            self.spline_endpoints = [defaults.endpoints, defaults.endpoints]
        elif isinstance(endpoints, str):
            self.spline_endpoints = [endpoints.lower(), endpoints.lower()]
        else:
            self.spline_endpoints = [_.lower() for _ in endpoints]

        self.t = t
        self.c_init = False if c is NoInput.blank else True
        if t is not NoInput.blank:
            self.t_posix = [_.replace(tzinfo=UTC).timestamp() for _ in t]
            self.spline = PPSplineF64(4, self.t_posix, None if c is NoInput.blank else c)
            if len(self.t) < 10 and "not_a_knot" in self.spline_endpoints:
                raise ValueError(
                    "`endpoints` cannot be 'not_a_knot' with only 1 interior breakpoint"
                )
        else:
            self.t_posix = None
            self.spline = None

        self._set_ad_order(order=ad)

    def __getitem__(self, date: datetime):
        date_posix = date.replace(tzinfo=UTC).timestamp()
        if self.spline is None or date <= self.t[0]:
            if isinstance(self.interpolation, Callable):
                return self.interpolation(date, self.nodes.copy())
            return self._local_interp_(date_posix)
        else:
            if date > self.t[-1]:
                warnings.warn(
                    "Evaluating points on a curve beyond the endpoint of the basic "
                    "spline interval is undefined.\n"
                    f"date: {date.strftime('%Y-%m-%d')}, spline end: {self.t[-1].strftime('%Y-%m-%d')}",
                    UserWarning,
                )
            return self._op_exp(self.spline.ppev_single(date_posix))

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def _local_interp_(self, date_posix: float):
        if date_posix < self.node_dates_posix[0]:
            return 0  # then date is in the past and DF is zero
        l_index = index_left_f64(self.node_dates_posix, date_posix, None)
        node_left_posix, node_right_posix = (
            self.node_dates_posix[l_index],
            self.node_dates_posix[l_index + 1],
        )
        node_left, node_right = self.node_dates[l_index], self.node_dates[l_index + 1]
        return interpolate(
            date_posix,
            node_left_posix,
            self.nodes[node_left],
            node_right_posix,
            self.nodes[node_right],
            self.interpolation,
            self.node_dates_posix[0],
        )

    # def plot(self, *args, **kwargs):
    #     return super().plot(*args, **kwargs)

    def rate(
        self,
        effective: datetime,
        termination: Union[datetime, str],
        modifier: Union[str, bool, NoInput] = NoInput(0),
        # calendar: Optional[Union[CustomBusinessDay, str, bool]] = False,
        # convention: Optional[str] = None,
        float_spread: float = None,  # TODO: NoInput
        spread_compound_method: str = None,
    ):
        """
        Calculate the rate on the `Curve` using DFs.

        If rates are sought for dates prior to the initial node of the curve `None`
        will be returned.

        Parameters
        ----------
        effective : datetime
            The start date of the period for which to calculate the rate.
        termination : datetime or str
            The end date of the period for which to calculate the rate.
        modifier : str, optional
            The day rule if determining the termination from tenor. If `False` is
            determined from the `Curve` modifier.
        float_spread : float, optional
            A float spread can be added to the rate in certain cases.
        spread_compound_method : str in {"none_simple", "isda_compounding"}
            The method if adding a float spread.
            If *"none_simple"* is used this results in an exact calculation.
            If *"isda_compounding"* or *"isda_flat_compounding"* is used this results
            in an approximation.

        Returns
        -------
        Dual, Dual2 or float

        Notes
        -----
        Calculating rates from a curve implies that the conventions attached to the
        specific index, e.g. USD SOFR, or GBP SONIA, are applicable and these should
        be set at initialisation of the ``Curve``. Thus, the convention used to
        calculate the ``rate`` is taken from the ``Curve`` from which ``rate``
        is called.

        ``modifier`` is only used if a tenor is given as the termination.

        Major indexes, such as legacy IBORs, and modern RFRs typically use a
        ``convention`` which is either `"Act365F"` or `"Act360"`. These conventions
        do not need additional parameters, such as the `termination` of a leg,
        the `frequency` or a leg or whether it is a `stub` to calculate a DCF.

        **Adding Floating Spreads**

        An optimised method for adding floating spreads to a curve rate is provided.
        This is quite restrictive and mainly used internally to facilitate other parts
        of the library.

        - When ``spread_compound_method`` is *"none_simple"* the spread is a simple
          linear addition.
        - When using *"isda_compounding"* or *"isda_flat_compounding"* the curve is
          assumed to be comprised of RFR
          rates and an approximation is used to derive to total rate.

        Examples
        --------

        .. ipython:: python

            curve_act365f = Curve(
                nodes={
                    dt(2022, 1, 1): 1.0,
                    dt(2022, 2, 1): 0.98,
                    dt(2022, 3, 1): 0.978,
                },
                convention='Act365F'
            )
            curve_act365f.rate(dt(2022, 2, 1), dt(2022, 3, 1))

        Using a different convention will result in a different rate:

        .. ipython:: python

            curve_act360 = Curve(
                nodes={
                    dt(2022, 1, 1): 1.0,
                    dt(2022, 2, 1): 0.98,
                    dt(2022, 3, 1): 0.978,
                },
                convention='Act360'
            )
            curve_act360.rate(dt(2022, 2, 1), dt(2022, 3, 1))
        """
        modifier = self.modifier if modifier is NoInput.blank else modifier
        # calendar = self.calendar if calendar is False else calendar
        # convention = self.convention if convention is None else convention

        # Alternative solution to PR 172.
        # if effective < self.node_dates[0]:
        #     raise ValueError(
        #         "`effective` date for rate period is before the initial node date of the Curve.\n"
        #         "If you are trying to calculate a rate for an historical FloatPeriod have you "
        #         "neglected to supply appropriate `fixings`?\n"
        #         "See Documentation > Cookbook > Working with Fixings."
        #     )
        if isinstance(termination, str):
            termination = add_tenor(effective, termination, modifier, self.calendar)
        try:
            n_, d_ = self[effective], self[termination]
            df_ratio = n_ / d_
        except ZeroDivisionError:
            return None

        if termination == effective:
            raise ZeroDivisionError(f"effective: {effective}, termination: {termination}")
        n_, d_ = (df_ratio - 1), dcf(effective, termination, self.convention)
        _ = n_ / d_ * 100

        if float_spread is not None and abs(float_spread) > 1e-9:
            if spread_compound_method == "none_simple":
                return _ + float_spread / 100
            elif spread_compound_method == "isda_compounding":
                # this provides an approximated rate
                r_bar, d, n = average_rate(effective, termination, self.convention, _)
                _ = ((1 + (r_bar + float_spread / 100) / 100 * d) ** n - 1) / (n * d)
                return 100 * _
            elif spread_compound_method == "isda_flat_compounding":
                # this provides an approximated rate
                r_bar, d, n = average_rate(effective, termination, self.convention, _)
                rd = r_bar / 100 * d
                _ = (
                    (r_bar + float_spread / 100)
                    / n
                    * (comb(n, 1) + comb(n, 2) * rd + comb(n, 3) * rd**2)
                )
                return _
            else:
                raise ValueError(
                    "Must supply a valid `spread_compound_method`, when `float_spread` "
                    " is not `None`."
                )

        return _

    def csolve(self):
        """
        Solves **and sets** the coefficients, ``c``, of the :class:`PPSpline`.

        Returns
        -------
        None

        Notes
        -----
        Only impacts curves which have a knot sequence, ``t``, and a ``PPSpline``.
        Only solves if ``c`` not given at curve initialisation.

        Uses the ``spline_endpoints`` attribute on the class to determine the solving
        method.
        """
        if self.spline is None or self.c_init:
            return None

        # Get the Spline classs by data types
        if self.ad == 0:
            Spline = PPSplineF64
        elif self.ad == 1:
            Spline = PPSplineDual
        else:
            Spline = PPSplineDual2

        t_posix = self.t_posix.copy()
        tau_posix = [k.replace(tzinfo=UTC).timestamp() for k in self.nodes.keys() if k >= self.t[0]]
        y = [self._op_log(v) for k, v in self.nodes.items() if k >= self.t[0]]

        # Left side constraint
        if self.spline_endpoints[0].lower() == "natural":
            tau_posix.insert(0, self.t_posix[0])
            y.insert(0, set_order_convert(0.0, self.ad, None))
            left_n = 2
        elif self.spline_endpoints[0].lower() == "not_a_knot":
            t_posix.pop(4)
            left_n = 0
        else:
            raise NotImplementedError(
                f"Endpoint method '{self.spline_endpoints[0]}' not implemented."
            )

        # Right side constraint
        if self.spline_endpoints[1].lower() == "natural":
            tau_posix.append(self.t_posix[-1])
            y.append(set_order_convert(0, self.ad, None))
            right_n = 2
        elif self.spline_endpoints[1].lower() == "not_a_knot":
            t_posix.pop(-5)
            right_n = 0
        else:
            raise NotImplementedError(
                f"Endpoint method '{self.spline_endpoints[0]}' not implemented."
            )

        self.spline = Spline(4, t_posix, None)
        self.spline.csolve(tau_posix, y, left_n, right_n, False)
        return None

    def shift(
        self,
        spread: float,
        id: Optional[str] = None,
        composite: bool = True,
        collateral: Optional[str] = None,
    ):
        """
        Create a new curve by vertically adjusting the curve by a set number of basis
        points.

        This curve adjustment preserves the shape of the curve but moves it up or
        down as a translation.
        This method is suitable as a way to assess value changes of instruments when
        a parallel move higher or lower in yields is predicted.

        Parameters
        ----------
        spread : float, Dual, Dual2
            The number of basis points added to the existing curve.
        id : str, optional
            Set the id of the returned curve.
        composite: bool, optional
            If True will return a CompositeCurve that adds a flat curve to the existing curve.
            This results in slower calculations but the curve will maintain a dynamic
            association with the underlying curve and will change if the underlying curve changes.
        collateral: str, optional
            Designate a collateral tag for the curve which is used by other methods.

        Returns
        -------
        Curve, CompositeCurve

        Examples
        --------
        .. ipython:: python

           from rateslib.curves import Curve

        .. ipython:: python

           curve = Curve(
               nodes = {
                   dt(2022, 1, 1): 1.0,
                   dt(2023, 1, 1): 0.988,
                   dt(2024, 1, 1): 0.975,
                   dt(2025, 1, 1): 0.965,
                   dt(2026, 1, 1): 0.955,
                   dt(2027, 1, 1): 0.9475
               },
               t = [
                   dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
                   dt(2025, 1, 1),
                   dt(2026, 1, 1),
                   dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
               ],
           )
           shifted_curve = curve.shift(25)
           curve.plot("1d", comparators=[shifted_curve], labels=["orig", "shift"])

        .. plot::

           from rateslib.curves import *
           import matplotlib.pyplot as plt
           from datetime import datetime as dt
           curve = Curve(
               nodes = {
                   dt(2022, 1, 1): 1.0,
                   dt(2023, 1, 1): 0.988,
                   dt(2024, 1, 1): 0.975,
                   dt(2025, 1, 1): 0.965,
                   dt(2026, 1, 1): 0.955,
                   dt(2027, 1, 1): 0.9475
               },
               t = [
                   dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
                   dt(2025, 1, 1),
                   dt(2026, 1, 1),
                   dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
               ],
           )
           spread_curve = curve.shift(25)
           fig, ax, line = curve.plot("1d", comparators=[spread_curve], labels=["orig", "shift"])
           plt.show()

        """
        if composite:
            start, end = self.node_dates[0], self.node_dates[-1]
            days = (end - start).days
            d = _DCF1d[self.convention.upper()]
            if type(self) is Curve or type(self) is ProxyCurve:
                shifted = Curve(
                    nodes={start: 1.0, end: 1.0 / (1 + d * spread / 10000) ** days},
                    convention=self.convention,
                    calendar=self.calendar,
                    modifier=self.modifier,
                    interpolation="log_linear",
                )
            elif type(self) is LineCurve:
                shifted = LineCurve(
                    nodes={start: spread / 100.0, end: spread / 100.0},
                    convention=self.convention,
                    calendar=self.calendar,
                    modifier=self.modifier,
                    interpolation="linear",
                )
            else:  # type is IndexCurve
                shifted = IndexCurve(
                    nodes={start: 1.0, end: 1.0 / (1 + d * spread / 10000) ** days},
                    convention=self.convention,
                    calendar=self.calendar,
                    modifier=self.modifier,
                    interpolation="log_linear",
                    index_base=self.index_base,
                    index_lag=self.index_lag,
                )

            _ = CompositeCurve(curves=[self, shifted], id=id)
            _.collateral = collateral
            return _

        else:  # use non-composite method, which is faster but does not preserve a dynamic spread.
            v1v2 = [1.0] * (self.n - 1)
            n = [0] * (self.n - 1)
            d = 1 / 365 if self.convention.upper() != "ACT360" else 1 / 360
            v_new = [1.0] * (self.n)
            for i, (k, v) in enumerate(self.nodes.items()):
                if i == 0:
                    continue
                n[i - 1] = (k - self.node_dates[i - 1]).days
                v1v2[i - 1] = (self.nodes[self.node_dates[i - 1]] / v) ** (1 / n[i - 1])
                v_new[i] = v_new[i - 1] / (v1v2[i - 1] + d * spread / 10000) ** n[i - 1]

            nodes = self.nodes.copy()
            for i, (k, v) in enumerate(nodes.items()):
                nodes[k] = v_new[i]

            kwargs = {}
            if type(self) is IndexCurve:
                kwargs = {"index_base": self.index_base, "index_lag": self.index_lag}
            _ = type(self)(
                nodes=nodes,
                interpolation=self.interpolation,
                t=self.t,
                c=NoInput(0),
                endpoints=self.spline_endpoints,
                id=id or uuid4().hex[:5] + "_",  # 1 in a million clash,
                convention=self.convention,
                modifier=self.modifier,
                calendar=self.calendar,
                ad=self.ad,
                **kwargs,
            )
            _.collateral = collateral
            return _

    def _translate_nodes(self, start: datetime):
        scalar = 1 / self[start]
        new_nodes = {k: scalar * v for k, v in self.nodes.items()}

        # re-organise the nodes on the new curve
        del new_nodes[self.node_dates[0]]
        flag, i = (start >= self.node_dates[1]), 1
        while flag:
            del new_nodes[self.node_dates[i]]
            flag, i = (start >= self.node_dates[i + 1]), i + 1

        new_nodes = {start: 1.0, **new_nodes}
        return new_nodes

    def translate(self, start: datetime, t: bool = False):
        """
        Create a new curve with an initial node date moved forward keeping all else
        constant.

        This curve adjustment preserves forward curve expectations as time evolves.
        This method is suitable as a way to create a subsequent *opening* curve from a
        previous day's *closing* curve.

        Parameters
        ----------
        start : datetime
            The new initial node date for the curve, must be in the domain:
            (node_date[0], node_date[1]]
        t : bool
            Set to *True* if the initial knots of the knot sequence should be
            translated forward.

        Returns
        -------
        Curve

        Examples
        ---------
        The basic use of this function translates a curve forward in time and the
        plot demonstrates its rates are exactly the same as initially forecast.

        .. ipython:: python

           curve = Curve(
               nodes = {
                   dt(2022, 1, 1): 1.0,
                   dt(2023, 1, 1): 0.988,
                   dt(2024, 1, 1): 0.975,
                   dt(2025, 1, 1): 0.965,
                   dt(2026, 1, 1): 0.955,
                   dt(2027, 1, 1): 0.9475
               },
               t = [
                   dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
                   dt(2025, 1, 1),
                   dt(2026, 1, 1),
                   dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
               ],
           )
           translated_curve = curve.translate(dt(2022, 12, 1))
           curve.plot(
               "1d",
               comparators=[translated_curve],
               labels=["orig", "translated"],
               left=dt(2022, 12, 1),
           )

        .. plot::

           from rateslib.curves import *
           import matplotlib.pyplot as plt
           from datetime import datetime as dt
           curve = Curve(
               nodes = {
                   dt(2022, 1, 1): 1.0,
                   dt(2023, 1, 1): 0.988,
                   dt(2024, 1, 1): 0.975,
                   dt(2025, 1, 1): 0.965,
                   dt(2026, 1, 1): 0.955,
                   dt(2027, 1, 1): 0.9475
               },
               t = [
                   dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
                   dt(2025, 1, 1),
                   dt(2026, 1, 1),
                   dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
               ],
               interpolation="log_linear",
           )
           translated_curve = curve.translate(dt(2022, 12, 1))
           fig, ax, line = curve.plot("1d", comparators=[translated_curve], labels=["orig", "translated"], left=dt(2022, 12, 1))
           plt.show()
           plt.close()

        .. ipython:: python

           curve.nodes
           translated_curve.nodes

        When a curve has a log-cubic spline the knot dates can be preserved or
        translated with the ``t`` argument. Preserving the knot dates preserves the
        interpolation of the curve. A knot sequence for a mixed curve which begins
        after ``start`` will not be affected in either case.

        .. ipython:: python

           curve = Curve(
               nodes={
                   dt(2022, 1, 1): 1.0,
                   dt(2022, 2, 1): 0.999,
                   dt(2022, 3, 1): 0.9978,
                   dt(2022, 4, 1): 0.9963,
                   dt(2022, 5, 1): 0.9940
               },
               t = [dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1),
                    dt(2022, 2, 1), dt(2022, 3, 1), dt(2022, 4, 1),
                    dt(2022, 5, 1), dt(2022, 5, 1), dt(2022, 5, 1), dt(2022, 5, 1)]
           )
           translated_curve = curve.translate(dt(2022, 1, 15))
           translated_curve2 = curve.translate(dt(2022, 1, 15), t=True)
           curve.plot("1d", left=dt(2022, 1, 15), comparators=[translated_curve, translated_curve2], labels=["orig", "translated", "translated2"])

        .. plot::

           from rateslib.curves import *
           import matplotlib.pyplot as plt
           from datetime import datetime as dt
           curve = Curve(
               nodes={
                   dt(2022, 1, 1): 1.0,
                   dt(2022, 2, 1): 0.999,
                   dt(2022, 3, 1): 0.9978,
                   dt(2022, 4, 1): 0.9963,
                   dt(2022, 5, 1): 0.9940
               },
               t = [dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1),
                    dt(2022, 2, 1), dt(2022, 3, 1), dt(2022, 4, 1),
                    dt(2022, 5, 1), dt(2022, 5, 1), dt(2022, 5, 1), dt(2022, 5, 1)]
           )
           translated = curve.translate(dt(2022, 1, 15))
           translated2 = curve.translate(dt(2022, 1, 15), t=True)
           fig, ax, line = curve.plot("1d", left=dt(2022, 1, 15), comparators=[translated, translated2], labels=["orig", "translated", "translated2"])
           plt.show()

        """
        if start <= self.node_dates[0]:
            raise ValueError("Cannot translate into the past. Review the docs.")

        new_nodes = self._translate_nodes(start)

        # re-organise the t-knot sequence
        if self.t is NoInput.blank:
            new_t: Union[list[datetime], NoInput] = NoInput(0)
        else:
            new_t = self.t.copy()

            if start <= new_t[0]:
                pass  # do nothing to t
            elif new_t[0] < start < new_t[4]:
                if t:
                    for i in range(4):
                        new_t[i] = start  # adjust left side of t to start
            elif new_t[4] <= start:
                raise ValueError(
                    "Cannot translate spline knots for given `start`, review the docs."
                )

        kwargs = {}
        if type(self) is IndexCurve:
            kwargs = {
                "index_base": self.index_value(start),
                "index_lag": self.index_lag,
            }
        new_curve = type(self)(
            nodes=new_nodes,
            interpolation=self.interpolation,
            t=new_t,
            c=NoInput(0),
            endpoints=self.spline_endpoints,
            modifier=self.modifier,
            calendar=self.calendar,
            convention=self.convention,
            id=NoInput(0),
            ad=self.ad,
            **kwargs,
        )
        return new_curve

    def _roll_nodes(self, tenor: datetime, days: int):
        """
        Roll nodes by adding days to each one and scaling DF values.

        Parameters
        ----------
        tenor : datetime
            The intended roll datetime
        days : int
            The number of days between ``tenor`` and initial curve node date.

        Returns
        -------
        dict
        """
        on_rate = self.rate(self.node_dates[0], "1d", "NONE")
        d = 1 / 365 if self.convention.upper() != "ACT360" else 1 / 360
        scalar = 1 / ((1 + on_rate * d / 100) ** days)
        new_nodes = {k + timedelta(days=days): v * scalar for k, v in self.nodes.items()}
        if tenor > self.node_dates[0]:
            new_nodes = {self.node_dates[0]: 1.0, **new_nodes}
        return new_nodes

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def roll(self, tenor: Union[datetime, str]):
        """
        Create a new curve with its shape translated in time but an identical initial
        node date.

        This curve adjustment is a simulation of a future state of the market where
        forward rates are assumed to have moved so that the present day's curve shape
        is reflected in the future (or the past). This is often used in trade
        strategy analysis.

        Parameters
        ----------
        tenor : datetime or str
            The date or tenor by which to roll the curve. If a tenor, as str, will
            derive the datetime as measured from the initial node date. If supplying a
            negative tenor, or a past datetime, there is a limit to how far back the
            curve can be rolled - it will first roll backwards and then attempt to
            :meth:`translate` forward to maintain the initial node date.

        Returns
        -------
        Curve

        Examples
        ---------
        The basic use of this function translates a curve forward in time and the
        plot demonstrates its rates are exactly the same as initially forecast.

        .. ipython:: python

           curve = Curve(
               nodes = {
                   dt(2022, 1, 1): 1.0,
                   dt(2023, 1, 1): 0.988,
                   dt(2024, 1, 1): 0.975,
                   dt(2025, 1, 1): 0.965,
                   dt(2026, 1, 1): 0.955,
                   dt(2027, 1, 1): 0.9475
               },
               t = [
                   dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
                   dt(2025, 1, 1),
                   dt(2026, 1, 1),
                   dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
               ],
           )
           rolled_curve = curve.roll("6m")
           rolled_curve2 = curve.roll("-6m")
           curve.plot(
               "1d",
               comparators=[rolled_curve, rolled_curve2],
               labels=["orig", "rolled", "rolled2"],
               right=dt(2026, 7, 1),
           )

        .. plot::

           from rateslib.curves import *
           import matplotlib.pyplot as plt
           from datetime import datetime as dt
           curve = Curve(
               nodes = {
                   dt(2022, 1, 1): 1.0,
                   dt(2023, 1, 1): 0.988,
                   dt(2024, 1, 1): 0.975,
                   dt(2025, 1, 1): 0.965,
                   dt(2026, 1, 1): 0.955,
                   dt(2027, 1, 1): 0.9475
               },
               t = [
                   dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
                   dt(2025, 1, 1),
                   dt(2026, 1, 1),
                   dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
               ],
           )
           rolled_curve = curve.roll("6m")
           rolled_curve2 = curve.roll("-6m")
           fig, ax, line = curve.plot("1d", comparators=[rolled_curve, rolled_curve2], labels=["orig", "rolled", "rolled2"], right=dt(2026,6,30))
           plt.show()

        """
        if isinstance(tenor, str):
            tenor = add_tenor(self.node_dates[0], tenor, "NONE", NoInput(0))

        if tenor == self.node_dates[0]:
            return self.copy()

        days = (tenor - self.node_dates[0]).days
        new_nodes = self._roll_nodes(tenor, days)
        if self.t is not NoInput.blank:
            new_t = [_ + timedelta(days=days) for _ in self.t]
        else:
            new_t = NoInput(0)
        if type(self) is IndexCurve:
            xtra = dict(index_lag=self.index_lag, index_base=self.index_base)
        else:
            xtra = {}
        new_curve = type(self)(
            nodes=new_nodes,
            interpolation=self.interpolation,
            t=new_t,
            c=NoInput(0),
            endpoints=self.spline_endpoints,
            modifier=self.modifier,
            calendar=self.calendar,
            convention=self.convention,
            id=None,
            ad=self.ad,
            **xtra,
        )
        if tenor > self.node_dates[0]:
            return new_curve
        else:  # tenor < self.node_dates[0]
            return new_curve.translate(self.node_dates[0])

    # PLOTTING METHODS

    def plot(
        self,
        tenor: str,
        right: Union[datetime, str, NoInput] = NoInput(0),
        left: Union[datetime, str, NoInput] = NoInput(0),
        comparators: list[Curve] = [],
        difference: bool = False,
        labels: list[str] = [],
    ):
        """
        Plot given forward tenor rates from the curve.

        Parameters
        ----------
        tenor : str
            The tenor of the forward rates to plot, e.g. "1D", "3M".
        right : datetime or str, optional
            The right bound of the graph. If given as str should be a tenor format
            defining a point measured from the initial node date of the curve.
            Defaults to the final node of the curve minus the ``tenor``.
        left : datetime or str, optional
            The left bound of the graph. If given as str should be a tenor format
            defining a point measured from the initial node date of the curve.
            Defaults to the initial node of the curve.
        comparators: list[Curve]
            A list of curves which to include on the same plot as comparators.
        difference : bool
            Whether to plot as comparator minus base curve or outright curve levels in
            plot. Default is `False`.
        labels : list[str]
            A list of strings associated with the plot and comparators. Must be same
            length as number of plots.

        Returns
        -------
        (fig, ax, line) : Matplotlib.Figure, Matplotplib.Axes, Matplotlib.Lines2D
        """
        if left is NoInput.blank:
            left_: datetime = self.node_dates[0]
        elif isinstance(left, str):
            left_ = add_tenor(self.node_dates[0], left, "NONE", NoInput(0))
        elif isinstance(left, datetime):
            left_ = left
        else:
            raise ValueError("`left` must be supplied as datetime or tenor string.")

        if right is NoInput.blank:
            right_: datetime = add_tenor(self.node_dates[-1], "-" + tenor, "NONE", NoInput(0))
        elif isinstance(right, str):
            right_ = add_tenor(self.node_dates[0], right, "NONE", NoInput(0))
        elif isinstance(right, datetime):
            right_ = right
        else:
            raise ValueError("`right` must be supplied as datetime or tenor string.")

        points: int = (right_ - left_).days
        x = [left_ + timedelta(days=i) for i in range(points)]
        rates = [self.rate(_, tenor) for _ in x]
        if not difference:
            y = [rates]
            if comparators is not None:
                for comparator in comparators:
                    y.append([comparator.rate(_, tenor) for _ in x])
        elif difference and len(comparators) > 0:
            y = []
            for comparator in comparators:
                diff = [comparator.rate(_, tenor) - rates[i] for i, _ in enumerate(x)]
                y.append(diff)
        return plot(x, y, labels)

    def _plot_fx(
        self,
        curve_foreign: Curve,
        fx_rate: Union[float, Dual],
        fx_settlement: Union[datetime, NoInput] = NoInput(0),
        left: datetime = None,
        right: datetime = None,
        points: int = None,
    ):  # pragma: no cover
        """
        Debugging method?
        """

        def forward_fx(date, curve_domestic, curve_foreign, fx_rate, fx_settlement):
            _ = self[date] / curve_foreign[date]
            if fx_settlement is not NoInput.blank:
                _ *= curve_foreign[fx_settlement] / curve_domestic[fx_settlement]
            _ *= fx_rate
            return _

        left, right = self.node_dates[0], self.node_dates[-1]
        points = (right - left).days
        x = [left + timedelta(days=i) for i in range(points)]
        rates = [forward_fx(_, self, curve_foreign, fx_rate, fx_settlement) for _ in x]
        return plot(x, [rates])

    def _set_node_vector(self, vector: list[DualTypes], ad):
        """Used to update curve values during a Solver iteration. ``ad`` in {1, 2}."""
        DualType = Dual if ad == 1 else Dual2
        DualArgs = ([],) if ad == 1 else ([], [])
        base_obj = DualType(0.0, [f"{self.id}{i}" for i in range(self.n)], *DualArgs)
        ident = np.eye(self.n)

        if self._ini_solve == 1:
            # then the first node on the Curve is not updated but
            # set it as a dual type with consistent vars.
            self.nodes[self.node_keys[0]] = DualType.vars_from(
                base_obj,
                self.nodes[self.node_keys[0]].real,
                base_obj.vars,
                ident[0, :].tolist(),
                *DualArgs[1:],
            )

        for i, k in enumerate(self.node_keys[self._ini_solve :]):
            self.nodes[k] = DualType.vars_from(
                base_obj,
                vector[i].real,
                base_obj.vars,
                ident[i + self._ini_solve, :].tolist(),
                *DualArgs[1:],
            )
        self.csolve()

    def _get_node_vector(self):
        """Get a 1d array of variables associated with nodes of this object updated by Solver"""
        return np.array(list(self.nodes.values())[self._ini_solve:])

    def _get_node_vars(self):
        """Get the variable names of elements updated by a Solver"""
        return tuple((f"{self.id}{i}" for i in range(self._ini_solve, self.n)))


class LineCurve(Curve):
    """
    Curve based on value parametrisation at given node dates with interpolation.

    Parameters
    ----------
    nodes : dict[datetime: float]
        Parameters of the curve denoted by a node date and a corresponding
        value at that point.
    interpolation : str in {"log_linear", "linear"} or callable
        The interpolation used in the non-spline section of the curve. That is the part
        of the curve between the first node in ``nodes`` and the first knot in ``t``.
        If a callable, this allows a user-defined interpolation scheme, and this must
        have the signature ``method(date, nodes)``, where ``date`` is the datetime
        whose DF will be returned and ``nodes`` is as above and is passed to the
        callable.
    t : list[datetime], optional
        The knot locations for the B-spline cubic interpolation section of the
        curve. If *None* all interpolation will be done by the method specified in
        ``interpolation``.
    c : list[float], optional
        The B-spline coefficients used to define the log-cubic spline. If not given,
        which is the expected case, uses :meth:`csolve` to calculate these
        automatically.
    endpoints : str or list, optional
        The left and right endpoint constraint for the spline solution. Valid values are
        in {"natural", "not_a_knot"}. If a list, supply the left endpoint then the
        right endpoint.
    id : str, optional, set by Default
        The unique identifier to distinguish between curves in a multi-curve framework.
    convention : str, optional,
        This argument is **not used** by :class:`LineCurve`. It is included for
        signature consistency with :class:`Curve`.
    modifier : str, optional
        This argument is **not used** by :class:`LineCurve`. It is included for
        signature consistency with :class:`Curve`.
    calendar : calendar or str, optional
        This argument is **not used** by :class:`LineCurve`. It is included for
        signature consistency with :class:`Curve`.
    ad : int in {0, 1, 2}, optional
        Sets the automatic differentiation order. Defines whether to convert node
        values to float, :class:`Dual` or :class:`Dual2`. It is advised against
        using this setting directly. It is mainly used internally.

    Notes
    -----

    This curve type is **value** based and it is parametrised by a set of
    (date, value) pairs set as ``nodes``. The initial node date of the curve is defined
    to be today, and can take a general value. The initial value
    will be affected by a :class:`~rateslib.solver.Solver`.

    .. note::

       This curve type can only ever be used for **forecasting** rates and projecting
       cashflow calculations. It cannot be used to discount cashflows becuase it is
       not DF based and there is no mathematical one-to-one conversion available to
       imply DFs.

    Intermediate values are determined through ``interpolation``. If local interpolation
    is adopted a value for an arbitrary date is dependent only on its immediately
    neighbouring nodes via the interpolation routine. Available options are:

    - *"linear"* (default for this curve type)
    - *"log_linear"* (useful for values that exponential, e.g. stock indexes or GDP)
    - *"flat_forward"*, (useful for replicating a DF based log-linear type curve)
    - *"flat_backward"*,

    And also the following which are not recommended for this curve type:

    - *"linear_index"*
    - *"linear_zero_rate"*,

    Global interpolation in the form of a **cubic** spline is also configurable
    with the parameters ``t``, ``c`` and ``endpoints``. See
    :ref:`splines<splines-doc>` for instruction of knot sequence calibration.
    Values before the first knot in ``t`` will be determined through the local
    interpolation method.

    This curve type cannot return arbitrary tenor rates. It will only return a single
    value which is applicable to that date. It is recommended to review
    :ref:`RFR and IBOR Indexing<c-curves-ibor-rfr>` to ensure indexing is done in a
    way that is consistent with internal instrument configuration.

    Examples
    --------

    .. ipython:: python

       line_curve = LineCurve(
           nodes={
               dt(2022,1,1): 0.975,  # <- initial value is general
               dt(2023,1,1): 1.10,
               dt(2024,1,1): 1.22,
               dt(2025,1,1): 1.14,
               dt(2026,1,1): 1.03,
               dt(2027,1,1): 1.03,
           },
           interpolation="linear",
       )
       line_curve.plot("1d")

    .. plot::

       from rateslib.curves import *
       import matplotlib.pyplot as plt
       from datetime import datetime as dt
       import numpy as np
       line_curve = LineCurve(
           nodes={
               dt(2022,1,1): 0.975,  # <- initial value is general
               dt(2023,1,1): 1.10,
               dt(2024,1,1): 1.22,
               dt(2025,1,1): 1.14,
               dt(2026,1,1): 1.03,
               dt(2027,1,1): 1.03,
           },
           interpolation="linear",
       )
       fig, ax, line = line_curve.plot("1D")
       plt.show()

    """

    _op_exp = staticmethod(lambda x: x)  # LineCurve spline is not log based so no exponent needed
    _op_log = staticmethod(lambda x: x)  # LineCurve spline is not log based so no log needed
    _ini_solve = 0  # No constraint placed on initial node in Solver
    _base_type = "values"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    # def plot(self, *args, **kwargs):
    #     return super().plot(*args, **kwargs)

    def rate(
        self,
        effective: datetime,
        *args,
    ):
        """
        Return the curve value for a given date.

        Note `LineCurve` s do not determine interest rates via DFs therefore do not
        have the concept of tenors or termination dates - the rate is simply the value
        attributed to that date on the curve.

        Parameters
        ----------
        effective : datetime
            The date of the curve value, which will be interpolated if necessary.

        Returns
        -------
        float, Dual, or Dual2
        """
        return self[effective]

    def shift(
        self,
        spread: float,
        id: Optional[str] = None,
        composite: bool = True,
        collateral: Optional[str] = None,
    ):
        """
        Raise or lower the curve in parallel by a set number of basis points.

        Parameters
        ----------
        spread : float, Dual, Dual2
            The number of basis points added to the existing curve.
        id : str, optional
            Set the id of the returned curve.
        composite: bool, optional
            If True will return a CompositeCurve that adds a flat curve to the existing curve.
            This results in slower calculations but the curve will maintain a dynamic
            association with the underlying curve and will change if the underlying curve changes.
        collateral: str, optional
            Designate a collateral tag for the curve which is used by other methods.

        Returns
        -------
        LineCurve

        Examples
        --------
        .. ipython:: python

           from rateslib.curves import LineCurve

        .. ipython:: python

           line_curve = LineCurve(
               nodes = {
                   dt(2022, 1, 1): 1.7,
                   dt(2023, 1, 1): 1.65,
                   dt(2024, 1, 1): 1.4,
                   dt(2025, 1, 1): 1.3,
                   dt(2026, 1, 1): 1.25,
                   dt(2027, 1, 1): 1.35
               },
               t = [
                   dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
                   dt(2025, 1, 1),
                   dt(2026, 1, 1),
                   dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
               ],
           )
           spread_curve = line_curve.shift(25)
           line_curve.plot("1d", comparators=[spread_curve])

        .. plot::

           from rateslib.curves import *
           import matplotlib.pyplot as plt
           from datetime import datetime as dt
           line_curve = LineCurve(
               nodes = {
                   dt(2022, 1, 1): 1.7,
                   dt(2023, 1, 1): 1.65,
                   dt(2024, 1, 1): 1.4,
                   dt(2025, 1, 1): 1.3,
                   dt(2026, 1, 1): 1.25,
                   dt(2027, 1, 1): 1.35
               },
               t = [
                   dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
                   dt(2025, 1, 1),
                   dt(2026, 1, 1),
                   dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
               ],
           )
           spread_curve = line_curve.shift(25)
           fig, ax, line = line_curve.plot("1d", comparators=[spread_curve])
           plt.show()

        """
        if composite:
            return super().shift(spread, id, composite, collateral)
        _ = LineCurve(
            nodes={k: v + spread / 100 for k, v in self.nodes.items()},
            interpolation=self.interpolation,
            t=self.t,
            c=NoInput(0),
            endpoints=self.spline_endpoints,
            id=id,
            convention=self.convention,
            modifier=self.modifier,
            calendar=self.calendar,
            ad=self.ad,
        )
        _.collateral = collateral
        return _

    def _translate_nodes(self, start: datetime):
        new_nodes = self.nodes.copy()

        # re-organise the nodes on the new curve
        del new_nodes[self.node_dates[0]]
        flag, i = (start >= self.node_dates[1]), 1
        while flag:
            del new_nodes[self.node_dates[i]]
            flag, i = (start >= self.node_dates[i + 1]), i + 1

        new_nodes = {start: self[start], **new_nodes}
        return new_nodes

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def translate(self, start: datetime, t: bool = False):
        """
        Create a new curve with an initial node date moved forward keeping all else
        constant.

        This curve adjustment preserves forward curve expectations as time evolves.
        This method is suitable as a way to create a subsequent *opening* curve from a
        previous day's *closing* curve.

        Parameters
        ----------
        start : datetime
            The new initial node date for the curve, must be in the domain:
            (node_date[0], node_date[1]]
        t : bool
            Set to *True* if the initial knots of the knot sequence should be
            translated forward.

        Returns
        -------
        Curve

        Examples
        ---------
        The basic use of this function translates a curve forward in time and the
        plot demonstrates its rates are exactly the same as initially forecast.

        .. ipython:: python

           line_curve = LineCurve(
               nodes = {
                   dt(2022, 1, 1): 1.7,
                   dt(2023, 1, 1): 1.65,
                   dt(2024, 1, 1): 1.4,
                   dt(2025, 1, 1): 1.3,
                   dt(2026, 1, 1): 1.25,
                   dt(2027, 1, 1): 1.35
               },
               t = [
                   dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
                   dt(2025, 1, 1),
                   dt(2026, 1, 1),
                   dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
               ],
           )
           translated_curve = line_curve.translate(dt(2022, 12, 1))
           line_curve.plot("1d", comparators=[translated_curve], left=dt(2022, 12, 1))

        .. plot::

           from rateslib.curves import *
           import matplotlib.pyplot as plt
           from datetime import datetime as dt
           line_curve = LineCurve(
               nodes = {
                   dt(2022, 1, 1): 1.7,
                   dt(2023, 1, 1): 1.65,
                   dt(2024, 1, 1): 1.4,
                   dt(2025, 1, 1): 1.3,
                   dt(2026, 1, 1): 1.25,
                   dt(2027, 1, 1): 1.35
               },
               t = [
                   dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
                   dt(2025, 1, 1),
                   dt(2026, 1, 1),
                   dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
               ],
           )
           translated_curve = line_curve.translate(dt(2022, 12, 1))
           fig, ax, line = line_curve.plot("1d", comparators=[translated_curve], labels=["orig", "translated"], left=dt(2022, 12,1))
           plt.show()
           plt.close()

        .. ipython:: python

           line_curve.nodes
           translated_curve.nodes

        When a line curve has a cubic spline the knot dates can be preserved or
        translated with the ``t`` argument. Preserving the knot dates preserves the
        interpolation of the curve. A knot sequence for a mixed curve which begins
        after ``start`` will not be affected in either case.

        .. ipython:: python

           curve = LineCurve(
               nodes={
                   dt(2022, 1, 1): 1.5,
                   dt(2022, 2, 1): 1.6,
                   dt(2022, 3, 1): 1.4,
                   dt(2022, 4, 1): 1.2,
                   dt(2022, 5, 1): 1.1,
               },
               t = [dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1),
                    dt(2022, 2, 1), dt(2022, 3, 1), dt(2022, 4, 1),
                    dt(2022, 5, 1), dt(2022, 5, 1), dt(2022, 5, 1), dt(2022, 5, 1)]
           )
           translated_curve = curve.translate(dt(2022, 1, 15))
           translated_curve2 = curve.translate(dt(2022, 1, 15), t=True)
           curve.plot("1d", left=dt(2022, 1, 15), comparators=[translated_curve, translated_curve2], labels=["orig", "translated", "translated2"])

        .. plot::

           from rateslib.curves import *
           import matplotlib.pyplot as plt
           from datetime import datetime as dt
           curve = LineCurve(
               nodes={
                   dt(2022, 1, 1): 1.5,
                   dt(2022, 2, 1): 1.6,
                   dt(2022, 3, 1): 1.4,
                   dt(2022, 4, 1): 1.2,
                   dt(2022, 5, 1): 1.1,
               },
               t = [dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1),
                    dt(2022, 2, 1), dt(2022, 3, 1), dt(2022, 4, 1),
                    dt(2022, 5, 1), dt(2022, 5, 1), dt(2022, 5, 1), dt(2022, 5, 1)]
           )
           translated = curve.translate(dt(2022, 1, 15))
           translated2 = curve.translate(dt(2022, 1, 15), t=True)
           fig, ax, line = curve.plot("1d", left=dt(2022, 1, 15), comparators=[translated, translated2], labels=["orig", "translated", "translated2"])
           plt.show()

        """
        return super().translate(start, t)

    def _roll_nodes(self, tenor: datetime, days: int):
        new_nodes = {k + timedelta(days=days): v for k, v in self.nodes.items()}
        if tenor > self.node_dates[0]:
            new_nodes = {self.node_dates[0]: self[self.node_dates[0]], **new_nodes}
        return new_nodes

    def roll(self, tenor: Union[datetime, str]):
        """
        Create a new curve with its shape translated in time

        This curve adjustment is a simulation of a future state of the market where
        forward rates are assumed to have moved so that the present day's curve shape
        is reflected in the future (or the past). This is often used in trade
        strategy analysis.

        Parameters
        ----------
        tenor : datetime or str
            The date or tenor by which to roll the curve. If a tenor, as str, will
            derive the datetime as measured from the initial node date. If supplying a
            negative tenor, or a past datetime, there is a limit to how far back the
            curve can be rolled - it will first roll backwards and then attempt to
            :meth:`translate` forward to maintain the initial node date.

        Returns
        -------
        Curve

        Examples
        ---------
        The basic use of this function translates a curve forward in time and the
        plot demonstrates its rates are exactly the same as initially forecast.

        .. ipython:: python

           line_curve = LineCurve(
               nodes = {
                   dt(2022, 1, 1): 1.7,
                   dt(2023, 1, 1): 1.65,
                   dt(2024, 1, 1): 1.4,
                   dt(2025, 1, 1): 1.3,
                   dt(2026, 1, 1): 1.25,
                   dt(2027, 1, 1): 1.35
               },
               t = [
                   dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
                   dt(2025, 1, 1),
                   dt(2026, 1, 1),
                   dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
               ],
           )
           rolled_curve = line_curve.roll("6m")
           rolled_curve2 = line_curve.roll("-6m")
           line_curve.plot(
               "1d",
               comparators=[rolled_curve, rolled_curve2],
               labels=["orig", "rolled", "rolled2"],
               right=dt(2026, 7, 1)
           )

        .. plot::

           from rateslib.curves import *
           import matplotlib.pyplot as plt
           from datetime import datetime as dt
           line_curve = LineCurve(
               nodes = {
                   dt(2022, 1, 1): 1.7,
                   dt(2023, 1, 1): 1.65,
                   dt(2024, 1, 1): 1.4,
                   dt(2025, 1, 1): 1.3,
                   dt(2026, 1, 1): 1.25,
                   dt(2027, 1, 1): 1.35
               },
               t = [
                   dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
                   dt(2025, 1, 1),
                   dt(2026, 1, 1),
                   dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
               ],
           )
           rolled_curve = line_curve.roll("6m")
           rolled_curve2 = line_curve.roll("-6m")
           fig, ax, line = line_curve.plot("1d", comparators=[rolled_curve, rolled_curve2], labels=["orig", "rolled", "rolled2"], right=dt(2026,6,30))
           plt.show()
           plt.close()

        """
        return super().roll(tenor)


class IndexCurve(Curve):
    """
    A subclass of :class:`~rateslib.curves.Curve` with an ``index_base`` value for
    index calculations.

    Parameters
    ----------
    args : tuple
        Position arguments required by :class:`Curve`.
    index_base: float
    index_lag : int
        Number of months of by which the index lags the date. For example if the initial
        curve node date is 1st Sep 2021 based on the inflation index published
        17th June 2023 then the lag is 3 months.
    kwargs : dict
        Keyword arguments required by :class:`Curve`.
    """

    def __init__(
        self,
        *args,
        index_base: Union[float, NoInput] = NoInput(0),
        index_lag: Union[int, NoInput] = NoInput(0),
        **kwargs,
    ):
        self.index_lag = defaults.index_lag if index_lag is NoInput.blank else index_lag
        self.index_base = index_base
        if self.index_base is NoInput.blank:
            raise ValueError("`index_base` must be given for IndexCurve.")
        super().__init__(*args, **{**{"interpolation": "linear_index"}, **kwargs})

    def index_value(self, date: datetime, interpolation: str = "daily"):
        """
        Calculate the accrued value of the index from the ``index_base``.

        Parameters
        ----------
        date : datetime
            The date for which the index value will be returned.
        interpolation : str in {"monthly", "daily"}
            The method for returning the index value. Monthly returns the index value
            for the start of the month and daily returns a value based on the
            interpolation between nodes (which is recommended *"linear_index*) for
            :class:`InflationCurve`.

        Returns
        -------
        None, float, Dual, Dual2

        Examples
        --------
        The SWESTR rate, for reference value date 6th Sep 2021, was published as
        2.375% and the RFR index for that date was 100.73350964. Below we calculate
        the value that was published for the RFR index on 7th Sep 2021 by the Riksbank.

        .. ipython:: python

           index_curve = IndexCurve(
               nodes={
                   dt(2021, 9, 6): 1.0,
                   dt(2021, 9, 7): 1 / (1 + 2.375/36000)
               },
               index_base=100.73350964,
               convention="Act360",
           )
           index_curve.rate(dt(2021, 9, 6), "1d")
           index_curve.index_value(dt(2021, 9, 7))
        """
        if interpolation.lower() == "daily":
            date_ = date
        elif interpolation.lower() == "monthly":
            date_ = datetime(date.year, date.month, 1)
        else:
            raise ValueError("`interpolation` for `index_value` must be in {'daily', 'monthly'}.")
        if date_ < self.node_dates[0]:
            return 0.0
            # return zero for index dates in the past
            # the proper way for instruments to deal with this is to supply i_fixings
        elif date_ == self.node_dates[0]:
            return self.index_base
        else:
            return self.index_base * 1 / self[date_]

    def plot_index(
        self,
        right: Union[datetime, str, NoInput] = NoInput(0),
        left: Union[datetime, str, NoInput] = NoInput(0),
        comparators: list[Curve] = [],
        difference: bool = False,
        labels: list[str] = [],
    ):
        """
        Plot given forward tenor rates from the curve.

        Parameters
        ----------
        tenor : str
            The tenor of the forward rates to plot, e.g. "1D", "3M".
        right : datetime or str, optional
            The right bound of the graph. If given as str should be a tenor format
            defining a point measured from the initial node date of the curve.
            Defaults to the final node of the curve minus the ``tenor``.
        left : datetime or str, optional
            The left bound of the graph. If given as str should be a tenor format
            defining a point measured from the initial node date of the curve.
            Defaults to the initial node of the curve.
        comparators: list[Curve]
            A list of curves which to include on the same plot as comparators.
        difference : bool
            Whether to plot as comparator minus base curve or outright curve levels in
            plot. Default is `False`.
        labels : list[str]
            A list of strings associated with the plot and comparators. Must be same
            length as number of plots.

        Returns
        -------
        (fig, ax, line) : Matplotlib.Figure, Matplotplib.Axes, Matplotlib.Lines2D
        """
        if left is NoInput.blank:
            left_: datetime = self.node_dates[0]
        elif isinstance(left, str):
            left_ = add_tenor(self.node_dates[0], left, "NONE", NoInput(0))
        elif isinstance(left, datetime):
            left_ = left
        else:
            raise ValueError("`left` must be supplied as datetime or tenor string.")

        if right is NoInput.blank:
            right_: datetime = self.node_dates[-1]
        elif isinstance(right, str):
            right_ = add_tenor(self.node_dates[0], right, "NONE", NoInput(0))
        elif isinstance(right, datetime):
            right_ = right
        else:
            raise ValueError("`right` must be supplied as datetime or tenor string.")

        points: int = (right_ - left_).days + 1
        x = [left_ + timedelta(days=i) for i in range(points)]
        rates = [self.index_value(_) for _ in x]
        if not difference:
            y = [rates]
            if comparators is not None:
                for comparator in comparators:
                    y.append([comparator.index_value(_) for _ in x])
        elif difference and len(comparators) > 0:
            y = []
            for comparator in comparators:
                diff = [comparator.index_value(_) - rates[i] for i, _ in enumerate(x)]
                y.append(diff)
        return plot(x, y, labels)


class CompositeCurve(IndexCurve):
    """
    A dynamic composition of a sequence of other curves.

    .. note::
       Can only composite curves of the same type: :class:`Curve`, :class:`IndexCurve`
       or :class:`LineCurve`. Other curve parameters such as ``modifier``, ``calendar``
       and ``convention`` must also match.

    Parameters
    ----------
    curves : sequence of :class:`Curve`, :class:`LineCurve` or :class:`IndexCurve`
        The curves to be composited.
    id : str, optional, set by Default
        The unique identifier to distinguish between curves in a multi-curve framework.

    Examples
    --------
    Composite two :class:`LineCurve` s. Here, simulating the effect of adding
    quarter-end turns to a cubic spline interpolator, which is otherwise difficult to
    mathematically derive.

    .. ipython:: python

       from rateslib.curves import LineCurve, CompositeCurve
       line_curve1 = LineCurve(
           nodes={
               dt(2022, 1, 1): 2.5,
               dt(2023, 1, 1): 3.5,
               dt(2024, 1, 1): 3.0,
           },
           t=[dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1),
              dt(2023, 1, 1),
              dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1)],
       )
       line_curve2 = LineCurve(
           nodes={
               dt(2022, 1, 1): 0,
               dt(2022, 3, 31): -0.2,
               dt(2022, 4, 1): 0,
               dt(2022, 6, 30): -0.2,
               dt(2022, 7, 1): 0,
               dt(2022, 9, 30): -0.2,
               dt(2022, 10, 1): 0,
               dt(2022, 12, 31): -0.2,
               dt(2023, 1, 1): 0,
               dt(2023, 3, 31): -0.2,
               dt(2023, 4, 1): 0,
               dt(2023, 6, 30): -0.2,
               dt(2023, 7, 1): 0,
               dt(2023, 9, 30): -0.2,
           },
           interpolation="flat_forward",
       )
       curve = CompositeCurve([line_curve1, line_curve2])
       curve.plot("1d")

    .. plot::

       from rateslib.curves import LineCurve, CompositeCurve
       import matplotlib.pyplot as plt
       from datetime import datetime as dt
       line_curve1 = LineCurve(
           nodes={
               dt(2022, 1, 1): 2.5,
               dt(2023, 1, 1): 3.5,
               dt(2024, 1, 1): 3.0,
           },
           t=[dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1),
              dt(2023, 1, 1),
              dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1)],
       )
       line_curve2 = LineCurve(
           nodes={
               dt(2022, 1, 1): 0,
               dt(2022, 3, 31): -0.2,
               dt(2022, 4, 1): 0,
               dt(2022, 6, 30): -0.2,
               dt(2022, 7, 1): 0,
               dt(2022, 9, 30): -0.2,
               dt(2022, 10, 1): 0,
               dt(2022, 12, 31): -0.2,
               dt(2023, 1, 1): 0,
               dt(2023, 3, 31): -0.2,
               dt(2023, 4, 1): 0,
               dt(2023, 6, 30): -0.2,
               dt(2023, 7, 1): 0,
               dt(2023, 9, 30): -0.2,
           },
           interpolation="flat_forward",
       )
       curve = CompositeCurve([line_curve1, line_curve2])
       fig, ax, line = curve.plot("1D")
       plt.show()

    We can also composite DF based curves by using a fast approximation or an
    exact match.

    .. ipython:: python

       from rateslib.curves import Curve, CompositeCurve
       curve1 = Curve(
           nodes={
               dt(2022, 1, 1): 1.0,
               dt(2023, 1, 1): 0.98,
               dt(2024, 1, 1): 0.965,
               dt(2025, 1, 1): 0.955
           },
           t=[dt(2023, 1, 1), dt(2023, 1, 1), dt(2023, 1, 1), dt(2023, 1, 1),
              dt(2024, 1, 1),
              dt(2025, 1, 1), dt(2025, 1, 1), dt(2025, 1, 1), dt(2025, 1, 1)],
       )
       curve2 =Curve(
           nodes={
               dt(2022, 1, 1): 1.0,
               dt(2022, 6, 30): 1.0,
               dt(2022, 7, 1): 0.999992,
               dt(2022, 12, 31): 0.999992,
               dt(2023, 1, 1): 0.999984,
               dt(2023, 6, 30): 0.999984,
               dt(2023, 7, 1): 0.999976,
               dt(2023, 12, 31): 0.999976,
               dt(2024, 1, 1): 0.999968,
               dt(2024, 6, 30): 0.999968,
               dt(2024, 7, 1): 0.999960,
               dt(2025, 1, 1): 0.999960,
           },
       )
       curve = CompositeCurve([curve1, curve2])
       curve.plot("1D", comparators=[curve1, curve2], labels=["Composite", "C1", "C2"])

    .. plot::

       from rateslib.curves import Curve, CompositeCurve
       import matplotlib.pyplot as plt
       from datetime import datetime as dt
       curve1 = Curve(
           nodes={
               dt(2022, 1, 1): 1.0,
               dt(2023, 1, 1): 0.98,
               dt(2024, 1, 1): 0.965,
               dt(2025, 1, 1): 0.955
           },
           t=[dt(2023, 1, 1), dt(2023, 1, 1), dt(2023, 1, 1), dt(2023, 1, 1),
              dt(2024, 1, 1),
              dt(2025, 1, 1), dt(2025, 1, 1), dt(2025, 1, 1), dt(2025, 1, 1)],
       )
       curve2 =Curve(
           nodes={
               dt(2022, 1, 1): 1.0,
               dt(2022, 6, 30): 1.0,
               dt(2022, 7, 1): 0.999992,
               dt(2022, 12, 31): 0.999992,
               dt(2023, 1, 1): 0.999984,
               dt(2023, 6, 30): 0.999984,
               dt(2023, 7, 1): 0.999976,
               dt(2023, 12, 31): 0.999976,
               dt(2024, 1, 1): 0.999968,
               dt(2024, 6, 30): 0.999968,
               dt(2024, 7, 1): 0.999960,
               dt(2025, 1, 1): 0.999960,
           },
       )
       curve = CompositeCurve([curve1, curve2])
       fig, ax, line = curve.plot("1D", comparators=[curve1, curve2], labels=["Composite", "C1", "C2"])
       plt.show()

    The :meth:`~rateslib.curves.CompositeCurve.rate` method of a :class:`CompositeCurve`
    composed of either :class:`Curve` or :class:`IndexCurve` s
    accepts an ``approximate`` argument. When *True* by default it used a geometric mean
    approximation to determine composite period rates.
    Below we demonstrate this is more than 1000x faster and within 1e-8 of the true
    value.

    .. ipython:: python

       curve.rate(dt(2022, 6, 1), "1y")
       %timeit curve.rate(dt(2022, 6, 1), "1y")

    .. ipython:: python

       curve.rate(dt(2022, 6, 1), "1y", approximate=False)
       %timeit curve.rate(dt(2022, 6, 1), "1y", approximate=False)

    """

    collateral = None

    def __init__(
        self,
        curves: Union[list, tuple],
        id: Union[str, NoInput] = NoInput(0),
    ) -> None:
        self.id = id or uuid4().hex[:5] + "_"  # 1 in a million clash

        self.curves = tuple(curves)
        self.node_dates = self.curves[0].node_dates
        self.calendar = curves[0].calendar
        self._base_type = curves[0]._base_type
        if self._base_type == "dfs":
            self.modifier = curves[0].modifier
            self.convention = curves[0].convention
        if type(curves[0]) is IndexCurve:
            self.index_lag = curves[0].index_lag
            self.index_base = curves[0].index_base

        # validate
        self._validate_curve_collection()

    def _validate_curve_collection(self):
        """Perform checks to ensure CompositeCurve can exist"""
        if type(self) is MultiCsaCurve and isinstance(self.curves[0], (LineCurve, IndexCurve)):
            raise TypeError("Multi-CSA curves must be of type `Curve`.")

        if type(self) is MultiCsaCurve and self.multi_csa_min_step > self.multi_csa_max_step:
            raise ValueError("`multi_csa_max_step` cannot be less than `min_step`.")

        types = [type(_) for _ in self.curves]
        if any([_ is CompositeCurve for _ in types]):
            raise TypeError(
                "Creating a CompositeCurve type containing sub CompositeCurve types is not "
                "yet implemented."
            )

        if not (
            all([_ is Curve or _ is ProxyCurve for _ in types])
            or all([_ is LineCurve for _ in types])
            or all([_ is IndexCurve for _ in types])
        ):
            raise TypeError(f"`curves` must be a list of similar type curves, got {types}.")

        ini_dates = [_.node_dates[0] for _ in self.curves]
        if not all([_ == ini_dates[0] for _ in ini_dates[1:]]):
            raise ValueError(f"`curves` must share the same initial node date, got {ini_dates}")

        if not type(self) is MultiCsaCurve:  # for multi_csa DF curve do not check calendars
            self._check_init_attribute("calendar")

        if self._base_type == "dfs":
            self._check_init_attribute("modifier")
            self._check_init_attribute("convention")

        if types[0] is IndexCurve:
            self._check_init_attribute("index_base")
            self._check_init_attribute("index_lag")

    def _check_init_attribute(self, attr):
        """Ensure attributes are the same across curve collection"""
        attrs = [getattr(_, attr, None) for _ in self.curves]
        if not all([_ == attrs[0] for _ in attrs[1:]]):
            raise ValueError(
                f"Cannot composite curves with different attributes, got for '{attr}': {attrs},"
            )

    def rate(
        self,
        effective: datetime,
        termination: Optional[Union[datetime, str]] = None,
        modifier: Optional[Union[str, bool]] = False,
        approximate: bool = True,
    ):
        """
        Calculate the composited rate on the curve.

        If rates are sought for dates prior to the initial node of the curve `None`
        will be returned.

        Parameters
        ----------
        effective : datetime
            The start date of the period for which to calculate the rate.
        termination : datetime or str
            The end date of the period for which to calculate the rate.
        modifier : str, optional
            The day rule if determining the termination from tenor. If `False` is
            determined from the `Curve` modifier.
        approximate : bool, optional
            When compositing :class:`Curve` or :class:`IndexCurve` calculating many
            individual rates is expensive. This uses an approximation typically with
            error less than 1/100th of basis point. Not used if ``multi_csa`` is True.

        Returns
        -------
        Dual, Dual2 or float
        """
        if self._base_type == "values":
            _ = 0.0
            for i in range(0, len(self.curves)):
                _ += self.curves[i].rate(effective, termination, modifier)
            return _
        elif self._base_type == "dfs":
            modifier = self.modifier if modifier is False else modifier
            if isinstance(termination, str):
                termination = add_tenor(effective, termination, modifier, self.calendar)

            d = _DCF1d[self.convention.upper()]

            if approximate:
                # calculates the geometric mean overnight rates in periods and adds
                _, n = 0.0, (termination - effective).days
                for curve_ in self.curves:
                    r = curve_.rate(effective, termination)
                    _ += ((1 + r * n * d / 100) ** (1 / n) - 1) / d

                _ = ((1 + d * _) ** n - 1) * 100 / (d * n)

            else:
                _, dcf_ = 1.0, 0.0
                date_ = effective
                while date_ < termination:
                    term_ = add_tenor(date_, "1B", None, self.calendar)
                    __, d_ = 0.0, (term_ - date_).days * d
                    dcf_ += d_
                    for curve in self.curves:
                        __ += curve.rate(date_, term_)
                    _ *= 1 + d_ * __ / 100
                    date_ = term_
                _ = 100 * (_ - 1) / dcf_
        else:
            raise TypeError(
                f"Base curve type is unrecognised: {self._base_type}"
            )  # pragma: no cover

        return _

    def __getitem__(self, date: datetime):
        if self._base_type == "dfs":
            # will return a composited discount factor
            if date == self.curves[0].node_dates[0]:
                return 1.0  # TODO (low:?) this is not variable but maybe should be tagged as "id0"?
            elif date < self.curves[0].node_dates[0]:
                return 0.0  # Any DF in the past is set to zero consistent with behaviour on `Curve`
            days = (date - self.curves[0].node_dates[0]).days
            d = _DCF1d[self.convention.upper()]

            total_rate = 0.0
            for curve in self.curves:
                avg_rate = ((1.0 / curve[date]) ** (1.0 / days) - 1) / d
                total_rate += avg_rate
            _ = 1.0 / (1 + total_rate * d) ** days
            return _

        elif self._base_type == "values":
            # will return a composited rate
            _ = 0.0
            for curve in self.curves:
                _ += curve[date]
            return _

        else:
            raise TypeError(
                f"Base curve type is unrecognised: {self._base_type}"
            )  # pragma: no cover

    def shift(
        self,
        spread: float,
        id: Optional[str] = None,
        composite: Optional[bool] = True,
        collateral: Optional[str] = None,
    ) -> CompositeCurve:
        """
        Create a new curve by vertically adjusting the curve by a set number of basis
        points.

        This curve adjustment preserves the shape of the curve but moves it up or
        down as a translation.
        This method is suitable as a way to assess value changes of instruments when
        a parallel move higher or lower in yields is predicted.

        Parameters
        ----------
        spread : float, Dual, Dual2
            The number of basis points added to the existing curve.
        id : str, optional
            Set the id of the returned curve.
        composite: bool, optional
            If True will return a CompositeCurve that adds a flat curve to the existing curve.
            This results in slower calculations but the curve will maintain a dynamic
            association with the underlying curve and will change if the underlying curve changes.
        collateral: str, optional
            Designate a collateral tag for the curve which is used by other methods.

        Returns
        -------
        CompositeCurve
        """
        if composite:
            # TODO (med) allow composite composite curves
            raise ValueError(
                "Creating a CompositeCurve containing sub CompositeCurves is not yet implemented.\n"
                "Set `composite` to False."
            )

        curves = (self.curves[0].shift(spread=spread, composite=composite),)
        curves += self.curves[1:]
        _ = CompositeCurve(curves=curves, id=id)
        _.collateral = collateral
        return _

    def translate(self, start: datetime, t: bool = False) -> CompositeCurve:
        """
        Create a new curve with an initial node date moved forward keeping all else
        constant.

        This curve adjustment preserves forward curve expectations as time evolves.
        This method is suitable as a way to create a subsequent *opening* curve from a
        previous day's *closing* curve.

        Parameters
        ----------
        start : datetime
            The new initial node date for the curve, must be in the domain:
            (node_date[0], node_date[1]]
        t : bool
            Set to *True* if the initial knots of the knot sequence should be
            translated forward.

        Returns
        -------
        CompositeCurve
        """
        return CompositeCurve(curves=[curve.translate(start, t) for curve in self.curves])

    def roll(self, tenor: Union[datetime, str]) -> CompositeCurve:
        """
        Create a new curve with its shape translated in time

        This curve adjustment is a simulation of a future state of the market where
        forward rates are assumed to have moved so that the present day's curve shape
        is reflected in the future (or the past). This is often used in trade
        strategy analysis.

        Parameters
        ----------
        tenor : datetime or str
            The date or tenor by which to roll the curve. If a tenor, as str, will
            derive the datetime as measured from the initial node date. If supplying a
            negative tenor, or a past datetime, there is a limit to how far back the
            curve can be rolled - it will first roll backwards and then attempt to
            :meth:`translate` forward to maintain the initial node date.

        Returns
        -------
        CompositeCurve
        """
        return CompositeCurve(curves=[curve.roll(tenor) for curve in self.curves])

    def index_value(self, date: datetime, interpolation: str = "daily"):
        """
        Calculate the accrued value of the index from the ``index_base``.

        See :meth:`IndexCurve.index_value()<rateslib.curves.IndexCurve.index_value>`
        """
        if not isinstance(self.curves[0], IndexCurve):
            raise TypeError("`index_value` not available on non `IndexCurve` types.")
        return super().index_value(date, interpolation)

    def _get_node_vector(self):
        return NotImplementedError("Instances of CompositeCurve do not have solvable variables.")


class MultiCsaCurve(CompositeCurve):
    """
    A dynamic composition of a sequence of other curves.

    .. note::
       Can only combine curves of the type: :class:`Curve`. Other curve parameters such as
       ``modifier``, and ``convention`` must also match.

    Parameters
    ----------
    curves : sequence of :class:`Curve`, :class:`LineCurve` or :class:`IndexCurve`
        The curves to be composited.
    id : str, optional, set by Default
        The unique identifier to distinguish between curves in a multi-curve framework.
    multi_csa_min_step: int, optional
        The minimum calculation step between subsequent DF evaluations to determine a multi-CSA
        curve term DF. Higher numbers make faster calculations but are less accurate. Should be
        in [1, max_step].
    multi_csa_max_step: int, optional
        The minimum calculation step between subsequent DF evaluations to determine a multi-CSA
        curve term DF. Higher numbers make faster calculations but are less accurate. Should be
        in [min_step, 1825].

    Notes
    -----
    A *MultiCsaCurve* uses a different calculation methodology than a *CompositeCurve* for
    determining the *rate* by selecting the curve within the collection with the highest rate.
    """

    def __init__(
        self,
        curves: Union[list, tuple],
        id: Union[str, NoInput] = NoInput(0),
        multi_csa_min_step: Optional[int] = 1,
        multi_csa_max_step: Optional[int] = 1825,
    ) -> None:
        self.multi_csa_min_step = max(1, multi_csa_min_step)
        self.multi_csa_max_step = min(1825, multi_csa_max_step)
        super().__init__(curves, id)

    def rate(
        self,
        effective: datetime,
        termination: Optional[Union[datetime, str]] = None,
        modifier: Optional[Union[str, bool]] = False,
    ):
        """
        Calculate the cheapest-to-deliver (CTD) rate on the curve.

        If rates are sought for dates prior to the initial node of the curve `None`
        will be returned.

        Parameters
        ----------
        effective : datetime
            The start date of the period for which to calculate the rate.
        termination : datetime or str
            The end date of the period for which to calculate the rate.
        modifier : str, optional
            The day rule if determining the termination from tenor. If `False` is
            determined from the `Curve` modifier.

        Returns
        -------
        Dual, Dual2 or float
        """
        modifier = self.modifier if modifier is False else modifier
        if isinstance(termination, str):
            termination = add_tenor(effective, termination, modifier, self.calendar)

        d = _DCF1d[self.convention.upper()]
        n = (termination - effective).days
        # TODO (low:perf) when these discount factors are looked up the curve repeats
        # the lookup could be vectorised to return two values at once.
        df_num = self[effective]
        df_den = self[termination]
        _ = (df_num / df_den - 1) * 100 / (d * n)
        return _

    def __getitem__(self, date: datetime):
        # will return a composited discount factor
        if date == self.curves[0].node_dates[0]:
            return 1.0  # TODO (low:?) this is not variable but maybe should be tagged as "id0"?
        # days = (date - self.curves[0].node_dates[0]).days
        # d = _DCF1d[self.convention.upper()]

        # method uses the step and picks the highest (cheapest rate)
        # in each period
        _ = 1.0
        d1 = self.curves[0].node_dates[0]

        def _get_step(step):
            return min(max(step, self.multi_csa_min_step), self.multi_csa_max_step)

        d2 = d1 + timedelta(days=_get_step(defaults.multi_csa_steps[0]))
        # cache stores looked up DF values to next loop, avoiding double calc
        cache, k = {i: 1.0 for i in range(len(self.curves))}, 1
        while d2 < date:
            min_ratio = 1e5
            for i, curve in enumerate(self.curves):
                d2_df = curve[d2]
                ratio_ = d2_df / cache[i]
                min_ratio = ratio_ if ratio_ < min_ratio else min_ratio
                cache[i] = d2_df
            _ *= min_ratio
            try:
                step = _get_step(defaults.multi_csa_steps[k])
            except IndexError:
                step = self.multi_csa_max_step
            d1, d2, k = d2, d2 + timedelta(days=step), k + 1

        # finish the loop on the correct date
        if date == d1:
            return _
        else:
            min_ratio = 1e5
            for i, curve in enumerate(self.curves):
                ratio_ = curve[date] / cache[i]  # cache[i] = curve[d1]
                min_ratio = ratio_ if ratio_ < min_ratio else min_ratio
            _ *= min_ratio
            return _

    def translate(self, start: datetime, t: bool = False) -> MultiCsaCurve:
        """
        Create a new curve with an initial node date moved forward keeping all else
        constant.

        This curve adjustment preserves forward curve expectations as time evolves.
        This method is suitable as a way to create a subsequent *opening* curve from a
        previous day's *closing* curve.

        Parameters
        ----------
        start : datetime
            The new initial node date for the curve, must be in the domain:
            (node_date[0], node_date[1]]
        t : bool
            Set to *True* if the initial knots of the knot sequence should be
            translated forward.

        Returns
        -------
        MultiCsaCurve
        """
        return MultiCsaCurve(
            curves=[curve.translate(start, t) for curve in self.curves],
            multi_csa_max_step=self.multi_csa_max_step,
            multi_csa_min_step=self.multi_csa_min_step,
        )

    def roll(self, tenor: Union[datetime, str]) -> MultiCsaCurve:
        """
        Create a new curve with its shape translated in time

        This curve adjustment is a simulation of a future state of the market where
        forward rates are assumed to have moved so that the present day's curve shape
        is reflected in the future (or the past). This is often used in trade
        strategy analysis.

        Parameters
        ----------
        tenor : datetime or str
            The date or tenor by which to roll the curve. If a tenor, as str, will
            derive the datetime as measured from the initial node date. If supplying a
            negative tenor, or a past datetime, there is a limit to how far back the
            curve can be rolled - it will first roll backwards and then attempt to
            :meth:`translate` forward to maintain the initial node date.

        Returns
        -------
        MultiCsaCurve
        """
        return MultiCsaCurve(
            curves=[curve.roll(tenor) for curve in self.curves],
            multi_csa_max_step=self.multi_csa_max_step,
            multi_csa_min_step=self.multi_csa_min_step,
        )

    def shift(
        self,
        spread: float,
        id: Optional[str] = None,
        composite: Optional[bool] = True,
        collateral: Optional[str] = None,
    ) -> MultiCsaCurve:
        """
        Create a new curve by vertically adjusting the curve by a set number of basis
        points.

        This curve adjustment preserves the shape of the curve but moves it up or
        down as a translation.
        This method is suitable as a way to assess value changes of instruments when
        a parallel move higher or lower in yields is predicted.

        Parameters
        ----------
        spread : float, Dual, Dual2
            The number of basis points added to the existing curve.
        id : str, optional
            Set the id of the returned curve.
        composite: bool, optional
            If True will return a CompositeCurve that adds a flat curve to the existing curve.
            This results in slower calculations but the curve will maintain a dynamic
            association with the underlying curve and will change if the underlying curve changes.
        collateral: str, optional
            Designate a collateral tag for the curve which is used by other methods.

        Returns
        -------
        CompositeCurve
        """
        if composite:
            # TODO (med) allow composite composite curves
            raise ValueError(
                "Creating a CompositeCurve containing sub CompositeCurves or MultiCsaCurves is "
                "not yet implemented.\nSet `composite` to False."
            )

        curves = tuple(_.shift(spread=spread, composite=composite) for _ in self.curves)
        _ = MultiCsaCurve(
            curves=curves,
            id=id,
            multi_csa_max_step=self.multi_csa_max_step,
            multi_csa_min_step=self.multi_csa_min_step,
        )
        _.collateral = collateral
        return _


class ProxyCurve(Curve):
    """
    A subclass of :class:`~rateslib.curves.Curve` which returns dynamic DFs based on
    other curves related via :class:`~rateslib.fx.FXForwards` parity.

    Parameters
    ----------
    cashflow : str
        The currency in which cashflows are represented (3-digit code).
    collateral : str
        The currency of the CSA against which cashflows are collateralised (3-digit
        code).
    fx_forwards : FXForwards
        The :class:`~rateslib.fx.FXForwards` object which contains the relating
        FX information and the available :class:`~rateslib.curves.Curve` s.
    convention : str
        The day count convention used for calculating rates. If `None` defaults
        to the convention in the local cashflow currency.
    modifier : str, optional
        The modification rule, in {"F", "MF", "P", "MP"}, for determining rates.
        If `False` will default to the modifier in the local cashflow currency.
    calendar : calendar or str, optional
        The holiday calendar object to use. If str, lookups named calendar
        from static data. Used for determining rates. If `False` will
        default to the calendar in the local cashflow currency.
    id : str, optional, set by Default
        The unique identifier to distinguish between curves in a multi-curve framework.

    Notes
    -----
    The DFs returned are calculated via the chaining method and the below formula,
    relating the DF curve in the local collateral currency and FX forward rates.

    .. math::

       w_{dom:for,i} = \\frac{f_{DOMFOR,i}}{F_{DOMFOR,0}} v_{for:for,i}

    The returned curve contains contrived methods to calculate this dynamically and
    efficiently from the combination of curves and FX rates that are available within
    the given :class:`FXForwards` instance.
    """

    _base_type = "dfs"

    def __init__(
        self,
        cashflow: str,
        collateral: str,
        fx_forwards: FXForwards,
        convention: Optional[str] = None,
        modifier: Optional[Union[str, bool]] = False,
        calendar: Optional[Union[CalInput, bool]] = False,
        id: Optional[str] = None,
    ):
        self.id = id or uuid4().hex[:5] + "_"  # 1 in a million clash
        cash_ccy, coll_ccy = cashflow.lower(), collateral.lower()
        self.collateral = coll_ccy
        self._is_proxy = True
        self.fx_forwards = fx_forwards
        self.cash_currency = cash_ccy
        self.cash_pair = f"{cash_ccy}{cash_ccy}"
        self.cash_idx = self.fx_forwards.currencies[cash_ccy]
        self.coll_currency = coll_ccy
        self.coll_pair = f"{coll_ccy}{coll_ccy}"
        self.coll_idx = self.fx_forwards.currencies[coll_ccy]
        self.pair = f"{cash_ccy}{coll_ccy}"
        self.path = self.fx_forwards._get_recursive_chain(
            self.fx_forwards.transform, self.coll_idx, self.cash_idx
        )[1]
        self.terminal = list(self.fx_forwards.fx_curves[self.cash_pair].nodes.keys())[-1]

        default_curve = Curve(
            {},
            convention=self.fx_forwards.fx_curves[self.cash_pair].convention
            if convention is None
            else convention,
            modifier=self.fx_forwards.fx_curves[self.cash_pair].modifier
            if modifier is False
            else modifier,
            calendar=self.fx_forwards.fx_curves[self.cash_pair].calendar
            if calendar is False
            else calendar,
        )
        self.convention = default_curve.convention
        self.modifier = default_curve.modifier
        self.calendar = default_curve.calendar
        self.node_dates = [self.fx_forwards.immediate, self.terminal]

    def __getitem__(self, date: datetime):
        return (
            self.fx_forwards.rate(self.pair, date, path=self.path)
            / self.fx_forwards.fx_rates_immediate.fx_array[self.cash_idx, self.coll_idx]
            * self.fx_forwards.fx_curves[self.coll_pair][date]
        )

    def to_json(self):  # pragma: no cover
        """
        Not implemented for :class:`~rateslib.fx.ProxyCurve` s.
        :return:
        """
        return NotImplementedError("`to_json` not available on proxy curve.")

    def from_json(self):  # pragma: no cover
        """
        Not implemented for :class:`~rateslib.fx.ProxyCurve` s.
        """
        return NotImplementedError("`from_json` not available on proxy curve.")

    def _set_ad_order(self):  # pragma: no cover
        """
        Not implemented for :class:`~rateslib.fx.ProxyCurve` s.
        """
        return NotImplementedError("`set_ad_order` not available on proxy curve.")

    def _get_node_vector(self):
        return NotImplementedError("Instances of ProxyCurve do not have solvable variables.")


def average_rate(effective, termination, convention, rate):
    """
    Return the geometric, 1 calendar day, average rate for the rate in a period.

    This is used for approximations usually in combination with floating periods.

    Parameters
    ----------
    effective : datetime
        The effective date of the rate.
    termination : datetime
        The termination date of the rate.
    convention : str
        The day count convention of the curve rate.
    rate : float, Dual, Dual2
        The rate to decompose to average, in percentage terms, e.g. 0.04 = 4% rate.

    Returns
    -------
    tuple : The rate, the 1-day DCF, and the number of calendar days
    """
    d = _DCF1d[convention.upper()]
    n = (termination - effective).days
    _ = ((1 + n * d * rate / 100) ** (1 / n) - 1) / d
    return _ * 100, d, n


def interpolate(x, x_1, y_1, x_2, y_2, interpolation, start=None):
    """
    Perform local interpolation between two data points.

    Parameters
    ----------
    x : Any with topology
        The x-value for which the interpolated y-value is sought.
    x_1 : same type as ``x``
        The left bound for the local interpolation.
    y_1 : float, Dual, Dual2
        The value at the left bound.
    x_2 : same type as ``x``
        The right bound for the local interpolation.
    y_2 : float, Dual, Dual2
        The value at the right bound.
    interpolation : str
        The interpolation rule to use in *{"linear", "log_linear", "linear_zero_rate",
        "flat_forward", "flat_backward"}*.
    start : datetime
        Used only if ``interpolation`` is *"linear_zero_rate"* to identify the start
        date of a curve.

    Returns
    -------
    float, Dual, Dual2

    Notes
    -----
    If ``x`` is outside the region ``[x_1, x_2]`` this function will extrapolate
    instead of interpolate using the same mathematical formula.

    Examples
    --------
    .. ipython:: python

       interpolate(50, 0, 10, 100, 50, "linear")
       interpolate(dt(2000, 1, 6), dt(2000, 1, 1), 10, dt(2000, 1, 11), 50, "linear")
    """
    if interpolation == "linear":

        def op(z):
            return z

    elif interpolation == "linear_index":

        def op(z):
            return 1 / z

        y_1, y_2 = 1 / y_1, 1 / y_2
    elif interpolation == "log_linear":
        op, y_1, y_2 = dual_exp, dual_log(y_1), dual_log(y_2)
    elif interpolation == "linear_zero_rate":
        # convention not used here since we just determine linear rate interpolation
        # 86400. scalar relates to using posix timestamp conversion
        y_2 = dual_log(y_2) / ((start - x_2) / (365.0 * 86400.0))
        if start == x_1:
            y_1 = y_2
        else:
            y_1 = dual_log(y_1) / ((start - x_1) / (365.0 * 86400.0))

        def op(z):
            return dual_exp((start - x) / (365.0 * 86400.0) * z)

    elif interpolation == "flat_forward":
        if x >= x_2:
            return y_2
        return y_1
    elif interpolation == "flat_backward":
        if x <= x_1:
            return y_1
        return y_2
    else:
        raise ValueError(
            '`interpolation` must be in {"linear", "log_linear", "linear_index", '
            '"linear_zero_rate", "flat_forward", "flat_backward"}, got: '
            f"{interpolation}."
        )
    ret = op(y_1 + (y_2 - y_1) * ((x - x_1) / (x_2 - x_1)))
    return ret


def index_left(
    list_input: list[Any],
    list_length: int,
    value: Any,
    left_count: Optional[int] = 0,
):
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

    split = floor((list_length - 1) / 2)
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


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.
