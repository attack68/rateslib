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
from typing import TYPE_CHECKING

from rateslib.enums.generics import NoInput

if TYPE_CHECKING:
    from rateslib.typing import DualTypes, DualTypes_, NoInput  # pragma: no cover


class _AmortizationType(Enum):
    """
    Enumerable type to define the possible types of amortization that some legs can handle.
    """

    NoAmortization = 0
    ConstantPeriod = 1
    CustomSchedule = 2


class Amortization:
    """
    An amortization schedule for any :class:`~rateslib.legs._BaseLeg`.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.legs import Amortization

    .. ipython:: python

       obj = Amortization(n=5, initial=1e6, amortization="to_zero")
       obj.outstanding
       obj.amortization

    Parameters
    ----------
    n: int
        The number of periods in the schedule.
    initial: float, Dual, Dual2, Variable
        The notional applied to the first period in the schedule.
    amortization: float, Dual, Dual2, Variable, list, tuple, str, optional
        The amortization structure to apply to the schedule.

    Notes
    -----
    If ``amortization`` is:

    - not specified then the schedule is assumed to have no amortization.
    - some scalar then the amortization amount will be a constant value per period.
    - a list or tuple of *n-1* scalars, then this is defines a custome amortization schedule.
    - a string flag then an amortization schedule will be calculated directly:

      - *"to_zero"*: each period will be a constant value ending with zero implied ending balance.
      - *"{float}%"*: each period will amortize by a constant percentage of the outstanding balance.


    .. rubric:: Using Amortization with Instruments

    This section exemplifies how to use :class:`~rateslib.legs.Amortization` with instruments.

      **Key Points**

      - Amortization can be added to *Instruments* using the per leg ``amortization`` argument.
      - It supports constant notional amortization, or custom schedules or
        the :class:`~rateslib.legs.Amortization` class can be used to calculate other simple
        structures.
      - Some *Instruments* have not yet integrated amortization into their calculation, such as
        *Bonds*.

    **Standard Amortization**

    The :class:`~rateslib.legs.FixedLeg` and :class:`~rateslib.legs.FloatLeg` classes both
    have ``amortization`` as an input argument. An :class:`~rateslib.legs.Amortization` class
    can be directly supplied or other values are internally passed to this class for
    syntactic convenience.

    The simplest, and most common, type of ``amortization`` to apply is a constant notional
    per period.

    .. ipython:: python
       :suppress:

       from rateslib import XCS, IRS, IndexFixedRateBond, FixedRateBond
       from rateslib.legs import FixedLeg, FloatLeg, Amortization
       from rateslib.scheduling import Schedule
       from datetime import datetime as dt

    .. tabs::

       .. tab:: FixedLeg

          .. ipython:: python

             fxl = FixedLeg(
                 schedule=Schedule(dt(2000, 1, 1), "1y", "Q"),
                 notional=10e6,
                 amortization=1e6,      # <- 1mm reduction per period
             )
             fxl.cashflows()[["Type", "Acc Start", "Notional"]]

       .. tab:: FloatLeg

          .. ipython:: python

             fll = FloatLeg(
                 schedule=Schedule(dt(2000, 1, 1), "1y", "M"),
                 notional=10e6,
                 amortization=0.5e6,    # 0.5mm reduction per period
             )
             fll.cashflows()[["Type", "Acc Start", "Notional"]]

    Here, the *amortization* is expressed in a specific notional amount reduction per period so,
    when applied to an :class:`~rateslib.instruments.IRS`, each leg with different
    frequencies should be input directly.

    If a *Leg* has a *final notional exchange* then any amortized amount would, under standard
    convention, be paid out at the same time as the notional change. The final cashflow will be
    reduced by the amount of interim exchanges that have already occurred. This can be
    exemplified on a :class:`~rateslib.instruments.XCS`.

    .. tabs::

       .. tab:: IRS

          .. ipython:: python

             irs = IRS(
                 effective=dt(2000, 1, 1),
                 termination="1Y",
                 frequency="Q",
                 leg2_frequency="S",
                 notional=1e6,
                 amortization=2e5,       # <- Reduces notional on 1st July to 600,000
                 leg2_amortization=-4e5, # <- Aligns the notional on 1st July
             )
             irs.cashflows()[["Type", "Acc Start", "Notional"]]

       .. tab:: Non-MTM XCS

          .. ipython:: python

             xcs = XCS(
                 effective=dt(2000, 1, 1),
                 termination="1y",
                 spec="eurusd_xcs",
                 notional=5e6,
                 amortization=1e6,      # <- 1mm reduction and notional exchange per period
                 leg2_mtm=False,
             )
             xcs.cashflows()[["Type", "Period", "Acc Start", "Payment", "Ccy", "Notional", "Reference Ccy"]]

       .. tab:: MTM XCS

          Mark-to-market :class:`~rateslib.instruments.XCS` also support ``amortization`` which
          affects the MTM cashflows respectively.

          .. ipython:: python

             xcs = XCS(
                 effective=dt(2000, 1, 1),
                 termination="1y",
                 spec="eurusd_xcs",
                 notional=5e6,
                 amortization=1e6,      # <- 1mm reduction and notional exchange per period
                 leg2_mtm=True,
             )
             xcs.cashflows()[["Type", "Period", "Acc Start", "Payment", "Ccy", "Notional", "Reference Ccy"]]

    .. rubric:: Custom Amortization

    By using the :class:`~rateslib.legs.Amortization` class custom amortization can be directly
    input to an *Instrument*. The following examples are the same, with the first being
    syntactic convenience for the second. The above examples are also syntactic convenience for
    applying the same amortization amount each period.

    .. tabs::

       .. tab:: Amortization List

          .. ipython:: python

             irs = IRS(
                 effective=dt(2000, 1, 1),
                 termination="1Y",
                 frequency="Q",
                 leg2_frequency="S",
                 notional=1e6,
                 amortization=[100000, 300000, -5000],    # <- Reduces notional on 1st July to 600,000
                 leg2_amortization=[-400000], # <- Aligns the notional on 1st July
             )
             irs.cashflows()[["Type", "Acc Start", "Notional"]]

       .. tab:: Amortization Object

          .. ipython:: python

             irs = IRS(
                 effective=dt(2000, 1, 1),
                 termination="1Y",
                 frequency="Q",
                 leg2_frequency="S",
                 notional=1e6,
                 amortization=Amortization(4, 1e6, [100000, 300000, -5000]),
                 leg2_amortization=Amortization(2, -1e6, [-400000])
             )
             irs.cashflows()[["Type", "Acc Start", "Notional"]]

    .. rubric:: Unsupported Instruments

    *Instruments* that currently do **not** support amortization are *Bonds*.

    .. tabs::

       .. tab:: FixedRateBond

          .. ipython:: python

             try:
                 FixedRateBond(
                     effective=dt(2000, 1, 1),
                     termination="1y",
                     spec="us_gb",
                     notional=5e6,
                     amortization=1e6,
                     fixed_rate=2.0,
                 )
             except Exception as e:
                 print(e)

       .. tab:: IndexFixedRateBond

          .. ipython:: python

             try:
                 IndexFixedRateBond(
                     effective=dt(2000, 1, 1),
                     termination="1y",
                     spec="us_gb",
                     notional=5e6,
                     amortization=1e6,
                     fixed_rate=2.0,
                     index_base=100.0,
                 )
             except Exception as e:
                 print(e)

    """  # noqa: E501

    _type: _AmortizationType

    @property
    def amortization(self) -> tuple[DualTypes, ...]:
        """A tuple of (n-1) amortization amounts for each *Period*."""
        return self._amortization

    @property
    def outstanding(self) -> tuple[DualTypes, ...]:
        """A tuple of n outstanding notional amounts for each *Period*."""
        return self._outstanding

    def __init__(
        self,
        n: int,
        initial: DualTypes,
        amortization: DualTypes_ | list[DualTypes] | tuple[DualTypes, ...] | str = NoInput(0),
    ) -> None:
        if isinstance(amortization, NoInput):
            self._type = _AmortizationType.NoAmortization
            self._amortization: tuple[DualTypes, ...] = (0.0,) * (n - 1)
            self._outstanding: tuple[DualTypes, ...] = (initial,) * n
        elif isinstance(amortization, list | tuple):
            self._type = _AmortizationType.CustomSchedule
            if len(amortization) != (n - 1):
                raise ValueError(
                    "Custom amortisation schedules must have `n-1` amortization amounts for `n` "
                    f"periods.\nGot '{len(amortization)}' amounts for '{n}' periods."
                )
            self._amortization = tuple(amortization)
            outstanding = [initial]
            for value in amortization:
                outstanding.append(outstanding[-1] - value)
            self._outstanding = tuple(outstanding)
        elif isinstance(amortization, str):
            if amortization.lower() == "to_zero":
                self._type = _AmortizationType.ConstantPeriod
                self._amortization = (initial / n,) * (n - 1)
                self._outstanding = (initial,) + tuple([initial * (1 - i / n) for i in range(1, n)])
            elif amortization[-1] == "%":
                self._type = _AmortizationType.CustomSchedule
                amortization_ = [initial * float(amortization[:-1]) / 100]
                outstanding_ = [initial]
                for i in range(1, n):
                    outstanding_.append(outstanding_[-1] - amortization_[-1])
                    if i != n - 1:
                        amortization_.append(outstanding_[-1] * float(amortization[:-1]) / 100)
                self._outstanding = tuple(outstanding_)
                self._amortization = tuple(amortization_)
            else:
                raise ValueError("`amortization` as string must be one of 'to_zero', '{float}%'.")
        else:  # isinstance(amortization, DualTypes)
            self._type = _AmortizationType.ConstantPeriod
            self._amortization = (amortization,) * (n - 1)
            self._outstanding = (initial,) + tuple(
                [initial - amortization * i for i in range(1, n)]
            )

    def __mul__(self, other: DualTypes) -> Amortization:
        return Amortization(
            n=len(self.outstanding),
            initial=self.outstanding[0] * other,
            amortization=[_ * other for _ in self.amortization],
        )

    def __rmul__(self, other: DualTypes) -> Amortization:
        return self.__mul__(other)


def _get_amortization(
    amortization: DualTypes_ | list[DualTypes] | tuple[DualTypes, ...] | str | Amortization,
    initial: DualTypes,
    n: int,
) -> Amortization:
    if isinstance(amortization, Amortization):
        return amortization
    else:
        return Amortization(n, initial, amortization)
