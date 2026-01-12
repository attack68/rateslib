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

import os
from itertools import product
from typing import TYPE_CHECKING, Protocol

from pandas import DataFrame, DatetimeIndex, MultiIndex, Series, isna

from rateslib import fixings
from rateslib.dual import Variable, gradient
from rateslib.dual.utils import _dual_float
from rateslib.enums.generics import NoInput
from rateslib.periods.parameters import (
    _FloatRateParams,
    _FXOptionParams,
    _IndexParams,
    _MtmParams,
    _NonDeliverableParams,
)
from rateslib.periods.protocols.npv import _WithNPV

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        CurveOption_,
        DualTypes,
        FXForwards_,
        Sequence,
        _BaseCurve_,
        _FXVolOption_,
        datetime_,
        int_,
    )


class _WithFixings(_WithNPV, Protocol):
    """
    Protocol for determining fixing sensitivity for a *Period* with AD.

    .. rubric:: Required methods

    .. autosummary::

       ~_WithFixings.reset_fixings

    .. rubric:: Provided methods

    .. autosummary::

       ~_WithFixings.reset_fixings

    """

    # def local_npv(
    #     self,
    #     *,
    #     rate_curve: CurveOption_ = NoInput(0),
    #     index_curve: _BaseCurve_ = NoInput(0),
    #     disc_curve: _BaseCurve_ = NoInput(0),
    #     fx: FXForwards_ = NoInput(0),
    #     fx_vol: _FXVolOption_ = NoInput(0),
    #     settlement: datetime_ = NoInput(0),
    #     forward: datetime_ = NoInput(0),
    # ) -> DualTypes: ...

    # @property
    # def settlement_param(self) -> _SettlementParams: ...

    def reset_fixings(self, state: int_ = NoInput(0)) -> None:
        """
        Resets any fixings values of the *Period* derived using the given data state.

        .. rubric:: Examples

        .. ipython:: python
           :suppress:

           from rateslib import fixings, dt, NoInput, FloatPeriod
           from pandas import Series

        .. ipython:: python

           fp = FloatPeriod(
               start=dt(2026, 1, 12),
               end=dt(2026, 1, 16),
               payment=dt(2026, 1, 16),
               frequency="M",
               fixing_method="rfr_payment_delay",
               method_param=0,
               rate_fixings="sofr"
           )
           fixings.add(
               name="sofr_1B",
               series=Series(
                   index=[dt(2026, 1, 12), dt(2026, 1, 13), dt(2026, 1, 14), dt(2026, 1, 15)],
                   data=[3.1, 3.2, 3.3, 3.4]
               )
           )
           # value is populated from given data
           assert 3.245 < fp.rate_params.rate_fixing.value < 3.255
           fp.reset_fixings()
           # private data related to fixing is removed and requires new data lookup
           fp.rate_params.rate_fixing._value
           fp.rate_params.rate_fixing._populated

        .. role:: green

        Parameters
        ----------
        state: int, :green:`optional`
            The *state id* of the data series that set the fixing. Only fixings determined by this
            data will be reset. If not given resets all fixings.
        """
        if isinstance(getattr(self, "index_params", None), _IndexParams):
            self.index_params.index_base.reset(state)  # type: ignore[attr-defined]
            self.index_params.index_fixing.reset(state)  # type: ignore[attr-defined]
        if isinstance(getattr(self, "rate_params", None), _FloatRateParams):
            self.rate_params.rate_fixing.reset(state)  # type: ignore[attr-defined]
        if isinstance(getattr(self, "mtm_params", None), _MtmParams):
            self.mtm_params.fx_fixing_start.reset(state)  # type: ignore[attr-defined]
            self.mtm_params.fx_fixing_end.reset(state)  # type: ignore[attr-defined]
        if isinstance(getattr(self, "non_deliverable_params", None), _NonDeliverableParams):
            self.non_deliverable_params.fx_fixing.reset(state)  # type: ignore[attr-defined]
        from rateslib.periods.float_period import ZeroFloatPeriod

        if isinstance(self, ZeroFloatPeriod):
            for float_period in self.float_periods:
                float_period.reset_fixings(state)
        if isinstance(getattr(self, "fx_option_params", None), _FXOptionParams):
            self.fx_option_params.option_fixing.reset(state)  # type: ignore[attr-defined]

    def local_fixings(
        self,
        identifiers: Sequence[tuple[str, Series]],
        scalars: Sequence[float] | NoInput = NoInput(0),
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DataFrame:
        """
        Calculate the sensitivity to fixings of the *Instrument*, expressed in local
        settlement currency.

        .. role:: red

        .. role:: green

        Parameters
        ----------
        indentifiers: Sequence of tuple[str, Series], :red:`required`
            These are the series string identifiers and the data values that will be used in each
            Series to determine the sensitivity against.
        scalars: Sequence of floats, :green:`optional (each set as 1.0)`
            A sequence of scalars to multiply the sensitivities by for each on of the
            ``identifiers``.
        rate_curve: _BaseCurve or dict of such indexed by string tenor, optional
            Used to forecast floating period rates, if necessary.
        index_curve: _BaseCurve, optional
            Used to forecast index values for indexation, if necessary.
        disc_curve: _BaseCurve, optional
            Used to discount cashflows.
        fx: FXForwards, optional
            The :class:`~rateslib.fx.FXForwards` object used for forecasting the
            ``fx_fixing`` for deliverable cashflows, if necessary. Or, an
            class:`~rateslib.fx.FXRates` object purely for immediate currency conversion.
        fx_vol: FXDeltaVolSmile, FXSabrSmile, FXDeltaVolSurface, FXSabrSurface, optional
            The FX volatility *Smile* or *Surface* object used for determining Black calendar
            day implied volatility values.
        settlement: datetime, optional (set as immediate date)
            The assumed settlement date of the *PV* determination. Used only to evaluate
            *ex-dividend* status.
        forward: datetime, optional (set as ``settlement``)
            The future date to project the *PV* to using the ``disc_curve``.

        Returns
        -------
        DataFrame
        """
        original_data, index, state = _replace_fixings_with_ad_variables(identifiers)
        # Extract sensitivity data
        pv: dict[str, DualTypes] = {
            self.settlement_params.currency: self.local_npv(
                rate_curve=rate_curve,
                index_curve=index_curve,
                disc_curve=disc_curve,
                fx=fx,
                fx_vol=fx_vol,
                settlement=settlement,
                forward=forward,
            )
        }
        df = _structure_sensitivity_data(pv, index, identifiers, scalars)
        _reset_fixings_data(self, original_data, state, identifiers)
        return df


def _replace_fixings_with_ad_variables(
    identifiers: Sequence[tuple[str, Series]],
) -> tuple[dict[str, tuple[int, Series]], DatetimeIndex, int]:
    """
    For a set of identifiers (which must already exist in the `fixings` object) extend those
    with the given data as new fixings expressed as a Variable which will capture sensitivity.

    Parameters
    ----------
    identifiers

    Returns
    -------
    tuple: the original data that will be reset later, the DatetimeIndex of relevant dates
    and the state id used for the added series
    """

    # for each identifier, replace the existing fixing Series with a new one with AD Variables.
    state = hash(os.urandom(64))
    original_data: dict[str, tuple[int, Series]] = {}
    index = DatetimeIndex(data=[])
    for identifier in identifiers:
        original_data[identifier[0]] = (fixings[identifier[0]][0], fixings[identifier[0]][1])
        ad_series = Series(
            index=identifier[1].index,
            data=[  # type: ignore[arg-type]
                Variable(_dual_float(v), [f"{identifier[0]}_{d.strftime('%Y%m%d')}"])  # type: ignore[attr-defined]
                for d, v in identifier[1].items()
            ],
        )
        index = index.union(other=ad_series.index, sort=None)  # type: ignore[arg-type]  # will sort
        fixings.pop(name=identifier[0])
        fixings.add(
            name=identifier[0],
            series=ad_series.combine(original_data[identifier[0]][1], _s2_before_s1),
            state=state,
        )

    return original_data, index, state


def _structure_sensitivity_data(
    pv: dict[str, DualTypes],
    index: DatetimeIndex,
    identifiers: Sequence[tuple[str, Series]],
    scalars: Sequence[float] | NoInput,
) -> DataFrame:
    if isinstance(scalars, NoInput):
        scalars_: Sequence[float] = [1.0] * len(identifiers)
    elif len(scalars) != len(identifiers):
        raise ValueError("If given, ``scalars`` must be same length as ``identifiers``.")
    else:
        scalars_ = scalars

    date_str = [_.strftime("%Y%m%d") for _ in index]

    # Construct DataFrame
    df = DataFrame(
        columns=MultiIndex.from_tuples(
            product(pv.keys(), [i[0] for i in identifiers]), names=["local_ccy", "identifier"]
        ),
        # index=date_list,
        index=index,
        data=[],
        dtype=float,
    )
    for ccy, v in pv.items():
        for j, identifier in enumerate(identifiers):
            df[(ccy, identifier[0])] = (
                gradient(v, vars=[identifier[0] + "_" + date for date in date_str]) * scalars_[j]
            )

    return df


class _SupportsResetFixings(Protocol):
    def reset_fixings(self, state: int_ = NoInput(0)) -> None: ...


def _reset_fixings_data(
    obj: _SupportsResetFixings,
    original_data: dict[str, tuple[int, Series]],
    state: int,
    identifiers: Sequence[tuple[str, Series]],
) -> None:
    # reset all data to original values.
    obj.reset_fixings(state=state)
    for identifier in identifiers:
        fixings.pop(name=identifier[0])
        fixings.add(
            name=identifier[0],
            series=original_data[identifier[0]][1],
            state=original_data[identifier[0]][0],
        )


def _s2_before_s1(v1: DualTypes, v2: DualTypes | None) -> DualTypes:
    if v2 is None or isna(v2):  # type: ignore[arg-type]
        return v1
    else:
        return v2
