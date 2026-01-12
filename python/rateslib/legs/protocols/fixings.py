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

from typing import TYPE_CHECKING, Protocol

from pandas import DataFrame, Series

from rateslib.enums.generics import NoInput
from rateslib.legs.protocols.npv import _WithNPV
from rateslib.periods.protocols.fixings import (
    _replace_fixings_with_ad_variables,
    _reset_fixings_data,
    _structure_sensitivity_data,
)

if TYPE_CHECKING:
    from rateslib.typing import (
        CurveOption_,
        DualTypes,
        FXForwards_,
        Sequence,
        _BaseCurve_,
        _BasePeriod,
        _FXVolOption_,
        datetime_,
        int_,
    )


class _WithFixings(_WithNPV, Protocol):
    """
    Protocol for determining fixing sensitivity for a *Period* with AD.

    .. rubric:: Provided methods

    .. autosummary::

       ~_WithFixings.reset_fixings

    """

    @property
    def periods(self) -> Sequence[_BasePeriod]: ...

    def reset_fixings(self, state: int_ = NoInput(0)) -> None:
        """
        Resets any fixings values of the *Leg* derived using the given data state.

        .. role:: green

        Parameters
        ----------
        state: int, :green:`optional`
            The *state id* of the data series that set the fixing. Only fixings determined by this
            data will be reset. If not given resets all fixings.

        Returns
        -------
        None
        """
        for period in self.periods:
            period.reset_fixings(state)

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
