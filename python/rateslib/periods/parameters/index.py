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

import warnings
from typing import TYPE_CHECKING

from pandas import Series

import rateslib.errors as err
from rateslib import defaults
from rateslib.curves.curves import _try_index_value
from rateslib.data.fixings import IndexFixing
from rateslib.enums.generics import (
    Err,
    NoInput,
    Ok,
    _drb,
)
from rateslib.enums.parameters import (
    IndexMethod,
    _get_index_method,
)

if TYPE_CHECKING:
    from rateslib.typing import (
        Any,
        DualTypes,
        DualTypes_,
        Result,
        _BaseCurve_,
        bool_,
        datetime_,
        int_,
        str_,
    )


class _IndexParams:
    """
    Parameters for *Period* cashflows adjusted under some indexation.

    Parameters
    ----------
    _index_method : IndexMethod
        The interpolation method, or otherwise, to determine index values from reference dates.
    _index_lag: int
        The indexation lag, in months, applied to the determination of index values.
    _index_base: float, Dual, Dual2, Variable, optional
        The specific value set of the base index value.
        If not given and ``index_fixings`` is a str fixings identifier that will be
        used to determine the base index value.
    _index_fixings: float, Dual, Dual2, Variable, Series, str, optional
        The index value for the reference date.
        If a scalar value this is used directly. If a string identifier will link to the
        central ``fixings`` object and data loader.
    _index_base_date: datetime, optional
        The reference date for determining the base index value. Not required if ``_index_base``
        value is given directly.
    _index_reference_date: datetime, optional
        The reference date for determining the index value. Not required if ``_index_fixings``
        is given as a scalar value.
    _index_only: bool, optional
        A flag which determines non-payment of notional on supported *Periods*.

    """

    _index_lag: int
    _index_method: IndexMethod
    _index_fixing: IndexFixing
    _index_base: IndexFixing
    _index_only: bool

    def __init__(
        self,
        *,
        _index_method: IndexMethod,
        _index_lag: int,
        _index_base: DualTypes_,
        _index_fixings: DualTypes | Series[DualTypes] | str_,  # type: ignore[type-var]
        _index_base_date: datetime_,
        _index_reference_date: datetime_,
        _index_only: bool,
    ) -> None:
        self._index_method = _index_method
        self._index_lag = _index_lag
        self._index_only = _index_only

        if isinstance(_index_fixings, Series):
            warnings.warn(err.FW_FIXINGS_AS_SERIES, FutureWarning)

        if isinstance(_index_base, NoInput) and isinstance(_index_fixings, Series):
            _index_base_value = IndexFixing._lookup(
                index_lag=self.index_lag,
                index_method=self.index_method,
                timeseries=_index_fixings,
                date=_index_base_date,  # type: ignore[arg-type]  # argument combinations
            )
            self._index_base = IndexFixing(
                date=_index_base_date,  # type: ignore[arg-type]  # argument combinations
                index_lag=self.index_lag,
                index_method=self.index_method,
                value=_index_base_value,
                identifier=NoInput(0),
            )
        else:
            self._index_base = IndexFixing(
                date=_index_base_date,  # type: ignore[arg-type]  # argument combinations
                index_lag=self.index_lag,
                index_method=self.index_method,
                value=_index_base,
                identifier=_index_fixings if isinstance(_index_fixings, str) else NoInput(0),
            )

        if isinstance(_index_fixings, Series):
            _index_ref_value = IndexFixing._lookup(
                index_lag=self.index_lag,
                index_method=self.index_method,
                timeseries=_index_fixings,
                date=_index_reference_date,  # type: ignore[arg-type]  # argument combinations
            )
            self._index_fixing = IndexFixing(
                date=_index_reference_date,  # type: ignore[arg-type]  # argument combinations
                index_lag=self.index_lag,
                index_method=self.index_method,
                value=_index_ref_value,
                identifier=NoInput(0),
            )
        else:
            self._index_fixing = IndexFixing(
                date=_index_reference_date,  # type: ignore[arg-type]  # argument combinations
                index_lag=self.index_lag,
                index_method=self.index_method,
                value=_index_fixings if not isinstance(_index_fixings, str) else NoInput(0),
                identifier=_index_fixings if isinstance(_index_fixings, str) else NoInput(0),
            )

    @property
    def index_base(self) -> IndexFixing:
        """The :class:`~rateslib.data.fixings.IndexFixing` associated with the index base date."""
        return self._index_base

    @index_base.setter
    def index_base(self, value: Any) -> None:
        raise ValueError(err.VE_ATTRIBUTE_IS_IMMUTABLE.format("index_base"))

    @property
    def index_fixing(self) -> IndexFixing:
        """The :class:`~rateslib.data.fixings.IndexFixing` associated with the index
        reference date."""
        return self._index_fixing

    @index_fixing.setter
    def index_fixing(self, value: Any) -> None:
        raise ValueError(err.VE_ATTRIBUTE_IS_IMMUTABLE.format("index_fixing"))

    @property
    def index_only(self) -> bool:
        """A flag which determines non-payment of notional on supported *Periods*."""
        return self._index_only

    @property
    def index_lag(self) -> int:
        """The indexation lag, in months, applied to the determination of index values."""
        return self._index_lag

    @property
    def index_method(self) -> IndexMethod:
        """The :class:`~rateslib.enums.parameters.IndexMethod` to determine index values
        from reference dates."""
        return self._index_method

    def try_index_value(
        self,
        index_curve: _BaseCurve_ = NoInput(0),
    ) -> Result[DualTypes]:
        """
        Determine the index reference value from fixing or forecast curve, with lazy error raising.

        Parameters
        ----------
        index_curve : _BaseCurve, optional
            The curve from which index values are forecast if required.

        Returns
        -------
        Result[float, Dual, Dual2, Variable]
        """
        return _try_index_value(
            index_fixings=self.index_fixing.value,
            index_date=self.index_fixing.date,
            index_curve=index_curve,
            index_lag=self.index_lag,
            index_method=self.index_method,
        )

    def try_index_base(
        self,
        index_curve: _BaseCurve_ = NoInput(0),
    ) -> Result[DualTypes]:
        """
        Determine the index base value from fixing or forecast curve, with lazy error raising.

        Parameters
        ----------
        index_curve : _BaseCurve, optional
            The curve from which index values are forecast if required.

        Returns
        -------
        Result[float, Dual, Dual2, Variable]
        """
        return _try_index_value(
            index_fixings=self.index_base.value,
            index_date=self.index_base.date,
            index_curve=index_curve,
            index_lag=self.index_lag,
            index_method=self.index_method,
        )

    def try_index_ratio(
        self,
        index_curve: _BaseCurve_ = NoInput(0),
    ) -> Result[tuple[DualTypes, DualTypes, DualTypes]]:
        """
        Replicates :meth:`~rateslib.periods.parameters._IndexParams.index_ratio` with
        lazy error raising.

        Returns
        -------
        Result[tuple[float, Dual, Dual2, Variable]] for the ratio, numerator, denominator.
        """
        denominator_ = self.try_index_base(index_curve=index_curve)
        if isinstance(denominator_, Err):
            return denominator_
        numerator_ = self.try_index_value(index_curve=index_curve)
        if isinstance(numerator_, Err):
            return numerator_
        n_, d_ = numerator_.unwrap(), denominator_.unwrap()
        return Ok((n_ / d_, n_, d_))

    def index_ratio(
        self,
        index_curve: _BaseCurve_ = NoInput(0),
    ) -> tuple[DualTypes, DualTypes, DualTypes]:
        """
        Calculate the index ratio for the *Period*, including the numerator and denominator.

        .. math::

           I(m) = \\frac{I_{val}(m)}{I_{base}}

        Parameters
        ----------
        index_curve : _BaseCurve, optional
            The curve from which index values are forecast if required.

        Returns
        -------
        tuple[float, Dual, Dual2, Variable] for the ratio, numerator, denominator.
        """
        return self.try_index_ratio(index_curve=index_curve).unwrap()


def _init_or_none_IndexParams(
    _index_base: DualTypes_,
    _index_lag: int_,
    _index_method: IndexMethod | str_,
    _index_fixings: DualTypes | Series[DualTypes] | str_,  # type: ignore[type-var]
    _index_only: bool_,
    _index_base_date: datetime_,
    _index_reference_date: datetime_,
) -> _IndexParams | None:
    if all(
        isinstance(_, NoInput)
        for _ in (
            _index_base,
            _index_lag,
            _index_method,
            _index_fixings,
        )
    ):
        return None
    else:
        if isinstance(_index_base, str):
            raise ValueError(err.VE_INDEX_BASE_NO_STR)
        return _IndexParams(
            _index_base=_index_base,
            _index_lag=_drb(defaults.index_lag, _index_lag),
            _index_method=_get_index_method(_drb(defaults.index_method, _index_method)),
            _index_fixings=_index_fixings,
            _index_base_date=_index_base_date,
            _index_reference_date=_index_reference_date,
            _index_only=_drb(False, _index_only),
        )
