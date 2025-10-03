from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Adjuster,
        CalTypes,
        Convention,
        DualTypes,
        FloatFixingMethod,
        FloatRateIndex,
        FloatRateSeries,
        Frequency,
        SpreadCompoundMethod,
    )


class _FloatLegRateParams:
    _fixing_method: FloatFixingMethod
    _method_param: int
    _fixing_index: FloatRateIndex
    _float_spread: DualTypes
    _spread_compound_method: SpreadCompoundMethod

    def __init__(
        self,
        *,
        _fixing_method: FloatFixingMethod,
        _method_param: int,
        _fixing_series: FloatRateSeries,
        _fixing_frequency: Frequency,
        _float_spread: DualTypes,
        _spread_compound_method: SpreadCompoundMethod,
    ) -> None:
        self._fixing_method: FloatFixingMethod = _fixing_method
        self._spread_compound_method = _spread_compound_method
        self._fixing_series = _fixing_series
        self._fixing_index = FloatRateIndex(
            frequency=_fixing_frequency,
            series=_fixing_series,
        )
        self._float_spread = _float_spread
        self._method_param: int = _method_param

    @property
    def fixing_series(self) -> FloatRateSeries:
        return self._fixing_series

    @property
    def fixing_index(self) -> FloatRateIndex:
        return self._fixing_index

    @property
    def fixing_convention(self) -> Convention:
        return self.fixing_index.convention

    @property
    def fixing_modifier(self) -> Adjuster:
        return self.fixing_index.modifier

    @property
    def fixing_frequency(self) -> Frequency:
        return self.fixing_index.frequency

    @property
    def fixing_calendar(self) -> CalTypes:
        return self.fixing_index.calendar

    @property
    def fixing_method(self) -> FloatFixingMethod:
        return self._fixing_method

    @property
    def method_param(self) -> int:
        return self._method_param

    @property
    def float_spread(self) -> DualTypes:
        return self._float_spread

    @float_spread.setter
    def float_spread(self, value: DualTypes) -> None:
        self._float_spread = value

    @property
    def spread_compound_method(self) -> SpreadCompoundMethod:
        return self._spread_compound_method


# def _init_FloatLegRateParams(
#     *,
#     _fixing_method: FloatFixingMethod,
#     _method_param: int,
#     _fixing_series: FloatRateSeries,
#     _fixing_frequency: Frequency,
#     _float_spread: DualTypes,
#     _spread_compound_method: SpreadCompoundMethod,
# ) -> _FloatLegRateParams:
#     return _FloatLegRateParams(
#         _fixing_method = _get_float_fixing_method(_drb(defaults.fixing_method)),
#         _method_param = _drb(defaults.method_param. _method_param),
#         _fixing_series = _fixing_series,
#         _fixing_series: FloatRateSeries,
#         _fixing_frequency: Frequency,
#         _float_spread: DualTypes,
#         _spread_compound_method: SpreadCompoundMethod,
#     )
