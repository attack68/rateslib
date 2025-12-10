from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from rateslib.enums.generics import NoInput

if TYPE_CHECKING:
    from rateslib.typing import (
        CurvesT_,
        DualTypes,
        FXForwards_,
        Solver_,
        VolT_,
        datetime_,
        str_,
    )


class _WithRate(Protocol):
    """
    Protocol to establish a *rate* pricing metric of any *Instrument* type.
    """

    _rate_scalar: float

    def rate(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
        metric: str_ = NoInput(0),
    ) -> DualTypes:
        raise NotImplementedError(f"`rate` must be implemented for type: {type(self).__name__}")

    def spread(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DualTypes:
        raise NotImplementedError(f"`spread` is not implemented for type: {type(self).__name__}")

    @property
    def rate_scalar(self) -> float:
        """
        A scaling quantity associated with the :class:`~rateslib.solver.Solver` risk calculations.
        """
        return self._rate_scalar
