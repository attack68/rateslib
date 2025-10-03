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
    An amortization schedule for any :class:`~rateslib.legs.base.BaseLeg`.

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

      - *"to_zero": each period will be a constant value ending with zero implied ending balance.
      - *"{float}%": each period will amortize by a constant percentage of the outstanding balance.

    """

    _type: _AmortizationType
    amortization: tuple[DualTypes, ...]
    outstanding: tuple[DualTypes, ...]

    def __init__(
        self,
        n: int,
        initial: DualTypes,
        amortization: DualTypes_ | list[DualTypes] | tuple[DualTypes, ...] | str = NoInput(0),
    ) -> None:
        if isinstance(amortization, NoInput):
            self._type = _AmortizationType.NoAmortization
            self.amortization = (0.0,) * (n - 1)
            self.outstanding = (initial,) * n
        elif isinstance(amortization, list | tuple):
            self._type = _AmortizationType.CustomSchedule
            if len(amortization) != (n - 1):
                raise ValueError(
                    "Custom amortisation schedules must have `n-1` amortization amounts for `n` "
                    f"periods.\nGot '{len(amortization)}' amounts for '{n}' periods."
                )
            self.amortization = tuple(amortization)
            outstanding = [initial]
            for value in amortization:
                outstanding.append(outstanding[-1] - value)
            self.outstanding = tuple(outstanding)
        elif isinstance(amortization, str):
            if amortization.lower() == "to_zero":
                self._type = _AmortizationType.ConstantPeriod
                self.amortization = (initial / n,) * (n - 1)
                self.outstanding = (initial,) + tuple([initial * (1 - i / n) for i in range(1, n)])
            elif amortization[-1] == "%":
                self._type = _AmortizationType.CustomSchedule
                amortization_ = [initial * float(amortization[:-1]) / 100]
                outstanding_ = [initial]
                for i in range(1, n):
                    outstanding_.append(outstanding_[-1] - amortization_[-1])
                    if i != n - 1:
                        amortization_.append(outstanding_[-1] * float(amortization[:-1]) / 100)
                self.outstanding = tuple(outstanding_)
                self.amortization = tuple(amortization_)
            else:
                raise ValueError("`amortization` as string must be one of 'to_zero', '{float}%'.")
        else:  # isinstance(amortization, DualTypes)
            self._type = _AmortizationType.ConstantPeriod
            self.amortization = (amortization,) * (n - 1)
            self.outstanding = (initial,) + tuple([initial - amortization * i for i in range(1, n)])

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
