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

from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.enums.generics import NoInput
from rateslib.scheduling import Schedule

if TYPE_CHECKING:
    from rateslib.typing import (
        Any,
        str_,
    )


def _get_args_from_spec(spec: str_) -> dict[str, Any]:
    """
    Get ``spec`` args from ``defaults`` or empty dict.
    """
    if isinstance(spec, NoInput):
        return {}
    return defaults.spec.get(spec.lower(), {})


def _update_not_noinput(base_kwargs: dict[str, Any], new_kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Update the `base_kwargs` with `new_kwargs` (user values) unless those new values are NoInput.
    """
    updaters = {
        k: v for k, v in new_kwargs.items() if k not in base_kwargs or not isinstance(v, NoInput)
    }
    return {**base_kwargs, **updaters}


def _update_with_defaults(
    base_kwargs: dict[str, Any], default_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """
    Update the `base_kwargs` with `default_kwargs` if the base_values are NoInput.blank.
    """
    updaters = {
        k: v
        for k, v in default_kwargs.items()
        if k in base_kwargs and base_kwargs[k] is NoInput.blank
    }
    return {**base_kwargs, **updaters}


def _inherit_or_negate(kwargs: dict[str, Any], ignore_blank: bool = False) -> dict[str, Any]:
    """Amend the values of leg2 kwargs if they are defaulted to inherit or negate from leg1."""

    def _replace(k: str, v: Any) -> Any:
        # either inherit or negate the value in leg2 from that in leg1
        if "leg2_" in k:
            if not isinstance(v, NoInput):
                return v  # do nothing if the attribute is an input

            try:
                leg1_v = kwargs[k[5:]]
            except KeyError:
                return v

            if leg1_v is NoInput.blank:
                if ignore_blank:
                    return v  # this allows an inheritor or negator to be called a second time
                else:
                    return NoInput(0)

            if v is NoInput(-1):
                if isinstance(leg1_v, list):
                    return [_ * -1.0 for _ in leg1_v]
                elif isinstance(leg1_v, tuple):
                    return tuple([_ * -1.0 for _ in leg1_v])
                else:
                    return leg1_v * -1.0
            elif v is NoInput(1):
                return leg1_v
        return v  # do nothing to leg1 attributes

    return {k: _replace(k, v) for k, v in kwargs.items()}


def _convert_to_schedule_kwargs(kwargs: dict[str, Any], leg: int) -> dict[str, Any]:
    _ = "" if leg == 1 else "leg2_"

    ex_div = kwargs.pop(f"{_}ex_div", NoInput(0))
    if isinstance(ex_div, int):
        ex_div = -1 * ex_div  # negate this input for business days backwards

    kwargs[f"{_}schedule"] = Schedule(
        effective=kwargs.pop(f"{_}effective", NoInput(0)),
        termination=kwargs.pop(f"{_}termination", NoInput(0)),
        frequency=kwargs.pop(f"{_}frequency", NoInput(0)),
        stub=kwargs.pop(f"{_}stub", NoInput(0)),
        front_stub=kwargs.pop(f"{_}front_stub", NoInput(0)),
        back_stub=kwargs.pop(f"{_}back_stub", NoInput(0)),
        roll=kwargs.pop(f"{_}roll", NoInput(0)),
        eom=kwargs.pop(f"{_}eom", NoInput(0)),
        modifier=kwargs.pop(f"{_}modifier", NoInput(0)),
        calendar=kwargs.pop(f"{_}calendar", NoInput(0)),
        payment_lag=kwargs.pop(f"{_}payment_lag", NoInput(0)),
        payment_lag_exchange=kwargs.pop(f"{_}payment_lag_exchange", NoInput(0)),
        extra_lag=ex_div,
    )
    return kwargs


class _KWArgs:
    """
    Class to manage keyword argument population of *Leg* based *Instruments*.

    This will first populate any provided ``spec`` arguments if given.
    Second, the user input arguments that are specific values will overwrite these.
    Thridly, system ``defaults`` wil be populated.
    Finally, any remaining NoInput arguments of leg2 that are set to `inherit` or `negate` will
    derive their values from leg1.
    """

    @property
    def leg1(self) -> dict[str, Any]:
        """Keyword arguments pass to construction of *Leg1*."""
        return self._leg1_args

    @property
    def leg2(self) -> dict[str, Any]:
        """Keyword arguments pass to construction of *Leg2*."""
        return self._leg2_args

    @property
    def meta(self) -> dict[str, Any]:
        """Meta keyword arguments associated with the *Instrument*."""
        return self._meta_args

    def __init__(
        self,
        user_args: dict[str, Any],
        default_args: dict[str, Any] | None = None,
        meta_args: list[str] | None = None,
        spec: str_ = NoInput(0),
    ) -> None:
        default_args_ = default_args or {}
        meta_args_ = meta_args or []

        kwargs = _get_args_from_spec(spec)
        kwargs = _update_not_noinput(kwargs, user_args)
        kwargs = _update_with_defaults(kwargs, default_args_)
        kwargs = _inherit_or_negate(kwargs)

        self._meta_args = {}
        for k in meta_args_:
            if k in kwargs:
                self._meta_args[k] = kwargs.pop(k)
        self._leg2_args = {k[5:]: v for k, v in kwargs.items() if "leg2_" in k}
        self._leg1_args = {k: v for k, v in kwargs.items() if "leg2_" not in k}

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, _KWArgs):
            return False
        else:
            # bools = [
            #     self.leg1.keys() == other.leg1.keys(),
            #     self.leg2.keys() == other.leg2.keys(),
            #     self.meta.keys() == other.meta.keys(),
            #     all(self.leg1[k] == other.leg1[k] for k in self.leg1.keys()),
            #     all(self.leg2[k] == other.leg2[k] for k in self.leg2.keys()),
            #     all(self.meta[k] == other.meta[k] for k in self.meta.keys()),
            # ]
            bools = [
                self.leg1 == other.leg1,
                self.leg2 == other.leg2,
                self.meta == other.meta,
            ]
            return all(bools)
