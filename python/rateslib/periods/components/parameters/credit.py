from __future__ import annotations


class _CreditParams:
    _premium_accrued: bool

    def __init__(self, _premium_accrued: bool) -> None:
        self._premium_accrued = _premium_accrued

    @property
    def premium_accrued(self) -> bool:
        """Whether the premium is accrued within the period to default."""
        return self._premium_accrued
