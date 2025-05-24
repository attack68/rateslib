from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Generic, ParamSpec, TypeVar

from rateslib import defaults

if TYPE_CHECKING:
    pass

P = ParamSpec("P")
R = TypeVar("R")


def _no_interior_validation(func: Callable[P, R]) -> Callable[P, R]:
    """
    Used with a Solver to provide a context to set a flag to prevent repetitive validation,
    for example during iteration. After conclusion of the function re-activate validation.
    """

    @wraps(func)
    def wrapper_no_interior_validation(*args: P.args, **kwargs: P.kwargs) -> R:
        self = args[0]
        if self._do_not_validate:  # type: ignore[attr-defined]
            # make no changes: handle recursive no interior validations.
            result = func(*args, **kwargs)
        else:
            # set to no further validation and reset at end of method
            self._do_not_validate = True  # type: ignore[attr-defined]
            result = func(*args, **kwargs)
            self._do_not_validate = False  # type: ignore[attr-defined]
        return result

    return wrapper_no_interior_validation


def _validate_states(func: Callable[P, R]) -> Callable[P, R]:
    """
    Add a decorator to a class instance method to first validate the object state before performing
    additional operations. If a change is detected the implemented `validate_state` function
    is responsible for resetting the cache and updating any `state_id`s.
    """

    @wraps(func)
    def wrapper_validate_states(*args: P.args, **kwargs: P.kwargs) -> R:
        self = args[0]
        self._validate_state()  # type: ignore[attr-defined]
        return func(*args, **kwargs)

    return wrapper_validate_states


def _clear_cache_post(func: Callable[P, R]) -> Callable[P, R]:
    """
    Add a decorator to a class instance method to clear the cache and set a new state
    post performing the function.
    """

    @wraps(func)
    def wrapper_clear_cache(*args: P.args, **kwargs: P.kwargs) -> R:
        self = args[0]
        result = func(*args, **kwargs)
        self._clear_cache()  # type: ignore[attr-defined]
        return result

    return wrapper_clear_cache


def _new_state_post(func: Callable[P, R]) -> Callable[P, R]:
    """
    Add a decorator to a class instance method to clear the cache and set a new state
    post performing the function.
    """

    @wraps(func)
    def wrapper_new_state(*args: P.args, **kwargs: P.kwargs) -> R:
        self = args[0]
        result = func(*args, **kwargs)
        self._set_new_state()  # type: ignore[attr-defined]
        return result

    return wrapper_new_state


class _WithState:
    """
    Record and manage the `state_id` of mutable classes.

    Attributes
    ----------
    _state: int: This is the most recent recorded state reference of this object.
    _mutable_by_association: bool: This is a rateslib definition of whether this object is
        directly mutable and therefore generates its own state id, or whether its state is
        derived from the most recently evaluated state of its associated objects.
    """

    _state: int = 0
    _mutable_by_association: bool = False
    _do_not_validate: bool = False

    def _set_new_state(self) -> None:
        """Set the state_id of a superclass. Some objects which are 'mutable by association'
        will overload the `get_compoisted_state` method to derive a state from their
        associated items."""
        if self._mutable_by_association:
            self._state = self._get_composited_state()
        else:
            self._state = hash(os.urandom(8))  # 64-bit entropy

    def _validate_state(self) -> None:
        """Used by 'mutable by association' objects to evaluate if their own record of
        associated objects states matches the current state of those objects.

        Mutable by update objects have no concept of state validation, they simply maintain
        a *state* id.
        """
        return None

    def _get_composited_state(self) -> int:
        """Used by 'mutable by association' objects to record the state of their associated
        objects and set this as the object's own state."""
        raise NotImplementedError("Must be implemented for 'mutable by association' types")


KT = TypeVar("KT")
VT = TypeVar("VT")


class _WithCache(Generic[KT, VT]):
    _cache: OrderedDict[KT, VT]
    _cache_len: int

    def _cached_value(self, key: KT, val: VT) -> VT:
        """Used to add a value to the cache and control memory size when returning some
        parameter from an object using cache and state management."""
        if defaults.curve_caching and key not in self._cache:
            if self._cache_len < defaults.curve_caching_max:
                self._cache[key] = val
                self._cache_len += 1
            else:
                self._cache.popitem(last=False)
                self._cache[key] = val
        return val

    def _clear_cache(self) -> None:
        """Clear the cache of values on a object controlled by cache and state management.

        Returns
        -------
        None

        Notes
        -----
        This should be used if any modification has been made to the *Curve*.
        Users are advised against making direct modification to *Curve* classes once
        constructed to avoid the issue of un-cleared caches returning erroneous values.

        Alternatively the curve caching as a feature can be set to *False* in ``defaults``.
        """
        self._cache = OrderedDict()
        self._cache_len = 0
