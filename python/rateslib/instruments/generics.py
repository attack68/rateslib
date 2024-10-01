from __future__ import annotations

import warnings

from pandas import DataFrame, concat

from rateslib import defaults
from rateslib.curves import Curve
from rateslib.default import NoInput
from rateslib.fx import FXForwards, FXRates
from rateslib.instruments.core import Sensitivities
from rateslib.solver import Solver

# Generic Instruments


class Spread(Sensitivities):
    """
    A spread instrument defined as the difference in rate between two *Instruments*.

    Parameters
    ----------
    instrument1 : Instrument
        The initial instrument, usually the shortest tenor, e.g. 5Y in 5s10s.
    instrument2 : Instrument
        The second instrument, usually the longest tenor, e.g. 10Y in 5s10s.

    Notes
    -----
    When using a :class:`Spread` each *Instrument* must either have pricing parameters
    pre-defined using the appropriate :ref:`pricing mechanisms<mechanisms-doc>` or share
    common pricing parameters defined at price time.

    Examples
    --------
    Creating a dynamic :class:`Spread` where the *Instruments* are dynamically priced,
    and each share the pricing arguments.

    .. ipython:: python

       curve1 = Curve({dt(2022, 1, 1): 1.0, dt(2022, 4, 1):0.995, dt(2022, 7, 1):0.985})
       irs1 = IRS(dt(2022, 1, 1), "3M", "Q")
       irs2 = IRS(dt(2022, 1, 1), "6M", "Q")
       spread = Spread(irs1, irs2)
       spread.npv(curve1)
       spread.rate(curve1)
       spread.cashflows(curve1)

    Creating an assigned :class:`Spread`, where each *Instrument* has its own
    assigned pricing arguments.

    .. ipython:: python

       curve1 = Curve({dt(2022, 1, 1): 1.0, dt(2022, 4, 1):0.995, dt(2022, 7, 1):0.985})
       curve2 = Curve({dt(2022, 1, 1): 1.0, dt(2022, 4, 1):0.99, dt(2022, 7, 1):0.98})
       irs1 = IRS(dt(2022, 1, 1), "3M", "Q", curves=curve1)
       irs2 = IRS(dt(2022, 1, 1), "6M", "Q", curves=curve2)
       spread = Spread(irs1, irs2)
       spread.npv()
       spread.rate()
       spread.cashflows()
    """

    _rate_scalar = 100.0

    def __init__(self, instrument1, instrument2):
        self.instrument1 = instrument1
        self.instrument2 = instrument2

    def __repr__(self):
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def npv(self, *args, **kwargs):
        """
        Return the NPV of the composited object by summing instrument NPVs.

        Parameters
        ----------
        args :
            Positional arguments required for the ``npv`` method of both of the
            underlying ``Instruments``.
        kwargs :
            Keyword arguments required for the ``npv`` method of both of the underlying
            ``Instruments``.

        Returns
        -------
        float, Dual or Dual2

        Notes
        -----

        If the argument ``local`` is added to return a dict of currencies, ensure
        that this is added as a **keyword** argument and not a positional argument.
        I.e. use `local=True`.
        """
        leg1_npv = self.instrument1.npv(*args, **kwargs)
        leg2_npv = self.instrument2.npv(*args, **kwargs)
        if kwargs.get("local", False):
            return {
                k: leg1_npv.get(k, 0) + leg2_npv.get(k, 0) for k in set(leg1_npv) | set(leg2_npv)
            }
        else:
            return leg1_npv + leg2_npv

    # def npv(self, *args, **kwargs):
    #     if len(args) == 0:
    #         args1 = (kwargs.get("curve1", None), kwargs.get("disc_curve1", None))
    #         args2 = (kwargs.get("curve2", None), kwargs.get("disc_curve2", None))
    #     else:
    #         args1 = args
    #         args2 = args
    #     return self.instrument1.npv(*args1) + self.instrument2.npv(*args2)

    def rate(self, *args, **kwargs):
        """
        Return the mid-market rate of the composited via the difference of instrument
        rates.

        Parameters
        ----------
        args :
            Positional arguments required for the ``rate`` method of both of the
            underlying ``Instruments``.
        kwargs :
            Keyword arguments required for the ``rate`` method of both of the underlying
            ``Instruments``.

        Returns
        -------
        float, Dual or Dual2
        """
        leg1_rate = self.instrument1.rate(*args, **kwargs)
        leg2_rate = self.instrument2.rate(*args, **kwargs)
        return (leg2_rate - leg1_rate) * 100.0

    # def rate(self, *args, **kwargs):
    #     if len(args) == 0:
    #         args1 = (kwargs.get("curve1", None), kwargs.get("disc_curve1", None))
    #         args2 = (kwargs.get("curve2", None), kwargs.get("disc_curve2", None))
    #     else:
    #         args1 = args
    #         args2 = args
    #     return self.instrument2.rate(*args2) - self.instrument1.rate(*args1)

    def cashflows(self, *args, **kwargs):
        return concat(
            [
                self.instrument1.cashflows(*args, **kwargs),
                self.instrument2.cashflows(*args, **kwargs),
            ],
            keys=["instrument1", "instrument2"],
        )

    def delta(self, *args, **kwargs):
        """
        Calculate the delta of the *Instrument*.

        For arguments see :meth:`Sensitivities.delta()<rateslib.instruments.Sensitivities.delta>`.
        """
        return super().delta(*args, **kwargs)

    def gamma(self, *args, **kwargs):
        """
        Calculate the gamma of the *Instrument*.

        For arguments see :meth:`Sensitivities.gamma()<rateslib.instruments.Sensitivities.gamma>`.
        """
        return super().gamma(*args, **kwargs)


class Fly(Sensitivities):
    """
    A butterfly instrument which is, mechanically, the spread of two spread instruments.

    Parameters
    ----------
    instrument1 : Instrument
        The initial instrument, usually the shortest tenor, e.g. 5Y in 5s10s15s.
    instrument2 : Instrument
        The second instrument, usually the mid-length tenor, e.g. 10Y in 5s10s15s.
    instrument3 : Instrument
        The third instrument, usually the longest tenor, e.g. 15Y in 5s10s15s.

    Notes
    -----
    When using a :class:`Fly` each *Instrument* must either have pricing parameters
    pre-defined using the appropriate :ref:`pricing mechanisms<mechanisms-doc>` or share
    common pricing parameters defined at price time.

    Examples
    --------
    See examples for :class:`Spread` for similar functionality.
    """

    _rate_scalar = 100.0

    def __init__(self, instrument1, instrument2, instrument3):
        self.instrument1 = instrument1
        self.instrument2 = instrument2
        self.instrument3 = instrument3

    def __repr__(self):
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def npv(self, *args, **kwargs):
        """
        Return the NPV of the composited object by summing instrument NPVs.

        Parameters
        ----------
        args :
            Positional arguments required for the ``npv`` method of both of the
            underlying ``Instruments``.
        kwargs :
            Keyword arguments required for the ``npv`` method of both of the underlying
            ``Instruments``.

        Returns
        -------
        float, Dual or Dual2
        """
        leg1_npv = self.instrument1.npv(*args, **kwargs)
        leg2_npv = self.instrument2.npv(*args, **kwargs)
        leg3_npv = self.instrument3.npv(*args, **kwargs)
        if kwargs.get("local", False):
            return {
                k: leg1_npv.get(k, 0) + leg2_npv.get(k, 0) + leg3_npv.get(k, 0)
                for k in set(leg1_npv) | set(leg2_npv) | set(leg3_npv)
            }
        else:
            return leg1_npv + leg2_npv + leg3_npv

    def rate(self, *args, **kwargs):
        """
        Return the mid-market rate of the composited via the difference of instrument
        rates.

        Parameters
        ----------
        args :
            Positional arguments required for the ``rate`` method of both of the
            underlying ``Instruments``.
        kwargs :
            Keyword arguments required for the ``rate`` method of both of the underlying
            ``Instruments``.

        Returns
        -------
        float, Dual or Dual2
        """
        leg1_rate = self.instrument1.rate(*args, **kwargs)
        leg2_rate = self.instrument2.rate(*args, **kwargs)
        leg3_rate = self.instrument3.rate(*args, **kwargs)
        return (-leg3_rate + 2 * leg2_rate - leg1_rate) * 100.0

    def cashflows(self, *args, **kwargs):
        return concat(
            [
                self.instrument1.cashflows(*args, **kwargs),
                self.instrument2.cashflows(*args, **kwargs),
                self.instrument3.cashflows(*args, **kwargs),
            ],
            keys=["instrument1", "instrument2", "instrument3"],
        )

    def delta(self, *args, **kwargs):
        """
        Calculate the delta of the *Instrument*.

        For arguments see :meth:`Sensitivities.delta()<rateslib.instruments.Sensitivities.delta>`.
        """
        return super().delta(*args, **kwargs)

    def gamma(self, *args, **kwargs):
        """
        Calculate the gamma of the *Instrument*.

        For arguments see :meth:`Sensitivities.gamma()<rateslib.instruments.Sensitivities.gamma>`.
        """
        return super().gamma(*args, **kwargs)


def _instrument_npv(instrument, *args, **kwargs):  # pragma: no cover
    # this function is captured by TestPortfolio pooling but is not registered as a parallel process
    # used for parallel processing with Portfolio.npv
    return instrument.npv(*args, **kwargs)


class Portfolio(Sensitivities):
    """
    Create a collection of *Instruments* to group metrics

    Parameters
    ----------
    instruments : list
        This should be a list of *Instruments*.

    Notes
    -----
    When using a :class:`Portfolio` each *Instrument* must either have pricing parameters
    pre-defined using the appropriate :ref:`pricing mechanisms<mechanisms-doc>` or share
    common pricing parameters defined at price time.

    Examples
    --------
    See examples for :class:`Spread` for similar functionality.
    """

    def __init__(self, instruments):
        if not isinstance(instruments, list):
            raise ValueError("`instruments` should be a list of Instruments.")
        self.instruments = instruments

    def __repr__(self):
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def npv(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
        **kwargs,
    ):
        """
        Return the NPV of the *Portfolio* by summing instrument NPVs.

        For arguments see :meth:`BaseDerivative.npv()<rateslib.instruments.BaseDerivative.npv>`.
        """
        # TODO look at legs.npv where args len is used.
        if not local and base is NoInput.blank and fx is NoInput.blank:
            warnings.warn(
                "No ``base`` currency is inferred, using ``local`` output. To return a single "
                "PV specify a ``base`` currency and ensure an ``fx`` or ``solver.fx`` object "
                "is available to perform the conversion if the currency differs from the local.",
                UserWarning,
            )
            local = True

        # if the pool is 1 do not do any parallel processing and return the single core func
        if defaults.pool == 1:
            return self._npv_single_core(
                curves=curves,
                solver=solver,
                fx=fx,
                base=base,
                local=local,
                **kwargs,
            )

        from functools import partial
        from multiprocessing import Pool

        func = partial(
            _instrument_npv,
            curves=curves,
            solver=solver,
            fx=fx,
            base=base,
            local=local,
            **kwargs,
        )
        p = Pool(defaults.pool)
        results = p.map(func, self.instruments)
        p.close()

        if local:
            _ = DataFrame(results).fillna(0.0)
            _ = _.sum()
            ret = _.to_dict()

            # ret = {}
            # for result in results:
            #     for ccy in result:
            #         if ccy in ret:
            #             ret[ccy] += result[ccy]
            #         else:
            #             ret[ccy] = result[ccy]

        else:
            ret = sum(results)

        return ret

    def _npv_single_core(self, *args, **kwargs):
        if kwargs.get("local", False):
            # dicts = [instrument.npv(*args, **kwargs) for instrument in self.instruments]
            # result = dict(reduce(operator.add, map(Counter, dicts)))

            ret = {}
            for instrument in self.instruments:
                i_npv = instrument.npv(*args, **kwargs)
                for ccy in i_npv:
                    if ccy in ret:
                        ret[ccy] += i_npv[ccy]
                    else:
                        ret[ccy] = i_npv[ccy]
        else:
            _ = (instrument.npv(*args, **kwargs) for instrument in self.instruments)
            ret = sum(_)
        return ret

    def cashflows(self, *args, **kwargs):
        return concat(
            [_.cashflows(*args, **kwargs) for _ in self.instruments],
            keys=[f"inst{i}" for i in range(len(self.instruments))],
        )

    def delta(self, *args, **kwargs):
        """
        Calculate the delta of the *Instrument*.

        For arguments see :meth:`Sensitivities.delta()<rateslib.instruments.Sensitivities.delta>`.
        """
        return super().delta(*args, **kwargs)

    def gamma(self, *args, **kwargs):
        """
        Calculate the gamma of the *Instrument*.

        For arguments see :meth:`Sensitivities.gamma()<rateslib.instruments.Sensitivities.gamma>`.
        """
        return super().gamma(*args, **kwargs)
