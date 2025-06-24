from __future__ import annotations

from datetime import datetime

from pandas import DataFrame

from rateslib import defaults
from rateslib.calendars import _get_years_and_months, get_calendar
from rateslib.curves import Curve
from rateslib.default import NoInput
from rateslib.dual import Dual, Dual2
from rateslib.fx import FXForwards, FXRates
from rateslib.instruments.bonds.securities import FixedRateBond
from rateslib.instruments.core import Sensitivities
from rateslib.periods import (
    _get_fx_and_base,
)
from rateslib.solver import Solver


class BondFuture(Sensitivities):
    """
    Create a bond future derivative.

    Parameters
    ----------
    coupon: float
        The nominal coupon rate set on the contract specifications.
    delivery: datetime or 2-tuple of datetimes
        The delivery window first and last delivery day, or a single delivery day.
    basket: tuple of FixedRateBond
        The bonds that are available as deliverables.
    nominal: float, optional
        The nominal amount of the contract.
    contracts: int, optional
        The number of contracts owned or short.
    calendar: str, optional
        The calendar to define delivery days within the delivery window.
    currency: str, optional
        The currency (3-digit code) of the settlement contract.
    calc_mode: str, optional
        The method to calculate conversion factors. See notes.

    Notes
    -----
    Conversion factors (CFs) ``calc_mode`` are:

    - *"ytm"* which calculates the CF as the clean price percent of par with the bond having a
      yield-to-maturity on the first delivery day in the delivery window.
    - *"ust_short"* which applies to CME 2y, 3y and 5y treasury futures. See
      :download:`CME Treasury Conversion Factors<_static/us-treasury-cfs.pdf>`.
    - *"ust_long"* which applies to CME 10y and 30y treasury futures.

    Examples
    --------
    The :meth:`~rateslib.instruments.BondFuture.dlv` method is a summary method which
    displays many attributes simultaneously in a DataFrame.
    This example replicates the screen print in the publication
    *The Futures Bond Basis: Second Edition (p77)* by Moorad Choudhry. To replicate
    that publication exactly no calendar has been provided. Using the London business day
    calendar and would affect the metrics of the third bond to a small degree (i.e.
    set `calendar="ldn"`)

    .. ipython:: python

       kws = dict(
           frequency="S",
           ex_div=7,
           convention="ActActICMA",
           currency="gbp",
           settle=1,
           curves="gilt_curve"
       )
       bonds = [
           FixedRateBond(dt(1999, 1, 1), dt(2009, 12, 7), fixed_rate=5.75, **kws),
           FixedRateBond(dt(1999, 1, 1), dt(2011, 7, 12), fixed_rate=9.00, **kws),
           FixedRateBond(dt(1999, 1, 1), dt(2010, 11, 25), fixed_rate=6.25, **kws),
           FixedRateBond(dt(1999, 1, 1), dt(2012, 8, 6), fixed_rate=9.00, **kws),
       ]
       prices=[102.732, 131.461, 107.877, 134.455]
       ytms=[bond.ytm(price, dt(2000, 3, 16)) for bond, price in zip(bonds, prices)]
       future = BondFuture(
           delivery=(dt(2000, 6, 1), dt(2000, 6, 30)),
           coupon=7.0,
           basket=bonds,
           nominal=100000,
           contracts=10,
           currency="gbp",
       )
       future.dlv(
           future_price=112.98,
           prices=[102.732, 131.461, 107.877, 134.455],
           repo_rate=6.24,
           settlement=dt(2000, 3, 16),
           convention="Act365f",
       )

    Various other metrics can be extracted in isolation including,
    ``notional``, and conversion factors (``cfs``),
    :meth:`~rateslib.instruments.BondFuture.gross_basis`,
    :meth:`~rateslib.instruments.BondFuture.net_basis`,
    :meth:`~rateslib.instruments.BondFuture.implied_repo`,
    :meth:`~rateslib.instruments.BondFuture.ytm`,
    :meth:`~rateslib.instruments.BondFuture.duration`,
    :meth:`~rateslib.instruments.BondFuture.convexity`,
    :meth:`~rateslib.instruments.BondFuture.ctd_index`,

    .. ipython:: python

        future.cfs
        future.notional
        future.gross_basis(
            future_price=112.98,
            prices=prices,
        )
        future.net_basis(
            future_price=112.98,
            prices=prices,
            repo_rate=6.24,
            settlement=dt(2000, 3, 16),
            delivery=dt(2000, 6, 30),
            convention="Act365f"
        )
        future.implied_repo(
            future_price=112.98,
            prices=prices,
            settlement=dt(2000, 3, 16)
        )
        future.ytm(future_price=112.98)
        future.duration(future_price=112.98)
        future.convexity(future_price=112.98)
        future.ctd_index(
            future_price=112.98,
            prices=prices,
            settlement=dt(2000, 3, 16)
        )

    As opposed to the **analogue methods** above, we can also use
    the **digital methods**,
    :meth:`~rateslib.instruments.BondFuture.npv`,
    :meth:`~rateslib.instruments.BondFuture.rate`,
    but we need to create *Curves* and a *Solver* in the usual way.

    .. ipython:: python

       gilt_curve = Curve(
           nodes={
               dt(2000, 3, 15): 1.0,
               dt(2009, 12, 7): 1.0,
               dt(2010, 11, 25): 1.0,
               dt(2011, 7, 12): 1.0,
               dt(2012, 8, 6): 1.0,
           },
           id="gilt_curve",
       )
       solver = Solver(
           curves=[gilt_curve],
           instruments=[(b, (), {"metric": "ytm"}) for b in bonds],
           s=ytms,
           id="gilt_solver",
           instrument_labels=["5.75% '09", "9% '11", "6.25% '10", "9% '12"],
       )

    Sensitivities are also available;
    :meth:`~rateslib.instruments.BondFuture.delta`
    :meth:`~rateslib.instruments.BondFuture.gamma`.

    .. ipython:: python

       future.delta(solver=solver)

    The delta of a *BondFuture* is individually assigned to the CTD. If the CTD changes
    the delta is reassigned.

    .. ipython:: python

       solver.s = [5.3842, 5.2732, 5.2755, 5.52]
       solver.iterate()
       future.delta(solver=solver)
       future.gamma(solver=solver)

    """

    def __init__(
        self,
        coupon: float,
        delivery: datetime | tuple[datetime, datetime],
        basket: tuple[FixedRateBond],
        # last_trading: Optional[int] = None,
        nominal: float | NoInput = NoInput(0),
        contracts: int | NoInput = NoInput(0),
        calendar: str | NoInput = NoInput(0),
        currency: str | NoInput = NoInput(0),
        calc_mode: str | NoInput = NoInput(0),
    ):
        self.currency = defaults.base_currency if currency is NoInput.blank else currency.lower()
        self.coupon = coupon
        if isinstance(delivery, datetime):
            self.delivery = (delivery, delivery)
        else:
            self.delivery = tuple(delivery)
        self.basket = tuple(basket)
        self.calendar = get_calendar(calendar)
        # self.last_trading = delivery[1] if last_trading is NoInput.blank else
        self.nominal = defaults.notional if nominal is NoInput.blank else nominal
        self.contracts = 1 if contracts is NoInput.blank else contracts
        self.calc_mode = (
            defaults.calc_mode_futures if calc_mode is NoInput.blank else calc_mode.lower()
        )
        self._cfs = NoInput(0)

    def __repr__(self):
        return f"<rl.BondFuture at {hex(id(self))}>"

    @property
    def notional(self):
        """
        Return the notional as number of contracts multiplied by contract nominal.

        Returns
        -------
        float
        """
        return self.nominal * self.contracts * -1  # long positions is negative notn

    @property
    def cfs(self):
        """
        Return the conversion factors for each bond in the ordered ``basket``.

        Returns
        -------
        tuple

        Notes
        -----
        This method uses the traditional calculation of obtaining a clean price
        for each bond on the **first delivery date** assuming the **yield-to-maturity**
        is set as the nominal coupon of the bond future, and scaled to 100.

        .. warning::

           Some exchanges, such as EUREX, specify their own conversion factors' formula
           which differs slightly in the definition of yield-to-maturity than the
           implementation offered by *rateslib*. This results in small differences and
           is *potentially* explained in the way dates, holidays and DCFs are handled
           by each calculator.

        For ICE-LIFFE and gilt futures the methods between the exchange and *rateslib*
        align which results in accurate values. Official values can be validated
        against the document
        :download:`ICE-LIFFE Jun23 Long Gilt<_static/long_gilt_initial_jun23.pdf>`.

        For an equivalent comparison with values which do not exactly align see
        :download:`EUREX Jun23 Bond Futures<_static/eurex_bond_conversion_factors.csv>`.

        Examples
        --------

        .. ipython:: python

           kws = dict(
               stub="ShortFront",
               frequency="S",
               calendar="ldn",
               currency="gbp",
               convention="ActActICMA",
               ex_div=7,
               settle=1,
           )
           bonds = [
               FixedRateBond(dt(1999, 1, 1), dt(2009, 12, 7), fixed_rate=5.75, **kws),
               FixedRateBond(dt(1999, 1, 1), dt(2011, 7, 12), fixed_rate=9.00, **kws),
               FixedRateBond(dt(1999, 1, 1), dt(2010, 11, 25), fixed_rate=6.25, **kws),
               FixedRateBond(dt(1999, 1, 1), dt(2012, 8, 6), fixed_rate=9.00, **kws),
           ]
           future = BondFuture(
               delivery=(dt(2000, 6, 1), dt(2000, 6, 30)), coupon=7.0, basket=bonds
           )
           future.cfs

        """
        if self._cfs is NoInput.blank:
            self._cfs = self._conversion_factors()
        return self._cfs

    def _conversion_factors(self):
        if self.calc_mode == "ytm":
            return tuple(bond.price(self.coupon, self.delivery[0]) / 100 for bond in self.basket)
        elif self.calc_mode == "ust_short":
            return tuple(self._cfs_ust(bond, True) for bond in self.basket)
        elif self.calc_mode == "ust_long":
            return tuple(self._cfs_ust(bond, False) for bond in self.basket)
        else:
            raise ValueError("`calc_mode` must be in {'ytm', 'ust_short', 'ust_long'}")

    def _cfs_ust(self, bond: FixedRateBond, short: bool):
        # See CME pdf in doc Notes for formula.
        coupon = bond.fixed_rate / 100.0
        n, z = _get_years_and_months(self.delivery[0], bond.leg1.schedule.termination)
        if not short:
            mapping = {
                0: 0,
                1: 0,
                2: 0,
                3: 3,
                4: 3,
                5: 3,
                6: 6,
                7: 6,
                8: 6,
                9: 9,
                10: 9,
                11: 9,
            }
            z = mapping[z]  # round down number of months to quarters
        if z < 7:
            v = z
        elif short:
            v = z - 6
        else:
            v = 3
        a = 1 / 1.03 ** (v / 6.0)
        b = (coupon / 2) * (6 - v) / 6.0
        if z < 7:
            c = 1 / 1.03 ** (2 * n)
        else:
            c = 1 / 1.03 ** (2 * n + 1)
        d = (coupon / 0.06) * (1 - c)
        factor = a * ((coupon / 2) + c + d) - b
        return round(factor, 4)

    def dlv(
        self,
        future_price: float | Dual | Dual2,
        prices: list[float, Dual, Dual2],
        repo_rate: float | Dual | Dual2 | list | tuple,
        settlement: datetime,
        delivery: datetime | NoInput = NoInput(0),
        convention: str | NoInput = NoInput(0),
        dirty: bool = False,
    ):
        """
        Return an aggregated DataFrame of DeLiVerable metrics.

        Parameters
        ----------
        future_price: float, Dual, Dual2
            The price of the future.
        prices: sequence of float, Dual, Dual2
            The prices of the bonds in the deliverable basket (ordered).
        repo_rate: float, Dual, Dual2 or list/tuple of such
            The repo rates of the bonds to delivery.
        settlement: datetime
            The settlement date of the bonds.
        delivery: datetime, optional
            The date of the futures delivery. If not given uses the final delivery
            day.
        convention: str, optional
            The day count convention applied to the repo rates.
        dirty: bool
            Whether the bond prices are given including accrued interest. Default is *False*.

        Returns
        -------
        DataFrame
        """
        if not isinstance(repo_rate, (tuple, list)):
            r_ = (repo_rate,) * len(self.basket)
        else:
            r_ = tuple(repo_rate)

        df = DataFrame(
            columns=[
                "Bond",
                "Price",
                "YTM",
                "C.Factor",
                "Gross Basis",
                "Implied Repo",
                "Actual Repo",
                "Net Basis",
            ],
            index=range(len(self.basket)),
        )
        df["Price"] = prices
        df["YTM"] = [
            bond.ytm(prices[i], settlement, dirty=dirty) for i, bond in enumerate(self.basket)
        ]
        df["C.Factor"] = self.cfs
        df["Gross Basis"] = self.gross_basis(future_price, prices, settlement, dirty=dirty)
        df["Implied Repo"] = self.implied_repo(
            future_price,
            prices,
            settlement,
            delivery,
            convention,
            dirty=dirty,
        )
        df["Actual Repo"] = r_
        df["Net Basis"] = self.net_basis(
            future_price,
            prices,
            r_,
            settlement,
            delivery,
            convention,
            dirty=dirty,
        )
        df["Bond"] = [
            f"{bond.fixed_rate:,.3f}% {bond.leg1.schedule.termination.strftime('%d-%m-%Y')}"
            for bond in self.basket
        ]
        return df

    def cms(
        self,
        prices: list[float],
        settlement: datetime,
        shifts: list[float],
        delivery: datetime | NoInput = NoInput(0),
        dirty: bool = False,
    ):
        """
        Perform CTD multi-security analysis.

        Parameters
        ----------
        prices: sequence of float, Dual, Dual2
            The prices of the bonds in the deliverable basket (ordered).
        settlement: datetime
            The settlement date of the bonds.
        shifts : list of float
            The scenarios to analyse.
        delivery: datetime, optional
            The date of the futures delivery. If not given uses the final delivery
            day.
        dirty: bool
            Whether the bond prices are given including accrued interest. Default is *False*.

        Returns
        -------
        DataFrame

        Notes
        -----
        This method only operates when the CTD basket has multiple securities
        """
        if len(self.basket) == 1:
            raise ValueError("Multi-security analysis cannot be performed with one security.")
        delivery = self.delivery[1] if delivery is NoInput.blank else delivery

        # build a curve for pricing
        today = self.basket[0].leg1.schedule.calendar.lag(
            settlement,
            -self.basket[0].kwargs["settle"],
            False,
        )
        unsorted_nodes = {
            today: 1.0,
            **{_.leg1.schedule.termination: 1.0 for _ in self.basket},
        }
        bcurve = Curve(
            nodes=dict(sorted(unsorted_nodes.items(), key=lambda _: _[0])),
            convention="act365f",  # use the most natural DCF without scaling
        )
        if dirty:
            metric = "dirty_price"
        else:
            metric = "clean_price"
        solver = Solver(
            curves=[bcurve],
            instruments=[(_, (), {"curves": bcurve, "metric": metric}) for _ in self.basket],
            s=prices,
        )
        if solver.result["status"] != "SUCCESS":
            return ValueError(
                "A bond curve could not be solved for analysis. "
                "See 'Cookbook: Bond Future CTD Multi-Security Analysis'.",
            )
        bcurve._set_ad_order(order=0)  # turn of AD for efficiency

        data = {
            "Bond": [
                f"{bond.fixed_rate:,.3f}% {bond.leg1.schedule.termination.strftime('%d-%m-%Y')}"
                for bond in self.basket
            ],
        }
        for shift in shifts:
            _curve = bcurve.shift(shift, composite=False)
            data.update(
                {
                    shift: self.net_basis(
                        future_price=self.rate(curves=_curve),
                        prices=[_.rate(curves=_curve, metric=metric) for _ in self.basket],
                        repo_rate=_curve.rate(settlement, self.delivery[1], "NONE"),
                        settlement=settlement,
                        delivery=delivery,
                        convention=_curve.convention,
                        dirty=dirty,
                    ),
                },
            )

        _ = DataFrame(data=data)
        return _

    def gross_basis(
        self,
        future_price: float | Dual | Dual2,
        prices: list[float, Dual, Dual2],
        settlement: datetime | NoInput = NoInput(0),
        dirty: bool = False,
    ):
        """
        Calculate the gross basis of each bond in the basket.

        Parameters
        ----------
        future_price: float, Dual, Dual2
            The price of the future.
        prices: sequence of float, Dual, Dual2
            The prices of the bonds in the deliverable basket (ordered).
        settlement: datetime
            The settlement date of the bonds, required only if ``dirty`` is *True*.
        dirty: bool
            Whether the bond prices are given including accrued interest.

        Returns
        -------
        tuple
        """
        if dirty:
            prices_ = tuple(
                prices[i] - bond.accrued(settlement) for i, bond in enumerate(self.basket)
            )
        else:
            prices_ = prices
        return tuple(prices_[i] - self.cfs[i] * future_price for i in range(len(self.basket)))

    def net_basis(
        self,
        future_price: float | Dual | Dual2,
        prices: list[float, Dual, Dual2],
        repo_rate: float | Dual | Dual2 | list | tuple,
        settlement: datetime,
        delivery: datetime | NoInput = NoInput(0),
        convention: str | NoInput = NoInput(0),
        dirty: bool = False,
    ):
        """
        Calculate the net basis of each bond in the basket via the proceeds
        method of repo.

        Parameters
        ----------
        future_price: float, Dual, Dual2
            The price of the future.
        prices: sequence of float, Dual, Dual2
            The prices of the bonds in the deliverable basket (ordered).
        repo_rate: float, Dual, Dual2 or list/tuple of such
            The repo rates of the bonds to delivery.
        settlement: datetime
            The settlement date of the bonds, required only if ``dirty`` is *True*.
        delivery: datetime, optional
            The date of the futures delivery. If not given uses the final delivery
            day.
        convention: str, optional
            The day count convention applied to the repo rates.
        dirty: bool
            Whether the bond prices are given including accrued interest.

        Returns
        -------
        tuple
        """
        if delivery is NoInput.blank:
            f_settlement = self.delivery[1]
        else:
            f_settlement = delivery

        if not isinstance(repo_rate, (list, tuple)):
            r_ = (repo_rate,) * len(self.basket)
        else:
            r_ = repo_rate

        if dirty:
            net_basis_ = tuple(
                bond.fwd_from_repo(
                    prices[i],
                    settlement,
                    f_settlement,
                    r_[i],
                    convention,
                    dirty=dirty,
                )
                - self.cfs[i] * future_price
                - bond.accrued(f_settlement)
                for i, bond in enumerate(self.basket)
            )
        else:
            net_basis_ = tuple(
                bond.fwd_from_repo(
                    prices[i],
                    settlement,
                    f_settlement,
                    r_[i],
                    convention,
                    dirty=dirty,
                )
                - self.cfs[i] * future_price
                for i, bond in enumerate(self.basket)
            )
        return net_basis_

    def implied_repo(
        self,
        future_price: float | Dual | Dual2,
        prices: list[float, Dual, Dual2],
        settlement: datetime,
        delivery: datetime | NoInput = NoInput(0),
        convention: str | NoInput = NoInput(0),
        dirty: bool = False,
    ):
        """
        Calculate the implied repo of each bond in the basket using the proceeds
        method.

        Parameters
        ----------
        future_price: float, Dual, Dual2
            The price of the future.
        prices: sequence of float, Dual, Dual2
            The prices of the bonds in the deliverable basket (ordered).
        settlement: datetime
            The settlement date of the bonds.
        delivery: datetime, optional
            The date of the futures delivery. If not given uses the final delivery
            day.
        convention: str, optional
            The day count convention used in the rate.
        dirty: bool
            Whether the bond prices are given including accrued interest.

        Returns
        -------
        tuple
        """
        if delivery is NoInput.blank:
            f_settlement = self.delivery[1]
        else:
            f_settlement = delivery

        implied_repos = tuple()
        for i, bond in enumerate(self.basket):
            invoice_price = future_price * self.cfs[i]
            implied_repos += (
                bond.repo_from_fwd(
                    price=prices[i],
                    settlement=settlement,
                    forward_settlement=f_settlement,
                    forward_price=invoice_price,
                    convention=convention,
                    dirty=dirty,
                ),
            )
        return implied_repos

    def ytm(
        self,
        future_price: float | Dual | Dual2,
        delivery: datetime | NoInput = NoInput(0),
    ):
        """
        Calculate the yield-to-maturity of the bond future.

        Parameters
        ----------
        future_price : float, Dual, Dual2
            The price of the future.
        delivery : datetime, optional
            The future delivery day on which to calculate the yield. If not given aligns
            with the last delivery day specified on the future.

        Returns
        -------
        tuple
        """
        if delivery is NoInput.blank:
            settlement = self.delivery[1]
        else:
            settlement = delivery
        adjusted_prices = [future_price * cf for cf in self.cfs]
        yields = tuple(
            bond.ytm(adjusted_prices[i], settlement) for i, bond in enumerate(self.basket)
        )
        return yields

    def duration(
        self,
        future_price: float,
        metric: str = "risk",
        delivery: datetime | NoInput = NoInput(0),
    ):
        """
        Return the (negated) derivative of ``price`` w.r.t. ``ytm`` .

        Parameters
        ----------
        future_price : float
            The price of the future.
        metric : str
            The specific duration calculation to return. See notes.
        delivery : datetime, optional
            The delivery date of the contract.

        Returns
        -------
        float

        See Also
        --------
        FixedRateBond.duration: Calculation the risk of a FixedRateBond.

        Example
        -------
        .. ipython:: python

           risk = future.duration(112.98)
           risk

        The difference in yield is shown to be 1bp for the CTD (index: 0)
        when the futures price is adjusted by the risk amount.

        .. ipython:: python

           future.ytm(112.98)
           future.ytm(112.98 + risk[0] / 100)
        """
        if delivery is NoInput.blank:
            f_settlement = self.delivery[1]
        else:
            f_settlement = delivery

        _ = ()
        for i, bond in enumerate(self.basket):
            invoice_price = future_price * self.cfs[i]
            ytm = bond.ytm(invoice_price, f_settlement)
            if metric == "risk":
                _ += (bond.duration(ytm, f_settlement, "risk") / self.cfs[i],)
            else:
                _ += (bond.duration(ytm, f_settlement, metric),)
        return _

    def convexity(
        self,
        future_price: float,
        delivery: datetime | NoInput = NoInput(0),
    ):
        """
        Return the second derivative of ``price`` w.r.t. ``ytm`` .

        Parameters
        ----------
        future_price : float
            The price of the future.
        delivery : datetime, optional
            The delivery date of the contract. If not given uses the last delivery day
            in the delivery window.

        Returns
        -------
        float

        See Also
        --------
        FixedRateBond.convexity: Calculate the convexity of a FixedRateBond.

        Example
        -------
        .. ipython:: python

           risk = future.duration(112.98)
           convx = future.convexity(112.98)
           convx

        Observe the change in risk duration when the prices is increased by 1bp.

        .. ipython:: python

           future.duration(112.98)
           future.duration(112.98 + risk[0] / 100)
        """
        if delivery is NoInput.blank:
            f_settlement = self.delivery[1]
        else:
            f_settlement = delivery

        _ = ()
        for i, bond in enumerate(self.basket):
            invoice_price = future_price * self.cfs[i]
            ytm = bond.ytm(invoice_price, f_settlement)
            _ += (bond.convexity(ytm, f_settlement) / self.cfs[i],)
        return _

    def ctd_index(
        self,
        future_price: float,
        prices: list | tuple,
        settlement: datetime,
        delivery: datetime | NoInput = NoInput(0),
        dirty: bool = False,
        ordered: bool = False,
    ):
        """
        Determine the index of the CTD in the basket from implied repo rate.

        Parameters
        ----------
        future_price : float
            The price of the future.
        prices: sequence of float, Dual, Dual2
            The prices of the bonds in the deliverable basket (ordered).
        settlement: datetime
            The settlement date of the bonds.
        delivery: datetime, optional
            The date of the futures delivery. If not given uses the final delivery
            day.
        dirty: bool
            Whether the bond prices are given including accrued interest.
        ordered : bool, optional
            Whether to return the sorted order of CTD indexes and not just a single index for
            the specific CTD.

        Returns
        -------
        int
        """
        implied_repo = self.implied_repo(
            future_price,
            prices,
            settlement,
            delivery,
            "Act365F",
            dirty,
        )
        if not ordered:
            ctd_index_ = implied_repo.index(max(implied_repo))
            return ctd_index_
        else:
            _ = dict(zip(range(len(implied_repo)), implied_repo))
            _ = dict(sorted(_.items(), key=lambda item: -item[1]))
            return list(_.keys())

    # Digital Methods

    def rate(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        metric: str = "future_price",
        delivery: datetime | NoInput = NoInput(0),
    ):
        """
        Return various pricing metrics of the security calculated from
        :class:`~rateslib.curves.Curve` s.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`Curve` or id or a list of such. A list defines the
            following curves in the order:

              - Forecasting :class:`Curve` for ``leg1``.
              - Discounting :class:`Curve` for ``leg1``.
        solver : Solver, optional
            The numerical :class:`Solver` that constructs ``Curves`` from calibrating
            instruments.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            ``FXRates`` or ``FXForwards`` object, converts from local currency
            into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code), set by default.
            Only used if ``fx`` is an ``FXRates`` or ``FXForwards`` object.
        metric : str in {"future_price", "ytm"}, optional
            Metric returned by the method.
        delivery: datetime, optional
            The date of the futures delivery. If not given uses the final delivery
            day.

        Returns
        -------
        float, Dual, Dual2

        Notes
        -----
        This method determines the *'futures_price'* and *'ytm'*  by assuming a net
        basis of zero and pricing from the cheapest to delivery (CTD).
        """
        metric = metric.lower()
        if metric not in ["future_price", "ytm"]:
            raise ValueError("`metric` must be in {'future_price', 'ytm'}.")

        if delivery is NoInput.blank:
            f_settlement = self.delivery[1]
        else:
            f_settlement = delivery
        prices_ = [
            bond.rate(curves, solver, fx, base, "clean_price", f_settlement) for bond in self.basket
        ]
        future_prices_ = [price / self.cfs[i] for i, price in enumerate(prices_)]
        future_price = min(future_prices_)
        ctd_index = future_prices_.index(min(future_prices_))

        if metric == "future_price":
            return future_price
        elif metric == "ytm":
            return self.basket[ctd_index].ytm(future_price * self.cfs[ctd_index], f_settlement)

    def npv(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
    ):
        """
        Determine the monetary value of the bond future position.

        This method is mainly included to calculate risk sensitivities. The
        monetary value of bond futures is not usually a metric worth considering.
        The profit or loss of a position based on entry level is a more common
        metric, however the initial value of the position does not affect the risk.

        See :meth:`BaseDerivative.npv`.
        """
        future_price = self.rate(curves, solver, fx, base, "future_price")
        fx, base = _get_fx_and_base(self.currency, fx, base)
        npv_ = future_price / 100 * -self.notional
        if local:
            return {self.currency: npv_}
        else:
            return npv_ * fx

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
