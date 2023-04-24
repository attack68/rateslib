.. _gamma-doc:

.. ipython:: python
   :suppress:

   from rateslib.fx import *
   from datetime import datetime as dt

*****************
Gamma Risk
*****************

Calculation
------------------------

Gamma, technically cross-gamma, risks are calculated in *rateslib* in a very similar
manner to delta, albeit the expression of the result is more complicated, since
cross-gamma risks are a matrix of values, whereas delta risk is typically a vector.

The output is a DataFrame indexed with a hierarchical index allowing the information
to be viewed all at once or sliced using pandas indexing tools.

The below gives a full example of constructing solvers with dependency chains, then
constructing a multi-currency portfolio and viewing the gamma risks in local and base
currencies.

First we initialise the :class:`~rateslib.curves.Curve` s. A SOFR, ESTR and
cross-currency curve for valuing EUR cashflows collateralized in USD.

.. ipython:: python

    sofr = Curve(
        nodes={dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 1.0, dt(2042, 1, 1): 1.0},
        id="sofr"
    )
    estr = Curve(
        nodes={dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 1.0, dt(2042, 1, 1): 1.0},
        id="estr"
    )
    eurusd = Curve(
        nodes={dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 1.0, dt(2042, 1, 1): 1.0},
        id="eurusd"
    )

Then we define our :class:`~rateslib.fx.FXForwards` object and associations.

.. ipython:: python

    fxr = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3))
    fxf = FXForwards(fxr, {
        "eureur": estr,
        "eurusd": eurusd,
        "usdusd": sofr
    })

Then we define the instruments that will be
incorporated into our :class:`~rateslib.solver.Solver` s.

.. ipython:: python

   instruments = [
        IRS(dt(2022, 1, 1), "10y", "A", currency="usd", curves="sofr"),
        IRS(dt(2032, 1, 1), "10y", "A", currency="usd", curves="sofr"),
        IRS(dt(2022, 1, 1), "10y", "A", currency="eur", curves="estr"),
        IRS(dt(2032, 1, 1), "10y", "A", currency="eur", curves="estr"),
        XCS(dt(2022, 1, 1), "10y", "A", currency="usd", leg2_currency="usd", curves=["estr", "eurusd", "sofr", "sofr"]),
        XCS(dt(2032, 1, 1), "10y", "A", currency="usd", leg2_currency="eur", curves=["estr", "eurusd", "sofr", "sofr"]),
    ]

Then we solve the SOFR and ESTR curves independently given local currency swap markets.

.. ipython:: python

    sofr_solver= Solver(
        curves=[sofr],
        instruments=instruments[:2],
        s=[3.45, 2.85],
        instrument_labels=["10y", "10y10y"],
        id="sofr",
        fx=fxf
    )
    estr_solver= Solver(
        curves=[estr],
        instruments=instruments[2:4],
        s=[2.25, 0.90],
        instrument_labels=["10y", "10y10y"],
        id="estr",
        fx=fxf
    )

Finally we solve the cross-currency solver with a dependency to the single currency
markets, as specified within the ``pre_solvers`` argument.

.. ipython:: python

    solver= Solver(
        curves=[eurusd],
        instruments=instruments[4:],
        s=[-10, -15],
        instrument_labels=["10y", "10y10y"],
        id="eurusd",
        fx=fxf,
        pre_solvers=[sofr_solver, estr_solver],
    )

Now we create a multi-currency :class:`~rateslib.instruments.Portfolio` and
calculate its cross-gamma.

.. ipython:: python

    pf = Portfolio([
        IRS(dt(2022, 1, 1), "20Y", "A", currency="eur", fixed_rate=2.0, notional=1e8, curves="estr"),
        IRS(dt(2022, 1, 1), "20Y", "A", currency="usd", fixed_rate=1.5, notional=-1.1e8, curves="sofr")
    ])
    cgamma = pf.gamma(solver=solver, base="eur")
    cgamma

We can slice this to display only the EUR risk.

.. ipython:: python

    idx = ("eur", "eur", slice(None), ["estr", "fx"], slice(None))
    cgamma.loc[idx, (slice(None), ["estr", "fx"], slice(None))]
