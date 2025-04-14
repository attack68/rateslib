.. _c-fx-smile-doc:

.. ipython:: python
   :suppress:

   import warnings
   warnings.filterwarnings('always')
   from rateslib.solver import *
   from rateslib.instruments import *
   from rateslib.fx_volatility import FXDeltaVolSmile, FXDeltaVolSurface, FXSabrSmile
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   from pandas import DataFrame

*********************************
FX Vol Surfaces
*********************************

The ``rateslib.fx_volatility`` module includes classes for *Smiles* and *Surfaces*
which can be used to price *FX Options* and *FX Option Strategies*.

.. autosummary::
   rateslib.fx_volatility.FXDeltaVolSmile
   rateslib.fx_volatility.FXSabrSmile
   rateslib.fx_volatility.FXDeltaVolSurface

Introduction and FX Volatility Smiles
*************************************

*Ratelib* offers two different *Smile* models for pricing FX volatility at a given expiry. An
:class:`~rateslib.fx_volatility.FXDeltaVolSmile` and an
:class:`~rateslib.fx_volatility.FXSabrSmile`.

The :class:`~rateslib.fx_volatility.FXDeltaVolSmile` is parametrised by a series of
*(delta-index, vol)* node points interpolated by a cubic spline. This interpolation is
automatically constructed with knot sequences that adjust to the number of given ``nodes``:

- Providing only one node, e.g. *(0.5, 11.0)*, will create a constant volatility level, here at 11%.
- Providing two nodes, e.g. (0.25, 8.0), (0.75, 10.0) will create a straight line gradient
  across the whole delta axis, here rising by 1% volatility every 0.25 in *delta-index*.
- Providing more nodes (appropriately calibrated) will create a traditional smile shape with
  the mentioned interpolation structure.

An *FXDeltaVolSmile* must also be initialised with a ``delta_type`` to define how it references
delta on its index. It can still be used to price *Instruments* even
if their *delta types* are different. That is, an *FXDeltaVolSmile* defined by *"forward"* delta
types can still price *FXOptions* defined with *"spot"* delta types or *premium adjusted*
delta types due to appropriate mathematical conversions.

An :class:`~rateslib.fx_volatility.FXSabrSmile` is a *Smile* parametrised by the
conventional :math:`\alpha, \beta, \rho, \nu` variables of the SABR model. The parameter
:math:`\beta` is considered a hyper-parameter and will not be varied by a
:class:`~rateslib.solver.Solver` but :math:`\alpha, \rho, \nu` will be varied.

Both *Smiles* must also be initialised with:

- An ``eval_date`` which serves the same purpose as the initial node point on a *Curve*,
  and indicates *'today'* or *'horizon'*.
- An ``expiry``, for which options priced with this *Smile* must have an equivalent
  expiry or errors will be raised.

An example of an *FXDeltaVolSmile* is shown below.

.. ipython:: python

   smile = FXDeltaVolSmile(
       eval_date=dt(2000, 1, 1),
       expiry=dt(2000, 7, 1),
       nodes={
           0.25: 10.3,
           0.5: 9.1,
           0.75: 10.8
       },
       delta_type="forward"
   )
   #  -->  smile.plot()
   #  -->  smile.plot(x_axis="moneyness")

.. container:: twocol

   .. container:: leftside50

      **Delta-Index vs Vol Plot**

      .. plot::

         from rateslib.fx_volatility import FXDeltaVolSmile
         from datetime import datetime as dt
         smile = FXDeltaVolSmile(
             eval_date=dt(2000, 1, 1),
             expiry=dt(2000, 7, 1),
             nodes={
                 0.25: 10.3,
                 0.5: 9.1,
                 0.75: 10.8
             },
             delta_type="forward"
         )
         fig, ax, lines = smile.plot()
         plt.show()
         plt.close()

   .. container:: rightside50

      **Moneyness vs Vol Plot**

      .. plot::

         from rateslib.fx_volatility import FXDeltaVolSmile
         from datetime import datetime as dt
         smile = FXDeltaVolSmile(
             eval_date=dt(2000, 1, 1),
             expiry=dt(2000, 7, 1),
             nodes={
                 0.25: 10.3,
                 0.5: 9.1,
                 0.75: 10.8
             },
             delta_type="forward"
         )
         fig, ax, lines = smile.plot(x_axis="moneyness")
         plt.show()
         plt.close()

Constructing a Smile
*********************

It is expected that *Smiles* will typically be calibrated to market prices, similar to
interest rate curves.

The following data describes *Instruments* to calibrate the EURUSD FX volatility surface on 7th May 2024.
We will take a cross-section of this data, at the 3-week expiry (28th May 2024), and create
both an *FXDeltaVolSmile* and *FXSabrSmile*.

.. image:: _static/fx_eurusd_3m_surf.PNG
  :alt: EURUSD FX volatility surface prices on 7th May 2024
  :width: 489

FX Options are **multi-currency derivative** *Instruments* and require an :class:`~rateslib.fx.FXForwards`
framework for pricing. We will do this first using other prevailing market data,
i.e. local currency interest rates at 3.90% and 5.32%, and an FX Swap rate at 8.85 points.

.. ipython:: python

   # Define the interest rate curves for EUR, USD and X-Ccy basis
   usdusd = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, calendar="nyc", id="usdusd")
   eureur = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, calendar="tgt", id="eureur")
   eurusd = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, id="eurusd")

   # Create an FX Forward market with spot FX rate data
   fxf = FXForwards(
       fx_rates=FXRates({"eurusd": 1.0760}, settlement=dt(2024, 5, 9)),
       fx_curves={"eureur": eureur, "usdusd": usdusd, "eurusd": eurusd},
   )

   pre_solver = Solver(
       curves=[eureur, eurusd, usdusd],
       instruments=[
           IRS(dt(2024, 5, 9), "3W", spec="eur_irs", curves="eureur"),
           IRS(dt(2024, 5, 9), "3W", spec="usd_irs", curves="usdusd"),
           FXSwap(dt(2024, 5, 9), "3W", pair="eurusd", curves=[None, "eurusd", None, "usdusd"]),
       ],
       s=[3.90, 5.32, 8.85],
       fx=fxf,
       id="rates_sv",
   )

Since EURUSD *Options* are **not** premium adjusted and the premium currency is USD we will match
the *FXDeltaVolSmile* with this definition and set it to a ``delta_type`` of *'spot'*, matching
the market convention of these quoted instruments.
Since we have 5 calibrating instruments we can safely utilise 5 degrees of freedom.

.. ipython:: python

   dv_smile = FXDeltaVolSmile(
       nodes={
           0.10: 10.0,
           0.25: 10.0,
           0.50: 10.0,
           0.75: 10.0,
           0.90: 10.0,
       },
       eval_date=dt(2024, 5, 7),
       expiry=dt(2024, 5, 28),
       delta_type="spot",
       id="eurusd_3w_smile"
   )

   sabr_smile = FXSabrSmile(
       nodes={
           "alpha": 0.10,  # default vol level set to 10%
           "beta": 1.0,  # model is fully lognormal
           "rho": 0.10,
           "nu": 1.0,  # initialised with curvature
       },
       eval_date=dt(2024, 5, 7),
       expiry=dt(2024, 5, 28),
       id="eurusd_3w_smile"
   )

The above *FXDeltaVolSmile* is initialised as a flat vol at 10%, whilst the *FXSabrSmile*
is initialised with around 10% with some shallow curvature. In order to calibrate
these, we need to create the pricing
instruments, given in the market prices data table.

.. ipython:: python

   # Setup the Solver instrument calibration for FXOptions and vol Smiles
   option_args=dict(
       pair="eurusd", expiry=dt(2024, 5, 28), calendar="tgt|fed", delta_type="spot",
       curves=[None, "eurusd", None, "usdusd"], vol="eurusd_3w_smile"
   )
   dv_solver = Solver(
       pre_solvers=[pre_solver],
       curves=[dv_smile],
       instruments=[
           FXStraddle(strike="atm_delta", **option_args),
           FXRiskReversal(strike=("-25d", "25d"), **option_args),
           FXRiskReversal(strike=("-10d", "10d"), **option_args),
           FXBrokerFly(strike=(("-25d", "25d"), "atm_delta"), **option_args),
           FXBrokerFly(strike=(("-10d", "10d"), "atm_delta"), **option_args),
       ],
       s=[5.493, -0.157, -0.289, 0.071, 0.238],
       fx=fxf,
       id="dv_solver",
   )

The *FXSabrSmile* can be similarly calibrated.

.. ipython:: python

   sabr_solver = Solver(
       pre_solvers=[pre_solver],
       curves=[sabr_smile],
       instruments=[
           FXStraddle(strike="atm_delta", **option_args),
           FXRiskReversal(strike=("-25d", "25d"), **option_args),
           FXRiskReversal(strike=("-10d", "10d"), **option_args),
           FXBrokerFly(strike=(("-25d", "25d"), "atm_delta"), **option_args),
           FXBrokerFly(strike=(("-10d", "10d"), "atm_delta"), **option_args),
       ],
       s=[5.493, -0.157, -0.289, 0.071, 0.238],
       fx=fxf,
       id="sabr_solver",
   )

   dv_smile.plot(f=fxf.rate("eurusd", dt(2024, 5, 30)), x_axis="delta", labels=["DeltaVol", "Sabr"])

.. container:: twocol

   .. container:: leftside50

      .. plot::
         :caption: Rateslib Vol Smile: 'delta index'

         from rateslib.curves import Curve
         from rateslib.instruments import *
         from rateslib.fx_volatility import FXDeltaVolSmile, FXSabrSmile
         from rateslib.fx import FXRates, FXForwards
         from rateslib.solver import Solver
         import matplotlib.pyplot as plt
         from datetime import datetime as dt
         dv_smile = FXDeltaVolSmile(
             nodes={
                 0.10: 10.0,
                 0.25: 10.0,
                 0.50: 10.0,
                 0.75: 10.0,
                 0.90: 10.0,
             },
             eval_date=dt(2024, 5, 7),
             expiry=dt(2024, 5, 28),
             delta_type="spot",
             id="eurusd_3w_smile"
         )
         sabr_smile = FXSabrSmile(
             nodes={
                 "alpha": 0.10,
                 "beta": 1.0,
                 "rho": 0.10,
                 "nu": 1.0,
             },
             eval_date=dt(2024, 5, 7),
             expiry=dt(2024, 5, 28),
             id="eurusd_3w_smile"
         )
         # Define the interest rate curves for EUR, USD and X-Ccy basis
         eureur = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, calendar="tgt", id="eureur")
         eurusd = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, id="eurusd")
         usdusd = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, calendar="nyc", id="usdusd")
         # Create an FX Forward market with spot FX rate data
         fxf = FXForwards(
             fx_rates=FXRates({"eurusd": 1.0760}, settlement=dt(2024, 5, 9)),
             fx_curves={"eureur": eureur, "usdusd": usdusd, "eurusd": eurusd},
         )
         # Setup the Solver instrument calibration for rates Curves and vol Smiles
         option_args=dict(
             pair="eurusd", expiry=dt(2024, 5, 28), calendar="tgt", delta_type="spot",
             curves=[None, "eurusd", None, "usdusd"], vol="eurusd_3w_smile"
         )
         pre_solver = Solver(
             curves=[eureur, eurusd, usdusd],
             instruments=[
                 IRS(dt(2024, 5, 9), "3W", spec="eur_irs", curves="eureur"),
                 IRS(dt(2024, 5, 9), "3W", spec="usd_irs", curves="usdusd"),
                 FXSwap(dt(2024, 5, 9), "3W", currency="eur", leg2_currency="usd", curves=[None, "eurusd", None, "usdusd"]),
             ],
             s=[3.90, 5.32, 8.85],
             fx=fxf,
         )
         sabr_solver = Solver(
             pre_solvers=[pre_solver],
             curves=[sabr_smile],
             instruments=[
                 FXStraddle(strike="atm_delta", **option_args),
                 FXRiskReversal(strike=("-25d", "25d"), **option_args),
                 FXRiskReversal(strike=("-10d", "10d"), **option_args),
                 FXBrokerFly(strike=(("-25d", "25d"), "atm_delta"), **option_args),
                 FXBrokerFly(strike=(("-10d", "10d"), "atm_delta"), **option_args),
             ],
             s=[5.493, -0.157, -0.289, 0.071, 0.238],
             fx=fxf,
             id="sabr_solver",
         )
         dv_solver = Solver(
             pre_solvers=[pre_solver],
             curves=[dv_smile],
             instruments=[
                 FXStraddle(strike="atm_delta", **option_args),
                 FXRiskReversal(strike=("-25d", "25d"), **option_args),
                 FXRiskReversal(strike=("-10d", "10d"), **option_args),
                 FXBrokerFly(strike=(("-25d", "25d"), "atm_delta"), **option_args),
                 FXBrokerFly(strike=(("-10d", "10d"), "atm_delta"), **option_args),
             ],
             s=[5.493, -0.157, -0.289, 0.071, 0.238],
             fx=fxf,
             id="dv_solver",
         )
         fig, ax, line = dv_smile.plot(f=fxf.rate("eurusd", dt(2024, 5, 30)), x_axis="delta", comparators=[sabr_smile], labels=["DeltaVol", "Sabr"])
         plt.show()
         plt.close()

   .. container:: rightside50

      |
      |

      .. figure:: _static/fx_eurusd_3w_smile.PNG
         :alt: BBG FENICS EURUSD Smile on 7th May 2024
         :width: 320

         BBG Fenics Vol Smile

      |
      |


FX Volatility Surfaces
**********************

An :class:`~rateslib.fx_volatility.FXDeltaVolSurface` in *rateslib* is a collection of
multiple, cross-sectional :class:`~rateslib.fx_volatility.FXDeltaVolSmile` where:

- each cross-sectional *Smile* will represent a *Smile* at that explicit *expiry*,
- the *delta type* and the *delta indexes* on each cross-sectional *Smile* are the same,
- each *Smile* has its own calibrated node values,
- *Smiles* for *expiries* that do not pre-exist are generated with an interpolation
  scheme that uses linear total variance, which is equivalent to flat-forward volatility

To demonstrate this, we will use an example adapted from Iain Clark's *Foreign Exchange
Option Pricing: A Practitioner's Guide*.

The ``eval_date`` is fictionally assumed to be 3rd May 2009 and the FX spot rate is 1.34664,
and the continuously compounded EUR and USD rates are 1.0% and 0.4759..% respectively. With these
we will be able to closely match his values for option strikes.

.. ipython:: python

   # Setup the FXForward market...
   eur = Curve({dt(2009, 5, 3): 1.0, dt(2011, 5, 10): 1.0})
   usd = Curve({dt(2009, 5, 3): 1.0, dt(2011, 5, 10): 1.0})
   fxf = FXForwards(
       fx_rates=FXRates({"eurusd": 1.34664}, settlement=dt(2009, 5, 5)),
       fx_curves={"eureur": eur, "usdusd": usd, "eurusd": eur},
   )
   solver = Solver(
       curves=[eur, usd],
       instruments=[
           Value(dt(2009, 5, 4), curves=eur, metric="cc_zero_rate"),
           Value(dt(2009, 5, 4), curves=usd, metric="cc_zero_rate")
       ],
       s=[1.00, 0.4759550366220911],
       fx=fxf,
   )

His *Table 4.2* is shown below, which outlines the delta type of the used instruments at their respective tenors,
and the ATM-delta straddle, the 25-delta broker-fly and the 25-delta risk reversal market volatility prices.

.. ipython:: python

   data = DataFrame(
       data = [["spot", 18.25, 0.95, -0.6], ["forward", 17.677, 0.85, -0.562]],
       index=["1y", "2y"],
       columns=["Delta Type", "ATM", "25dBF", "25dRR"],
   )
   data

Constructing a Surface
**********************

We will now create a *Surface* that will be calibrated by those given rates.
The *Surface* is initialised at a flat 18% volatility.

.. ipython:: python

   surface = FXDeltaVolSurface(
       eval_date=dt(2009, 5, 3),
       delta_indexes=[0.25, 0.5, 0.75],
       expiries=[dt(2010, 5, 3), dt(2011, 5, 3)],
       node_values=np.ones((2, 3))* 18.0,
       delta_type="forward",
       id="surface",
   )

The calibration of the *Surface* requires a *Solver* that will iterate and update the surface
node values until convergence with the given instrument rates.

.. note::

   The *Surface* is
   parametrised by a *'forward'* *delta type* but that the 1Y *Instruments* use *'spot'*.
   Internally this is all handled appropriately with necessary conversions, but it is the users
   responsibility to label the *Surface* and *Instrument* with the correct types. As Clark and
   others highlight "failing to take [correct delta types] into account introduces a mismatch -
   large enough to be relevant for calibration and pricing, but small enough that it may not be
   noticed at first". Parametrising the *Surface* with a *'forward'* delta type is the **recommended**
   choice because it is more standardised and the configuration of which *delta types* to use for
   the *Instruments* can be a separate consideration.

   For performance reasons it is recommended to match unadjusted delta type *Surfaces* with
   calibrating *Instruments* that also have unadjusted delta types. And vice versa with premium adjusted
   delta types. However, *rateslib* has internal root solvers which can handle these cross-delta type
   specifications, although it degrades the performance of the *Solver* because the calculations are more
   difficult. Mixing 'spot' and 'forward' is not a difficult distinction to refactor and that does
   not cause performance degradation.

.. ipython:: python

   fx_args_0 = dict(
       pair="eurusd",
       curves=[None, eur, None, usd],
       expiry=dt(2010, 5, 3),
       delta_type="spot",
       vol="surface",
   )
   fx_args_1 = dict(
       pair="eurusd",
       curves=[None, eur, None, usd],
       expiry=dt(2011, 5, 3),
       delta_type="forward",
       vol="surface",
   )

   solver = Solver(
       surfaces=[surface],
       instruments=[
           FXStraddle(strike="atm_delta", **fx_args_0),
           FXBrokerFly(strike=(("-25d", "25d"), "atm_delta"), **fx_args_0),
           FXRiskReversal(strike=("-25d", "25d"), **fx_args_0),
           FXStraddle(strike="atm_delta", **fx_args_1),
           FXBrokerFly(strike=(("-25d", "25d"), "atm_delta"), **fx_args_1),
           FXRiskReversal(strike=("-25d", "25d"), **fx_args_1),
       ],
       s=[18.25, 0.95, -0.6, 17.677, 0.85, -0.562],
       fx=fxf,
   )

The table below is *rateslib's* replicated calculations of Clark's Table 4.5.
Note that due to:

- using a different parametric form for *Smiles* (i.e. a natural cubic spline),
- inferring his FX forwards market rates,
- and not necessarily knowing the exact dates and holiday calendars of his example,

this produces minor deviations from his calculated values.

.. ipython:: python
   :suppress:

   args = dict(
       pair="eurusd",
       curves=[None, eur, None, usd],
       vol=surface,
       delta_type="forward"
   )

   ops = [
       FXPut(strike="-25d", expiry=dt(2010, 5, 3), **args),
       FXPut(strike="atm_delta", expiry=dt(2010, 5, 3), **args),
       FXCall(strike="25d", expiry=dt(2010, 5, 3), **args),
       FXPut(strike="-25d", expiry=dt(2010, 11, 3), **args),
       FXPut(strike="atm_delta", expiry=dt(2010, 11, 3), **args),
       FXCall(strike="25d", expiry=dt(2010, 11, 3), **args),
       FXPut(strike="-25d", expiry=dt(2011, 5, 3), **args),
       FXPut(strike="atm_delta", expiry=dt(2011, 5, 3), **args),
       FXCall(strike="25d", expiry=dt(2011, 5, 3), **args),
   ]
   for op in ops:
       op.rate(fx=fxf)

   strikes = [float(_._pricing.k) for _ in ops]
   vols = [float(_._pricing.vol) for _ in ops]
   data2 = DataFrame(
       data=[strikes[0:3], vols[0:3], strikes[3:6], vols[3:6], strikes[6:9], vols[6:9]],
       index=[("1y", "k"), ("1y", "vol"), ("18m", "k"), ("18m", "vol"), ("2y", "k"), ("2y", "vol")],
       columns=["25d Put", "ATM Put", "25d Call"]
   )

.. ipython:: python

   with option_context("display.float_format", lambda x: '%.4f' % x):
       print(data2)

See the article entitled **FX Volatility Surface Temporal Interpolation** in the
:ref:`Cookbook <cookbook-doc>` to read more about time-weighted volatility, and accounting for
weekends and holidays etc..

Plotting
*********

Three relevant cross-sectional *Smiles* from above are plotted.

.. ipython:: python

   sm12 = surface.smiles[0]
   sm18 = surface.get_smile(dt(2010, 11, 3))
   sm24 = surface.smiles[1]
   sm12.plot(comparators=[sm18, sm24], labels=["1y", "18m", "2y"])

.. plot::

   from rateslib.curves import Curve
   from rateslib.solver import Solver
   from rateslib.fx import FXForwards, FXRates
   from rateslib.instruments import FXStraddle, FXRiskReversal, FXBrokerFly, Value
   from rateslib.fx_volatility import FXDeltaVolSmile, FXDeltaVolSurface
   from datetime import datetime as dt
   from matplotlib import pyplot as plt
   eur = Curve({dt(2009, 5, 3): 1.0, dt(2011, 5, 10): 1.0})
   usd = Curve({dt(2009, 5, 3): 1.0, dt(2011, 5, 10): 1.0})
   fxf = FXForwards(
       fx_rates=FXRates({"eurusd": 1.34664}, settlement=dt(2009, 5, 5)),
       fx_curves={"eureur": eur, "usdusd": usd, "eurusd": eur},
   )
   solver = Solver(
       curves=[eur, usd],
       instruments=[
           Value(dt(2009, 5, 4), curves=eur, metric="cc_zero_rate"),
           Value(dt(2009, 5, 4), curves=usd, metric="cc_zero_rate")
       ],
       s=[1.00, 0.4759550366220911],
       fx=fxf,
   )
   surface = FXDeltaVolSurface(
       eval_date=dt(2009, 5, 3),
       delta_indexes=[0.25, 0.5, 0.75],
       expiries=[dt(2010, 5, 3), dt(2011, 5, 3)],
       node_values=np.ones((2, 3))* 18.0,
       delta_type="forward",
       id="surface",
   )
   fx_args_0 = dict(
       pair="eurusd",
       curves=[None, eur, None, usd],
       expiry=dt(2010, 5, 3),
       delta_type="spot",
       vol="surface",
   )
   fx_args_1 = dict(
       pair="eurusd",
       curves=[None, eur, None, usd],
       expiry=dt(2011, 5, 3),
       delta_type="forward",
       vol="surface",
   )

   solver = Solver(
       surfaces=[surface],
       instruments=[
           FXStraddle(strike="atm_delta", **fx_args_0),
           FXBrokerFly(strike=(("-25d", "25d"), "atm_delta"), **fx_args_0),
           FXRiskReversal(strike=("-25d", "25d"), **fx_args_0),
           FXStraddle(strike="atm_delta", **fx_args_1),
           FXBrokerFly(strike=(("-25d", "25d"), "atm_delta"), **fx_args_1),
           FXRiskReversal(strike=("-25d", "25d"), **fx_args_1),
       ],
       s=[18.25, 0.95, -0.6, 17.677, 0.85, -0.562],
       fx=fxf,
   )
   sm12 = surface.smiles[0]
   sm18 = surface.get_smile(dt(2010, 11, 3))
   sm24 = surface.smiles[1]
   fig, ax, lines = sm12.plot(comparators=[sm18, sm24], labels=["1y", "18m", "2y"])
   plt.show()
   plt.close()

Alternative a 3D surface plot can also be shown.

.. ipython:: python

   surface.plot()

.. plot::

   from rateslib.curves import Curve
   from rateslib.solver import Solver
   from rateslib.fx import FXForwards, FXRates
   from rateslib.instruments import FXStraddle, FXRiskReversal, FXBrokerFly, Value
   from rateslib.fx_volatility import FXDeltaVolSmile, FXDeltaVolSurface
   from datetime import datetime as dt
   from matplotlib import pyplot as plt
   eur = Curve({dt(2009, 5, 3): 1.0, dt(2011, 5, 10): 1.0})
   usd = Curve({dt(2009, 5, 3): 1.0, dt(2011, 5, 10): 1.0})
   fxf = FXForwards(
       fx_rates=FXRates({"eurusd": 1.34664}, settlement=dt(2009, 5, 5)),
       fx_curves={"eureur": eur, "usdusd": usd, "eurusd": eur},
   )
   solver = Solver(
       curves=[eur, usd],
       instruments=[
           Value(dt(2009, 5, 4), curves=eur, metric="cc_zero_rate"),
           Value(dt(2009, 5, 4), curves=usd, metric="cc_zero_rate")
       ],
       s=[1.00, 0.4759550366220911],
       fx=fxf,
   )
   surface = FXDeltaVolSurface(
       eval_date=dt(2009, 5, 3),
       delta_indexes=[0.25, 0.5, 0.75],
       expiries=[dt(2010, 5, 3), dt(2011, 5, 3)],
       node_values=np.ones((2, 3))* 18.0,
       delta_type="forward",
       id="surface",
   )
   fx_args_0 = dict(
       pair="eurusd",
       curves=[None, eur, None, usd],
       expiry=dt(2010, 5, 3),
       delta_type="spot",
       vol="surface",
   )
   fx_args_1 = dict(
       pair="eurusd",
       curves=[None, eur, None, usd],
       expiry=dt(2011, 5, 3),
       delta_type="forward",
       vol="surface",
   )

   solver = Solver(
       surfaces=[surface],
       instruments=[
           FXStraddle(strike="atm_delta", **fx_args_0),
           FXBrokerFly(strike=(("-25d", "25d"), "atm_delta"), **fx_args_0),
           FXRiskReversal(strike=("-25d", "25d"), **fx_args_0),
           FXStraddle(strike="atm_delta", **fx_args_1),
           FXBrokerFly(strike=(("-25d", "25d"), "atm_delta"), **fx_args_1),
           FXRiskReversal(strike=("-25d", "25d"), **fx_args_1),
       ],
       s=[18.25, 0.95, -0.6, 17.677, 0.85, -0.562],
       fx=fxf,
   )
   fig, ax, lines = surface.plot()
   plt.show()
   plt.close()
