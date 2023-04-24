.. _risk-toc-doc:

*****************
Risk Sensitivity
*****************

Pricing is one aspect of securities and derivatives trading. **Risk** is
another equally, if not more, important feature. The mechanisms and
API framework constructed in *rateslib* are, in fact, designed around, and
primarily to suit, risk sensitivity calculation.

The goal is to produce accurate :ref:`Delta<delta-doc>` risks, in both
calibrating instrument and FX space, as well as :ref:`Cross-Gamma<gamma-doc>`
risks, which combined will produce a reliably accurate PnL estimate of a
portfolio given market movements.

Risk sensitivities are efficiently calculated through *rateslib's* own
automatic differentiation toolset.

.. toctree::
    :maxdepth: 0
    :titlesonly:

    j_delta.rst
    j_gamma.rst