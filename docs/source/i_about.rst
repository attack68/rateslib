.. _about-doc:

******
About
******

Release Notes
**************

:ref:`See here for release notes.<whatsnew-doc>`.

The History and Context of Rateslib
************************************

*Rateslib* BETA was first released in April 2023.

The foundations of *rateslib* is really the code library
`Book IRDS3 <https://github.com/attack68/book_irds3>`_, which laid down
basic principles and was a sandbox code environment for the
publication `Pricing and Trading Interest Rate Derivatives: A Practical Guide to Swaps <https://www.amazon.com/Pricing-Trading-Interest-Rate-Derivatives/dp/0995455538>`_.
Some of the code and algorithms also date back to the author's
time trading IRSs as a market-maker between 2006 and 2017. The algorithms and mathematical
code developments of *rateslib*
are all characterised and explained in
`Coding Interest Rates: FX, Swaps and Bonds <https://www.amazon.com/dp/0995455554>`_.

.. container:: twocol

   .. container:: leftside40

      .. image:: _static/thumb_coding_3.png
         :alt: Coding Interest Rates: FX, Swaps and Bonds
         :target: https://www.amazon.com/dp/0995455554
         :width: 145
         :align: center

   .. container:: rightside60

      .. image:: _static/thumb_ptirds3.png
         :alt: Pricing and Trading Interest Rate Derivatives
         :target: https://www.amazon.com/Pricing-Trading-Interest-Rate-Derivatives/dp/0995455538
         :width: 145
         :align: center

.. raw:: html

   <div class="clear"></div>


.. _pillars-doc:

Five Pillars of Rateslib's Design Philosophy
*********************************************

1) Maximise flexibility : Minimise user input
-----------------------------------------------------------------------

This is a user interface (UI) objective. *Rateslib* aims to
make technical and complex financial instrument analysis easily accessible and
consumable. This philosophy has shaped the the entire design and API architecture of *rateslib*.

For example, this library will not add esoteric or complex algorithms for valuing a
financial instrument. Although doing so satisfies the philosophy of maximising
flexibility, every addition adds documentation obfuscation, uncertainty
about market practice and likely extends parameter numbers. It breaks the
philosophy.

On the other hand this library will allow various *Curve* construction interpolation
algorithms because these are market standard and their parametrisation is simple and
well documented, and these have considerable impact on all users. It satisfies the
philosophy.

2) Prioritise sensitivities above valuation
-----------------------------------------------------

This is a functionality objective. Risk sensitivities are harder to calculate than
valuation. To calculate risk it is
necessary to be able to calculate value. To calculate value it is not necessary
to calculate risk. Therefore making design choices around calculating risk avoids
the problem of building *Instruments* in a value sense and 'hacking' risk sensitivities
later.

This philosophy indirectly drives performant solutions. It is also the reason that
the library constructed and implemented its own automatic differentiation (AD)
toolset to properly label and be able to pass around derivatives from any object
or scenario to another. It also means that *npv*, *delta* and *gamma* have the
same arguments signature, which promotes usability.

3) Unify asset classes within a single UI
-------------------------------------------------------

This defines scope. *Rateslib* aims to unify interest rates, FX and inflation
in its first version.
The ambition is to incorporate volatility products into version two. Within
this unification we must include the commonly traded instruments within
each of these classes.


4) Achieve scalable performance
--------------------------------------------

This is a functionality objective.
Version one of *rateslib* is pure Python. The performance constraints this places are
restrictive. However, every method *rateslib* offers must be capable of producing
results in practical time. A fixed income library written in Python cannot achieve
what *rateslib* achieves without AD. Additionally many manual optimisations are
implemented and are documented.

A wider article about
`performance in rateslib <https://www.linkedin.com/pulse/rateslib-performance-1000-irs-rateslib>`_
is available following the link.

5) Be transparent and validate as default
--------------------------------------------

This is a community objective.
No algorithm, optimisation or approximation is added without being documented
in **Coding Interest Rates**. The code coverage of the library strives to be 100%.
This API documentation should be exhaustive with demonstrative examples.
Where official sources (ISDA documentation, academic papers, issuer specifications)
are available their examples should be used as unit tests within *rateslib*.

A good
example for these are the UK DMO's Gilt calculations which are replicated exactly
by *rateslib*. Another example is the
`replication of Norges Bank NOWA calculator <https://www.linkedin.com/pulse/rateslib-vs-norges-bank-nowa-calculator-rateslib>`_

About the Author
****************
An extended bio is available on Amazon `here <https://www.amazon.com/J-H-M-Darbyshire/e/B0725PW9HY>`_.
It is also possible to connect via LinkedIn `here <https://www.linkedin.com/in/hamish-darbyshire/>`_.
