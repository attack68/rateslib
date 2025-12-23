.. _about-doc:

******
About
******

Release Notes
**************

:ref:`See here for release notes.<whatsnew-doc>`.

The History and Context of Rateslib
************************************

*Rateslib beta* was released in April 2023, *rateslib v1.0* in Feb 2024 and *v2.0* in June 2025.

Modern *rateslib* is historically derived from three parts:

- The code library introduced in `Book IRDS3 <https://github.com/attack68/book_irds3>`_:
  a sandbox code environment for the
  publication `Pricing and Trading Interest Rate Derivatives: A Practical Guide to Swaps <https://www.amazon.com/Pricing-Trading-Interest-Rate-Derivatives/dp/0995455538>`_.
  This code prioritised a pedagogical style with an API as simple and intuitive as possible.
- Code and algorithms privately devised and developed during the author's time trading IRSs
  as a market-maker and quantative developer for institutions such as Barclays and Nordea.
- Collaborations with other quant groups, and commerical licence holders to implement more
  complex and nuanced areas of fixed income and FX pricing into an accessible and scalable UI.

The algorithms and mathematical code developments of *rateslib* are all characterised and
explained in `Coding Interest Rates: FX, Swaps and Bonds <https://www.amazon.com/dp/0995455562>`_.

.. container:: twocol

   .. container:: leftside40

      .. image:: _static/thumb_coding_2_1.png
         :alt: Coding Interest Rates: FX, Swaps and Bonds
         :target: https://www.amazon.com/dp/0995455562
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

For example, this library will not add esoteric and non-standard algorithms for valuing a
financial instrument. Although doing so satisfies the philosophy of maximising
flexibility, every addition adds documentation obfuscation, uncertainty
about market practice and likely extends parameter numbers. It breaks the
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

This defines scope. As of *v3.0* *rateslib* aims to unify interest rates, FX, inflation,
FX vol and IR vol within the same framework and pricing mechanics.

4) Achieve scalable performance
--------------------------------------------

This is a functionality objective. *Rateslib* uses Rust extensions for bottlenecks and aims
to port as much core calculation code to Rust as possible to drive performance.
*Rateslib* also seeks to expand its *serialization* tools for effective persistent storage and
network transfer of data.

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
It is also possible to connect via LinkedIn `at this link <https://www.linkedin.com/in/hamish-darbyshire/>`_.
