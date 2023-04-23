.. _about-doc:

******
About
******

About Rateslib
******************

*Rateslib* BETA was first released in April 2023.

The foundations of *rateslib* is really the code library
`Book IRDS3 <https://github.com/attack68/book_irds3>`_, which laid down
basic principles and was a sandbox code environment for the
publication *Pricing and Trading Interest Rate Derivatives: A Practical Guide to Swaps
(2022, 3rd Edition)*. Some of the code and algorithms also date back to the authors
time trading IRSs as a market-maker between 2006 and 2017.

.. image:: _static/thumb_ptirds3.png
  :alt: Pricing and Trading Interest Rate Derivatives
  :target: https://www.amazon.com/Pricing-Trading-Interest-Rate-Derivatives/dp/0995455538
  :width: 92

The algorithms and mathematical work of *rateslib* is expected to be released in
early 2024 under the working title of *Coding Interest Rates: FX, Swaps and Bonds*

.. image:: _static/thumb_coding1.png
  :alt: Coding Interest Rates: FX, Swaps and Bonds
  :target: https://www.amazon.com/Pricing-Trading-Interest-Rate-Derivatives/dp/0995455538
  :width: 92

About the Author
****************
TBD.

Development
*******************

As a new library the future development of *rateslib* is open to many avenues.
Some possibilities are listed below. The author is very interested in any feedback
and this can be given on the public **Issues** board at the project github
repository: `Rateslib Project <https://github.com/attack68/rateslib>`_, or by direct
email contact through **rateslib@gmail.com**.

.. list-table::
   :widths: 20 35 35 10
   :header-rows: 1


   * - Feature
     - Description
     - Consideration
     - Timeframe
   * - Inflation Bonds
     - Adding the pricing nuances for inflation bond markets.
     - Very likely
     - By end 2023
   * - Inflation Swaps
     - Adding the pricing nuances for inflation swaps.
     - Very likely
     - By end 2023
   * - Vanilla Swaptions
     - Adding the instruments priced by a volatility input.
     - Likely
     - By mid 2024
   * - Vanilla FX options
     - Adding option instruments and benchmark trades such as risk-reversals.
     - Possible, depending on ease of extension from swaptions.
     - By end 2024
   * - SABR model for options
     - Adding the parameters to construct SABR vol surfaces/ cuves.
     - Possible, with dependencies to other developments.
     - By end 2024
   * - Optimization of code
     - Using C extensions or re-writing certain blocks to improve performance.
     - Unlikely, depending upon community adoption and contributions.
     - no ETA
   * - AD backend
     - Changing the AD implementation to another 3rd party (JAX, PyAudi)
     - Very unlikely, maturity of those libraries must increase and the performance
       improvements must be sufficient to warrant such a large codebase change.
     - no ETA
   * - JSON facility
     - Designing objects, as well as object oriented associations to be passed from
       server to client and vice versa to operate a cloud solution.
     - Possible, due to the author's interest in the topic, but not imminent.
     - no ETA
   * - Excel interaction
     - Allowing *rateslib* to be accessed via Excel in a structured way.
     - Possible, but not imminent, due to the unscoped problem.
     - no ETA
   * - Datafeeds
     - Allowing *rateslib* to access and consume data in a streaming environment
       working with other data providers APIs.
     - Unlikely, due to the subjectivity of every data consumer.
     - no ETA
