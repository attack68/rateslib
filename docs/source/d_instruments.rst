.. _instruments-doc:

.. ipython:: python
   :suppress:

   from rateslib.instruments import *
   from datetime import datetime as dt

**********************
Instruments
**********************

The following groups outline the different :ref:`Instrument<instruments-doc>` types
offered by ``rateslib``.

Common methods which all :ref:`Instrument<instruments-doc>` strive to possess are
``rate()``, ``npv()``, ``analytic_delta``, ``delta()``, ``gamma()`` and
``cashflows()``.

.. toctree::
    :maxdepth: 0
    :titlesonly:

    e_securities.rst
    e_singlecurrency.rst
    e_multicurrency.rst
    e_combinations.rst