.. _instruments-toc-doc:

************
Instruments
************

:ref:`Instruments<instruments-doc>` in ``rateslib`` are sequentially constructed.

- First :ref:`Periods<periods-doc>` are defined in the ``rateslib.periods`` module.
- Secondly :ref:`Legs<legs-doc>` are defined in the ``rateslib.legs`` module and these
  combine and control a list of organised :ref:`Periods<periods-doc>`.
- Finally :ref:`Instruments<instruments-doc>` are defined in the
  ``rateslib.instruments`` module and these combine and control one or two
  :ref:`Legs<legs-doc>`.

It is recommended to review the documentation in the above order, since the
composited objects are more explicit in their documentation of each parameter.

Users are expected to rarely use :ref:`Periods<periods-doc>` or
:ref:`Legs<legs-doc>` directly but they are exposed in the public API in order
to construct custom objects.

.. toctree::
    :hidden:
    :maxdepth: 2

    d_periods.rst
    d_legs.rst
    d_instruments.rst