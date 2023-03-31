.. _instruments-toc-doc:

************
Instruments
************

The complexity of building fully featured, financial instruments in a library
with a user friendly API is typically due to the large numbers of calibrating
parameters and combinations in which they can occur. To maintain complete
flexibility whilst providing a consistent API the components are segmented into
basic constructing objects, which are outlined below.

Users are expected to rarely use :ref:`Periods<periods-doc>` or
:ref:`Legs<legs-doc>` directly but their documentation is important for
object and calculation transparency.

.. toctree::
    :maxdepth: 2

    d_periods.rst
    d_legs.rst
    d_instruments.rst