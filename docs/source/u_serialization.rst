.. _serialization-doc:

.. ipython:: python
   :suppress:

   from rateslib import from_json
   from rateslib.dual import Dual, Dual2

****************************
Serialization
****************************

*Rateslib* is an object oriented library requiring dynamic associations to
create pricing and risk functionality. The objective of serialization and
deserialization is to convert *rateslib's* objects into a form that
can be persisted or transported, with a view towards being able to recapture
those dynamic associations.

Ultimately this means that data could be stored in a database and retrieved,
or it could be transmitted over the web from a server to a client based browser and
reconstructed.

.. warning::

   This feature is currently experimental, and in production.

JSON
*******

Serialization is a general term. The specific form of serialization that *rateslib*
aims to create is JSON serialization, meaning an object can be deconstructed
into a JSON string and reconstructed from the same string. JSON has pros and cons
associated with its choice. A con being that it is not necessarily the most
efficient form of transfer or storage, but pros being that is is human-readable,
transparent and provides a defined data structure.

The key method used for this purpose is:

.. autosummary::
   rateslib.from_json

Objects in Scope
******************

Dual
------

The automatic differentiation (AD) data types (:class:`~rateslib.dual.Dual` and :class:`~rateslib.dual.Dual2`)
are fundamental data types. These must be serializable to propagate the attribute of serialization to other
objects.

.. ipython:: python

   dual = Dual(3.141, ["x", "y"], [1.0, 0.0])
   dual.to_json()
   from_json(dual.to_json())

.. ipython:: python

   dual2 = Dual2(3.141, ["x", "y"], [1.0, 0.0], [])
   dual2.to_json()
   from_json(dual2.to_json())
