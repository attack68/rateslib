.. to_json
   ==========

.. .. currentmodule:: rateslib.dual

.. py:method:: Dual2.to_json()

   Create a JSON string representation of the object.

   :rtype: str

   .. rubric:: Notes

   Dual type variables that are converted to JSON and reconstructed do not preserve the rust ARC pointer which is
   a performance consideration when defining efficient operations.

   .. rubric:: Examples

   .. ipython:: python

      from rateslib import Dual2, from_json

      x1 = Dual2(1.0, ["x"], [], [])
      x1.to_json()
      x2 = from_json(x1.to_json())
      x1.ptr_eq(x2)
