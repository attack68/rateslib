.. _spec-fra:

.. ipython:: python
   :suppress:

   from rateslib import *

****
FRA
****

EUR
********

.. _spec-eur-fra3:

3m
---

.. ipython:: python

   defaults.spec["eur_fra3"]
   FRA(dt(2000, 1, 1), spec="eur_fra3").kwargs

.. _spec-eur-fra6:

6m
-----

.. ipython:: python

   defaults.spec["eur_fra6"]
   FRA(dt(2000, 1, 1), spec="eur_fra6").kwargs

.. _spec-eur-fra1:

1m
-----

.. ipython:: python

   defaults.spec["eur_fra1"]
   FRA(dt(2000, 1, 1), spec="eur_fra1").kwargs

SEK
********

.. _spec-sek-fra3:

3m
---

.. ipython:: python

   defaults.spec["sek_fra3"]
   FRA(dt(2000, 1, 1), spec="sek_fra3").kwargs

NOK
********

.. _spec-nok-fra3:

3m
---

.. ipython:: python

   defaults.spec["nok_fra3"]
   FRA(dt(2000, 1, 1), spec="nok_fra3").kwargs

.. _spec-nok-fra6:

6m
---

.. ipython:: python

   defaults.spec["nok_fra6"]
   FRA(dt(2000, 1, 1), spec="nok_fra6").kwargs