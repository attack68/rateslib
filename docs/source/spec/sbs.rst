.. _spec-sbs:

.. ipython:: python
   :suppress:

   from rateslib import *

****
SBS
****

EUR
********

.. _spec-eur-sbs36:

3s6s
----------

.. ipython:: python

   defaults.spec["eur_sbs36"]
   SBS(dt(2000, 1, 1), "10y", spec="eur_sbs36").kwargs


NOK
*****

.. _spec-nok-sbs36:

3s6s
-----

.. ipython:: python

   defaults.spec["nok_sbs36"]
   SBS(dt(2000, 1, 1), "10y", spec="nok_sbs36").kwargs


AUD
****

.. _spec-aud-sbs36:

3s6s
-----

.. ipython:: python

   defaults.spec["aud_sbs36"]
   SBS(dt(2000, 1, 1), "10y", spec="aud_sbs36").kwargs

.. _spec-aud-sbs31:

3s1s
-----

.. ipython:: python

   defaults.spec["aud_sbs31"]
   SBS(dt(2000, 1, 1), "10y", spec="aud_sbs31").kwargs

NZD
****

.. _spec-nzd-sbs36:

3s6s
-----

.. ipython:: python

   defaults.spec["nzd_sbs36"]
   SBS(dt(2000, 1, 1), "10y", spec="nzd_sbs36").kwargs

.. _spec-nzd-sbs31:

3s1s
-----

.. ipython:: python

   defaults.spec["nzd_sbs31"]
   SBS(dt(2000, 1, 1), "10y", spec="nzd_sbs31").kwargs