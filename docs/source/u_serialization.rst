.. _serialization-doc:

****************************
Serialization
****************************

*Rateslib* is an object oriented library requiring dynamic associations to
create pricing and risk functionality. The objective of serialization and
deserialization is to convert *rateslib's* objects into a form that
can be persisted or transported, with a view towards being able to recapture
those dynamic associations (i.e. saved and loaded from a database, or sent between
client and server over a network).

.. warning::

   This feature is currently experimental, and in development. This page documents the
   objects that have, to date, had focused development for the two forms of serialization
   *rateslib* intends to offer. Other objects in *rateslib* may well have serialization methods
   in place but the stability and robustness of those may not have been properly tested.

Forms
*******

Serialization is a general term. The specific forms of serialization that *rateslib*
aims to create is JSON serialization and Python pickling.

.. tabs::

   .. group-tab:: JSON

      JSON serialization converts an object to a string and reconstructs it from that same string.
      The advantage of JSON is that it is human readable (and often editable) whilst a
      disadvantage is that it likely to be a lesser efficient form of transfer or storage.
      It is considered a safe form of serialization.

      .. ipython:: python

         from rateslib import from_json

      .. autosummary::

         rateslib.serialization.from_json

   .. group-tab:: Python Pickle

      Pickling is a native conversion of binary objects to byte arrays. That same array can be
      reconstituted into a Python object. This is typically very efficient but the resultant
      arrays of data are not human-readable (or editable). This is considered an unsafe form of
      serialization due to the arbitrary nature of code execution upon deserialization.

      Pickling of data is often performed internally by Python's multiprocessing module if it
      needs to share data across threads.

      .. ipython:: python

         from pickle import dumps, loads


Objects in Scope
******************

We aim to keep the objects in scope synchronized with availability for both *json* and
*pickle* serialization for the stated objects below.

.. ipython:: python
   :suppress:

   from rateslib.dual import Dual, Dual2, Variable
   from rateslib.scheduling import Cal, UnionCal, NamedCal
   from rateslib.fx import FXRates
   from rateslib.splines import PPSplineF64
   from datetime import datetime as dt
   from rateslib.scheduling import Imm, Adjuster, RollDay, Frequency, Schedule, StubInference

.. tabs::

   .. group-tab:: JSON

      .. ipython:: python
         :suppress:

         # dual
         dual = Dual(3.141, ["x", "y"], [1.0, 0.0])
         dual2 = Dual2(3.141, ["x", "y"], [1.0, 0.0], [])
         variable = Variable(3.141, ["z"])

         # scheduling
         cal = Cal([], [5,6])
         union_cal = UnionCal([cal], [])
         named_cal = NamedCal("bus")
         adjuster = Adjuster.Following()
         imm = Imm.Wed3
         roll = RollDay.Day(31)
         frequency = Frequency.Months(4, roll)
         stub_inference = StubInference.LongFront
         schedule = Schedule(dt(2000, 1, 1), dt(2001, 1, 1), "S")

         # splines
         ppspline_f64 = PPSplineF64(k=4, t=[0,0,0,0,4,4,4,4], c=[0.3, 0.2, 0.6, 0.2])
         ppspline_dual = PPSplineDual(3, [0,0,0,1,1,1], [Dual(0.1, [], []), Dual(0.2, [], []), Dual(0.3, [], [])])
         ppspline_dual2 = PPSplineDual2(3, [0,0,0,1,1,1], [Dual2(0.1, [], [], []), Dual2(0.2, [], [], []), Dual2(0.3, [], [], [])])

         # fx_rates
         fxr = FXRates({"eurusd": 1.15}, settlement=dt(2000, 1, 1))

      **Serializing**

      .. ipython:: python

         # --------------dual ---------------------
         dual.to_json()             # Dual
         dual2.to_json()            # Dual2
         variable.to_json()         # Variable
         # ---------- scheduling ------------------
         cal.to_json()              # Cal
         union_cal.to_json()        # UnionCal
         named_cal.to_json()        # NamedCal
         adjuster.to_json()         # Adjuster
         imm.to_json()              # Imm
         roll.to_json()             # RollDay
         frequency.to_json()        # Frequency
         stub_inference.to_json()   # StubInference
         schedule.to_json()         # Schedule
         # ------------ splines --------------------
         ppspline_f64.to_json()     # PPSplineF64
         ppspline_dual.to_json()    # PPSplineDual
         ppspline_dual2.to_json()   # PPSplineDual2
         # ------------ fx_rates -------------------
         fxr.to_json()              # FXRates

      **Deserializing**

      .. ipython:: python
         :suppress:

         dual_json = dual.to_json()             # Dual
         dual2_json = dual2.to_json()            # Dual2
         variable_json = variable.to_json()         # Variable
         cal_json = cal.to_json()              # Cal
         union_cal_json = union_cal.to_json()        # UnionCal
         named_cal_json = named_cal.to_json()        # NamedCal
         adjuster_json = adjuster.to_json()         # Adjuster
         imm_json = imm.to_json()              # Imm
         roll_json = roll.to_json()             # RollDay
         frequency_json = frequency.to_json()        # Frequency
         stub_inference_json = stub_inference.to_json()   # StubInference
         schedule_json = schedule.to_json()         # Schedule
         ppspline_f64_json = ppspline_f64.to_json()         # PPSplineF64
         ppspline_dual_json = ppspline_dual.to_json()         # PPSplineF64
         ppspline_dual2_json = ppspline_dual2.to_json()         # PPSplineF64
         fxr_json = fxr.to_json()              # FXRates

      .. ipython:: python

         # --------------dual ---------------------
         from_json(dual_json)
         from_json(dual2_json)
         from_json(variable_json)
         # ---------- scheduling ------------------
         from_json(cal_json)
         from_json(union_cal_json)
         from_json(named_cal_json)
         from_json(adjuster_json)
         from_json(imm_json)
         from_json(roll_json)
         from_json(frequency_json)
         from_json(stub_inference_json)
         from_json(schedule_json)
         # ------------ splines --------------------
         from_json(ppspline_f64_json)
         from_json(ppspline_dual_json)
         from_json(ppspline_dual2_json)
         # ------------ fx_rates -------------------
         from_json(fxr_json)

   .. group-tab:: Python Pickle

      **Serializing**

      .. ipython:: python

         # --------------dual ---------------------
         dumps(dual)                # Dual
         dumps(dual2)               # Dual2
         dumps(variable)            # Variable
         # ---------- scheduling ------------------
         dumps(cal)                 # Cal
         dumps(union_cal)           # UnionCal
         dumps(named_cal)           # NamedCal
         dumps(adjuster)            # Adjuster
         dumps(imm)                 # Imm
         dumps(roll)                # RollDay
         dumps(frequency)           # Frequency
         dumps(stub_inference)      # StubInference
         dumps(schedule)            # Schedule
         # ------------ splines --------------------
         dumps(ppspline_f64)        # PPSplineF64
         dumps(ppspline_dual)       # PPSplineDual
         dumps(ppspline_dual2)      # PPSplineDual2
         # ------------ fx_rates -------------------
         dumps(fxr)                 # FXRates

      .. ipython:: python
         :suppress:

         # --------------dual ---------------------
         dual_bytes = dumps(dual)                # Dual
         dual2_bytes = dumps(dual2)               # Dual2
         variable_bytes = dumps(variable)            # Variable
         # ---------- scheduling ------------------
         cal_bytes = dumps(cal)                 # Cal
         union_cal_bytes = dumps(union_cal)           # UnionCal
         named_cal_bytes = dumps(named_cal)           # NamedCal
         adjuster_bytes = dumps(adjuster)            # Adjuster
         imm_bytes = dumps(imm)                 # Imm
         roll_bytes = dumps(roll)                # RollDay
         frequency_bytes = dumps(frequency)           # Frequency
         stub_inference_bytes = dumps(stub_inference)      # StubInference
         schedule_bytes = dumps(schedule)            # Schedule
         # ------------ splines --------------------
         ppspline_f64_bytes = dumps(ppspline_f64)            # PPSplineF64
         ppspline_dual_bytes = dumps(ppspline_dual)            # PPSplineF64
         ppspline_dual2_bytes = dumps(ppspline_dual2)            # PPSplineF64
         # ------------ fx_rates -------------------
         fxr_bytes = dumps(fxr)                 # FXRates

      **Deserializing**

      .. ipython:: python

         # --------------dual ---------------------
         loads(dual_bytes)
         loads(dual2_bytes)
         loads(variable_bytes)
         # ---------- scheduling ------------------
         loads(cal_bytes)
         loads(union_cal_bytes)
         loads(named_cal_bytes)
         loads(adjuster_bytes)
         loads(imm_bytes)
         loads(roll_bytes)
         loads(frequency_bytes)
         loads(stub_inference_bytes)
         loads(schedule_bytes)
         # ------------ splines --------------------
         loads(ppspline_f64_bytes)
         loads(ppspline_dual_bytes)
         loads(ppspline_dual2_bytes)
         # ------------ fx_rates -------------------
         loads(fxr_bytes)
