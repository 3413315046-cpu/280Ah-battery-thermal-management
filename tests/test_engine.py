import sys
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]))
from thermal_engine import Settings, run_all, plant_initial, plant_step, metrics_state, heat_current, h_specific, t_from_h, FAN_DB

class TestThermalEngine(unittest.TestCase):
 def test_enthalpy_inverse(self):
  t=np.array([10.,30.,38.,39.5,43.,55.,75.])
  self.assertTrue(np.allclose(t_from_h(h_specific(t)),t,atol=1e-9))

 def test_step_energy_bounded_and_no_air_conditioner(self):
  s=Settings()
  T,H=plant_initial(s)
  T1,H1=plant_step(T,H,s,0,60,0,0.75,'uniform')
  self.assertTrue(np.all(T1>=s.ambient))
  self.assertTrue(np.all(H1>=H))
  # If no generation and thermal states at ambient, pure passive step is stationary.
  s0=replace(s, profile='单次放电后静置')
  T2,H2=plant_step(np.full(12,s0.ambient),H,s0,5*3600,60,0.5,.75,'uniform')
  self.assertTrue(np.allclose(T2,s0.ambient))
  self.assertTrue(np.allclose(H2,H))

 def test_soc_consistent_charge_discharge_timing(self):
  s=Settings(rate=1.5,profile='完整充放电循环')
  self.assertEqual(heat_current(0,s),420)
  self.assertEqual(heat_current(1/1.5*3600+1,s),-210)
  self.assertEqual(heat_current((1/1.5+2/1.5+.01)*3600,s),0)

 def test_three_cases_same_shapes_and_finite(self):
  s=Settings(duration_h=2)
  cases=run_all(s)
  self.assertEqual(len(cases),3)
  lengths={len(d) for d in cases.values()}
  self.assertEqual(lengths,{121})
  for k,df in cases.items():
   self.assertTrue(np.isfinite(df.select_dtypes(include='number').to_numpy()).all())
   self.assertGreater(df.Tmax_C.min(),0)
   self.assertGreater(df.PCM_T_C.min(),0)
   self.assertTrue((df.energy_kWh.diff().fillna(0)>-1e-8).all())
   self.assertTrue((df.dTmax_C>=0).all())
   self.assertTrue((df.PCM_liquid.between(0,1)).all())
   self.assertTrue((df.valve_energy_kWh>=0).all())
  self.assertAlmostEqual(cases['固定送风'].fan_speed.iloc[0],s.fixed_speed)
  self.assertAlmostEqual(cases['固定阈值'].fan_speed.iloc[0],0)
  self.assertAlmostEqual(cases['双预测MPC'].fan_speed.iloc[0],0)

 def test_long_default_and_hot_case(self):
  for s in [Settings(duration_h=48),Settings(duration_h=12,ambient=35,initial=35,rate=1.5)]:
   cases=run_all(s)
   for label,df in cases.items():
    self.assertTrue(np.isfinite(df.Tmax_C).all())
    self.assertGreaterEqual(df.Tmax_C.min(),10)
    self.assertGreaterEqual(df.PCM_T_C.min(),10)
    self.assertTrue((df.PCM_remaining_MJ>=-1e-6).all())
    self.assertGreaterEqual(df.energy_kWh.iloc[-1],0)

if __name__=='__main__': unittest.main(verbosity=2)
