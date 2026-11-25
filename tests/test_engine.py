import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]))
from thermal_engine import Settings, run_all, plant_initial, plant_step, h_specific, t_from_h, heat_current
class EngineTests(unittest.TestCase):
 def test_enthalpy_inverse(self):
  x=np.array([10.,30.,38.,39.5,43.,55.])
  self.assertTrue(np.allclose(t_from_h(h_specific(x)),x,atol=1e-8))
 def test_idle_equilibrium(self):
  s=Settings(profile="单次放电后静置")
  T,H,Ta=plant_initial(s)
  T1,H1,Ta1=plant_step(np.full(12,s.ambient),H,np.full(12,s.ambient),s,7200,60,.5,.75,"uniform",0)
  self.assertTrue(np.allclose(T1,s.ambient))
  self.assertTrue(np.allclose(Ta1,s.ambient))
 def test_charge_discharge_schedule(self):
  s=Settings(rate=1.5)
  self.assertEqual(heat_current(0,s),420)
  self.assertEqual(heat_current(1/1.5*3600+1,s),-210)
 def test_all_strategies_finite_and_energy_monotonic(self):
  s=Settings(duration_h=2)
  out=run_all(s)
  self.assertEqual(set(out),{"固定送风","固定阈值","双预测MPC"})
  for d in out.values():
   self.assertTrue(np.isfinite(d.select_dtypes("number").to_numpy()).all())
   self.assertTrue((d.total_energy_kWh.diff().fillna(0)>=-1e-10).all())
   self.assertTrue((d.dTmax_C>=0).all())
   self.assertGreater(d.Tmax_C.min(),0)
 def test_48h_default(self):
  s=Settings(duration_h=48)
  out=run_all(s)
  for d in out.values():
   self.assertEqual(len(d),2881)
   self.assertTrue(np.isfinite(d.select_dtypes("number").to_numpy()).all())
   self.assertGreaterEqual(d.Tmax_C.min(),10)
   self.assertGreaterEqual(d.PCM_T_C.min(),10)
   self.assertGreaterEqual(d.total_energy_kWh.iloc[-1],0)
if __name__=="__main__":unittest.main()
