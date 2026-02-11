import unittest
import sys
import os
import time

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fusion.engine import InferenceEngine

class TestInferenceEngine(unittest.TestCase):
    def setUp(self):
        self.engine = InferenceEngine(smoothing_window=1) # Disable smoothing for logic tests

    def test_input_dominance(self):
        """Test Scenario 1: High Input Activity -> Active Usage"""
        # Active input (0.0s idle), No Face, No Gaze
        result = self.engine.update(is_looking_at_screen=False, is_face_present=False, input_idle_time=0.0)
        
        self.assertEqual(result["usage_state"], "Active")
        self.assertEqual(result["attention_mode"], "Interactive")
        self.assertGreater(result["active_usage_probability"], 0.8)
        self.assertGreater(result["confidence_score"], 0.8)

    def test_visual_validation_active(self):
        """Test Scenario 2: No Input, Face + Gaze -> Active (Non-Interactive)"""
        # No input (10s idle), Face Present, Gaze On Screen
        result = self.engine.update(is_looking_at_screen=True, is_face_present=True, input_idle_time=10.0)
        
        self.assertEqual(result["usage_state"], "Active")
        self.assertEqual(result["attention_mode"], "Non-Interactive")
        self.assertGreater(result["active_usage_probability"], 0.6)

    def test_passive_state(self):
        """Test Scenario 3: No Input, Face Present, No Gaze -> Passive"""
        # No input, Face Present, Looking Away
        result = self.engine.update(is_looking_at_screen=False, is_face_present=True, input_idle_time=10.0)
        
        self.assertEqual(result["usage_state"], "Passive")
        self.assertEqual(result["attention_mode"], "Non-Interactive") # Or Distracted if I changed it? Re-check logic.
        # In code I put "Non-Interactive" for attention_mode in this case too.
        
    def test_inactive_state(self):
        """Test Scenario 4: No Signals -> Inactive"""
        # No input, No Face
        result = self.engine.update(is_looking_at_screen=False, is_face_present=False, input_idle_time=10.0)
        
        self.assertEqual(result["usage_state"], "Inactive")
        self.assertEqual(result["attention_mode"], "Absent")
        self.assertLess(result["active_usage_probability"], 0.2)
        
    def test_metrics_accumulation(self):
        """Test Scenario 5: Metrics Accumulation"""
        # Reset engine
        self.engine = InferenceEngine(smoothing_window=1)
        
        # Simulate 1 second of Active
        # We need to sleep to get real dt? Or mock time?
        # InferenceEngine uses time.time(). I should mock it or just sleep a tiny bit.
        # Or I can manually inject dt if I modify the class, but simplest is to just call update twice with a sleep.
        
        # Initial call to set last_update_time
        self.engine.update(False, False, 0.0) 
        time.sleep(0.1)
        
        # Second call - should add ~0.1s to Active
        result = self.engine.update(False, False, 0.0) # Active
        
        active_duration = result["metrics"]["Active"]
        self.assertGreater(active_duration, 0.0)
        self.assertLess(active_duration, 0.2)

if __name__ == '__main__':
    unittest.main()
