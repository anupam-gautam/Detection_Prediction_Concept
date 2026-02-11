import unittest
import sys
import os
import shutil

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fusion.engine import InferenceEngine

class TestInferenceAnalytics(unittest.TestCase):
    def test_export(self):
        engine = InferenceEngine(smoothing_window=1)
        
        # Simulate some data
        engine.update(False, False, 0.0) # Active
        engine.update(True, True, 10.0) # Active (Non-Interactive)
        engine.update(False, False, 10.0) # Inactive
        
        raw_file = "test_raw.txt"
        report_file = "test_report.txt"
        
        engine.export_session_data(raw_file, report_file)
        
        self.assertTrue(os.path.exists(raw_file))
        self.assertTrue(os.path.exists(report_file))
        
        with open(raw_file, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 4) # Header + 3 entries
            
        with open(report_file, 'r') as f:
            content = f.read()
            self.assertIn("SESSION ANALYTICS REPORT", content)
            self.assertIn("Active", content)
            
        # Cleanup
        if os.path.exists(raw_file): os.remove(raw_file)
        if os.path.exists(report_file): os.remove(report_file)

if __name__ == '__main__':
    unittest.main()
