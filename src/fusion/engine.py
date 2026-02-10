from collections import deque

class FusionEngine:
    def __init__(self, smoothing_window=5):
        """
        Args:
            smoothing_window (int): Number of frames to keep for smoothing the result.
        """
        self.history = deque(maxlen=smoothing_window)

    def update(self, is_looking_at_screen, is_face_present, input_active):
        """
        Updates the state and returns the fused decision (bool).
        
        Fusion Logic:
            OnScreen = (Gaze On Screen) OR (Face Present AND Input Active)
        
        Args:
            is_looking_at_screen (bool): From VisionDetector (Gaze).
            is_face_present (bool): From VisionDetector.
            input_active (bool): From InputMonitor.
            
        Returns:
            bool: True if the user is considered "On Screen", False otherwise.
        """
        
        # 1. Calculate Instantaneous State
        if is_looking_at_screen:
            current_state = True
        elif is_face_present and input_active:
             current_state = True
        else:
            current_state = False
        
        # 2. Add to history
        self.history.append(current_state)
        
        # 3. Smoothing (Majority Vote)
        if len(self.history) == 0:
            return False
            
        true_count = sum(self.history)
        return true_count > (len(self.history) / 2)
