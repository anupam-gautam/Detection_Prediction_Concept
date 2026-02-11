from collections import deque
import time
import math

class InferenceEngine:
    def __init__(self, smoothing_window=5):
        """
        Args:
            smoothing_window (int): Number of frames to keep for smoothing the result.
        """
        self.history = deque(maxlen=smoothing_window)
        self.last_update_time = time.time()
        
        # Internal state
        self.active_usage_probability = 0.0
        self.usage_state = "Inactive"
        self.confidence_score = 0.0
        self.confidence_level = "Low"
        self.attention_mode = "Absent"
        self.reasoning_summary = "Initializing..."

        # Metrics Accumulation
        self.state_durations = {
            "Active": 0.0,
            "Passive": 0.0,
            "Inactive": 0.0
        }
        self.session_log = [] # List of tuples/dicts: (timestamp, state, confidence, etc.)

    def _normalize_input(self, idle_time):
        """
        Normalize input activity based on idle time.
        0s idle -> 1.0 score
        5s idle -> ~0.0 score (exponential decay)
        """
        if idle_time < 0.1: return 1.0
        # Decay: score = exp(-lambda * t)
        # We want score ~0.1 at 5s -> -2.3 = -lambda * 5 -> lambda ~ 0.46
        return math.exp(-0.5 * idle_time)

    def _normalize_gaze(self, gaze_ratio):
        """
        Normalize gaze ratio (0.0 - 1.0).
        Center is 0.5. We map deviation from 0.5 to a score.
        Closer to center -> High score.
        """
        deviation = abs(gaze_ratio - 0.5)
        # Max deviation is 0.5. 
        # Score = 1.0 - (deviation / 0.5)
        # 0.5 deviation -> 0.0 score
        # 0.0 deviation -> 1.0 score
        return max(0.0, 1.0 - (deviation * 2.0))

    def update(self, is_looking_at_screen, is_face_present, input_idle_time):
        """
        Updates the state and returns the structured inference result.
        
        Args:
            is_looking_at_screen (bool): From VisionDetector (Gaze).
            is_face_present (bool): From VisionDetector.
            input_idle_time (float): Seconds since last input.
            
        Returns:
            dict: Structured inference result.
        """
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Cap dt to avoid large jumps if system lagged or paused
        if dt > 1.0: dt = 0.03 # Assume 30FPS equivalent if lag

        # 1. Normalize Signals
        input_score = self._normalize_input(input_idle_time)
        face_score = 1.0 if is_face_present else 0.0
        # If looking at screen, high gaze score. If not, low.
        # We can use the boolean for now, or use raw ratio if passed.
        # Assuming boolean is the robust "on screen" check.
        gaze_score = 1.0 if is_looking_at_screen else 0.0

        # Output Defaults
        usage_state = "Inactive"
        attention_mode = "Absent"
        reasoning = ""
        active_prob = 0.0
        confidence = 0.5

        # 2. Logic & Reasoning
        
        # A. Input Dominance (Active - Interactive)
        # "Sustained input activity alone is sufficient to infer Active Usage."
        if input_score > 0.6: # Threshold for "Active" input
            usage_state = "Active"
            attention_mode = "Interactive"
            active_prob = 0.9 + (0.1 * input_score) # 0.9 - 1.0
            reasoning = "User actively interacting with input devices."
            confidence = 0.95
        
        # B. Visual Validation (Input is low/absent)
        else:
            if face_score > 0.5: # Face Present
                if gaze_score > 0.5: # Gaze On Screen
                    # "Non-Interactive Focused Usage" -> Active
                    usage_state = "Active"
                    attention_mode = "Non-Interactive"
                    active_prob = 0.8
                    reasoning = "User focused on screen (Face + Gaze detected)."
                    confidence = 0.85
                else:
                    # Face Present, Gaze Away -> Passive
                    usage_state = "Passive" 
                    attention_mode = "Non-Interactive"
                    active_prob = 0.4
                    reasoning = "User present but gaze is diverted."
                    confidence = 0.7
            else:
                # No Input, No Face -> Inactive / Absent
                usage_state = "Inactive"
                attention_mode = "Absent"
                active_prob = 0.1
                reasoning = "No input or face detecting."
                confidence = 0.8

        # 3. Temporal Smoothing for Output Stability
        self.history.append(active_prob)
        smoothed_prob = sum(self.history) / len(self.history)
        
        # Update Internal State
        self.active_usage_probability = smoothed_prob
        self.usage_state = usage_state
        self.attention_mode = attention_mode
        self.reasoning_summary = reasoning
        self.confidence_score = confidence

        # 4. Filter short spurious states (optional, but good for "Stability")
        # For now, we trust the smoothed_prob could determine the state, 
        # but usage_state is derived from instantaneous logic. 
        # A mismatch between smoothed_prob and instantaneous state might occur.
        # Let's keep it simple: usage_state is instantaneous, prob is smoothed.
        # But we must accumulate based on decided state.
        
        # Accumulate Metrics
        if self.usage_state in self.state_durations:
            self.state_durations[self.usage_state] += dt

        # Log Session Data
        self.session_log.append({
            "timestamp": current_time,
            "state": self.usage_state,
            "mode": self.attention_mode,
            "prob": round(self.active_usage_probability, 3),
            "conf": round(self.confidence_score, 2),
            "input_idle": round(input_idle_time, 2)
        })

        # 5. Confidence Level Mapping
        if self.confidence_score > 0.8:
            self.confidence_level = "High"
        elif self.confidence_score > 0.5:
             self.confidence_level = "Medium"
        else:
            self.confidence_level = "Low"

        return {
            "active_usage_probability": round(self.active_usage_probability, 2),
            "usage_state": self.usage_state,
            "confidence_score": round(self.confidence_score, 2),
            "confidence_level": self.confidence_level,
            "attention_mode": self.attention_mode,
            "reasoning_summary": self.reasoning_summary,
            "metrics": {k: round(v, 1) for k, v in self.state_durations.items()}
        }

    def export_session_data(self, raw_data_path, report_path):
        """
        Exports the session data to two files: raw data log and summary report.
        Overwrites existing files.
        """
        import datetime

        # 1. Export Raw Data
        try:
            with open(raw_data_path, 'w') as f:
                f.write("Timestamp,State,Mode,Probability,Confidence,InputIdleTime\n")
                for entry in self.session_log:
                    ts = datetime.datetime.fromtimestamp(entry["timestamp"]).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    line = f"{ts},{entry['state']},{entry['mode']},{entry['prob']},{entry['conf']},{entry['input_idle']}\n"
                    f.write(line)
            print(f"Raw session data saved to {raw_data_path}")
        except Exception as e:
            print(f"Error saving raw data: {e}")

        # 2. Export Summary Report 
        try:
            total_duration = sum(self.state_durations.values())
            if total_duration == 0: total_duration = 0.001 # Avoid div/0

            with open(report_path, 'w') as f:
                f.write("==================================================\n")
                f.write("       SESSION ANALYTICS REPORT\n")
                f.write("==================================================\n\n")
                
                f.write(f"Session End Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Duration:   {total_duration:.2f} seconds\n\n")
                
                f.write("--------------------------------------------------\n")
                f.write("ENGAGEMENT BREAKDOWN\n")
                f.write("--------------------------------------------------\n")
                for state, duration in self.state_durations.items():
                    percentage = (duration / total_duration) * 100
                    f.write(f"{state:<10}: {duration:6.2f}s ({percentage:5.1f}%)\n")
                
                f.write("\n--------------------------------------------------\n")
                f.write("ATTENTION SUMMARY\n")
                f.write("--------------------------------------------------\n")
                
                # Simple average attention score calculation (Active=1.0, Passive=0.5, Inactive=0.0)
                # Weighted average
                weighted_sum = (self.state_durations["Active"] * 1.0) + (self.state_durations["Passive"] * 0.5)
                avg_attention = (weighted_sum / total_duration) * 100
                
                f.write(f"Average Engagement Score: {avg_attention:.1f} / 100\n")
                
                # Dominant State
                dominant_state = max(self.state_durations, key=self.state_durations.get)
                f.write(f"Dominant State:           {dominant_state}\n")
                
                f.write("\n==================================================\n")
                
            print(f"Session report saved to {report_path}")
        except Exception as e:
            print(f"Error saving session report: {e}")
