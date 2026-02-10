import time
from pynput import mouse, keyboard
import threading

class InputMonitor:
    def __init__(self):
        self.last_activity_time = time.time()
        self.mouse_listener = None
        self.keyboard_listener = None
        self.is_running = False

    def _reset_timer(self, *args):
        self.last_activity_time = time.time()

    def start(self):
        if self.is_running:
            return
        
        # Start non-blocking listeners
        self.mouse_listener = mouse.Listener(
            on_move=self._reset_timer,
            on_click=self._reset_timer,
            on_scroll=self._reset_timer
        )
        self.keyboard_listener = keyboard.Listener(
            on_press=self._reset_timer
        )
        
        self.mouse_listener.start()
        self.keyboard_listener.start()
        self.is_running = True

    def stop(self):
        if self.mouse_listener:
            self.mouse_listener.stop()
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        self.is_running = False

    def get_idle_time(self):
        return time.time() - self.last_activity_time

    def is_active(self, threshold_seconds=5.0):
        return self.get_idle_time() < threshold_seconds
