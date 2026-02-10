import time
from pynput.mouse import Controller as MouseController
from pynput.keyboard import Controller as KeyController
from pynput import mouse, keyboard

# Global variable to track the last time something happened
last_activity_time = time.time()

def reset_timer(*args):
    global last_activity_time
    last_activity_time = time.time()

# Start background listeners for mouse and keyboard
mouse.Listener(on_move=reset_timer, on_click=reset_timer, on_scroll=reset_timer).start()
keyboard.Listener(on_press=reset_timer).start()

print("Monitoring... (Ctrl+C to stop)")

while True:
    seconds_idle = time.time() - last_activity_time
    status = "ACTIVE" if seconds_idle < 10 else "IDLE"
    
    print(f"User is {status} ({int(seconds_idle)}s idle)", end="\r")
    time.sleep(1)