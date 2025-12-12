from wayland_automation.keyboard_controller import Keyboard
import time

kb = Keyboard()
time.sleep(4)  # Gives time to switch to another window

kb.typewrite("Hello, Wayland!", interval=0.02)
#kb.press("enter")