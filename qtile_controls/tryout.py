from libqtile.command.client import InteractiveCommandClient
import subprocess
import time

def setup_workspace():
    # Connect to Qtile
    c = InteractiveCommandClient()
    
    # Switch to group 2
    c.group["4"].toscreen()
    
    # Launch Chrome
    subprocess.Popen(["alacritty"])  # or "chromium" depending on your system
    time.sleep(2)  # Wait for Chrome to open
    
    # Move Chrome to group 2 (if not already there)
    c.window.togroup("4")

if __name__ == "__main__":
    setup_workspace()