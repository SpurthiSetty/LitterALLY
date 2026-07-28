# LitterALLY
This is the goithub for the Arduino INtern s Challenge 

# TrashTALK - Smart Waste Detection System

TrashTALK is an integrated edge-AI system built on the Arduino UNO Q. It uses a Time-of-Flight (ToF) distance sensor and a piezobuzzer on the microcontroller unit (MCU) side to detect when an item is placed in front of the scanner. Upon holding the item in place within 25 cm for 3 seconds, a hardware Bridge signal triggers a Python application running on the Linux Microprocessing Unit (MPU) to activate a USB webcam and stream live video to a Streamlit Web UI.

---

## System Architecture

[ Modulino Distance ] --(10 Hz)--> [ UNO Q MCU ]
                                       |
                      (Hold < 25cm for 3s)
                                       |
                               [ Modulino Buzzer ] (Beep!)
                                       |
                             Arduino RouterBridge
                                       |
                                       v
                             [ Linux MPU Python ]
                                       |
                                (OpenCV V4L2)
                                       |
                                       v
                            [ USB Webcam Feed ]
                                       |
                                       v
                        [ Streamlit UI @ Port 7000 ]

---

## Repository Structure

TrashTALK/
├── sketch/
│   └── sketch.ino       # C++ Sketch running on Zephyr MCU (Distance & Buzzer)
├── python/
│   └── main.py          # Python app running on Linux MPU (Bridge, OpenCV, Streamlit)
├── app.yaml             # App Lab environment configuration
└── README.md            # System documentation

---

## Setup Instructions for Teammates

### 1. Hardware Requirements
- Board: Arduino UNO Q
- Sensors & Modules:
  - Modulino Distance Sensor (Time-of-Flight)
  - Modulino Buzzer
- Peripherals:
  - Standard USB Webcam (plugged into the UNO Q USB Host Port)
  - Power supply / USB-C connection to your PC

---

### 2. Opening the Project in Arduino App Lab
1. Launch Arduino App Lab.
2. Connect your UNO Q board via USB/Wi-Fi and ensure it is selected at the bottom bar.
3. Import or open the TrashTALK project folder.
4. Add the required brick to the workspace:
   - In the left panel under Bricks, click "WebUI - Streamlit".

---

### 3. Code Files Setup

#### sketch/sketch.ino
Ensure your sketch contains the timing and state-machine filtering logic to handle distance detection:

#include "Modulino.h"
#include <Arduino_RouterBridge.h>

ModulinoDistance distance;
ModulinoBuzzer buzzer;

const int MIN_DEADZONE_MM = 20;            // 2 cm optical floor limit
const int DIST_THRESHOLD_MM = 250;         // 25 cm target threshold
const unsigned long HOLD_TIME_MS = 3000;   // 3 seconds hold requirement
const unsigned long POLL_INTERVAL_MS = 100; // 10 Hz refresh rate

unsigned long lastPollTime = 0;
unsigned long targetStartTime = 0;
bool trackingActive = false;
bool cameraActive = false;

void setup() {
    Monitor.begin();
    delay(1000);

    Modulino.begin();
    distance.begin();
    buzzer.begin();

    Monitor.println("====================================");
    Monitor.println("  SYSTEM READY: CAM + BUZZ + DIST   ");
    Monitor.println("====================================");
}

void loop() {
    unsigned long currentMillis = millis();

    if (currentMillis - lastPollTime >= POLL_INTERVAL_MS) {
        lastPollTime = currentMillis;

        if (distance.available()) {
            int raw_mm = distance.get();

            // Ignore invalid 0 mm optical noise glitch
            if (raw_mm == 0) {
                return;
            }

            bool isWithinRange = (raw_mm >= MIN_DEADZONE_MM && raw_mm <= DIST_THRESHOLD_MM);

            if (isWithinRange) {
                // Hand/object is currently present (< 25 cm)
                if (!trackingActive) {
                    // Start the 3-second timer
                    trackingActive = true;
                    targetStartTime = currentMillis;
                    Monitor.println(">>> Target detected! Starting 3s timer... <<<");
                } 
                else if ((currentMillis - targetStartTime >= HOLD_TIME_MS) && !cameraActive) {
                    // 3 seconds passed -> Turn on camera
                    cameraActive = true;
                    Monitor.println(">>> 3s HOLD COMPLETE! BEEPING & STARTING CAMERA <<<");
                    
                    buzzer.tone(1000, 300); // Audible beep
                    Bridge.notify("camera_cmd", "START");
                }
            } 
            else {
                // Object moved out of range (> 25 cm or removed)
                if (cameraActive) {
                    // Stop condition: Distance > 25 cm while camera was running
                    Monitor.println(">>> OBJECT MOVED OUT OF RANGE (>25cm): STOPPING CAMERA <<<");
                    Bridge.notify("camera_cmd", "STOP");
                    cameraActive = false;
                }
                
                // Reset tracking state completely
                trackingActive = false;
            }
        }
    }
}


#### python/main.py
Paste the following unified MPU controller script:

from arduino.app_utils import App, Bridge
from arduino.app_bricks.streamlit_ui import st
import cv2
import time

camera = None
camera_active = False

def get_working_camera():
    """Scans V4L2 device indices to discover and lock the connected USB webcam."""
    global camera
    if camera is not None and camera.isOpened():
        return camera
    
    # Common V4L2 device index fallbacks on embedded Linux
    for idx in [2, 0, 1, 4]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"\n>>> USB WEBCAM READY AT INDEX {idx}! <<<\n")
                camera = cap
                return camera
            cap.release()
    return None

def on_camera_cmd(status):
    """Bridge handler for notifications from C++ sketch."""
    global camera_active
    print(f"\n>>> [PYTHON RECEIVED BRIDGE CMD]: {status} <<<")
    if status == "START":
        print(">>> [PYTHON] ACTIVATING CAMERA FEED... <<<")
        camera_active = True
    elif status == "STOP":
        print(">>> [PYTHON] PAUSING CAMERA FEED... <<<\n")
        camera_active = False
    return True

# Register Bridge command mapping
Bridge.provide("camera_cmd", on_camera_cmd)

# --- Streamlit Web Interface Setup ---
st.title("TrashTALK - Live WebCam Feed")

# Placeholders to prevent duplicate UI re-renders
status_box = st.empty()
image_box = st.empty()

# Initialize camera hardware on startup
camera = get_working_camera()

def loop():
    global camera, camera_active
    
    if camera_active:
        if camera is None or not camera.isOpened():
            camera = get_working_camera()

        if camera is not None and camera.isOpened():
            ret, frame = camera.read()
            if ret:
                status_box.success("STATUS: WEBCAM LIVE STREAMING")
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_box.image(frame_rgb, channels="RGB", use_container_width=True)
            else:
                status_box.warning("Camera active, reading frame...")
    else:
        status_box.info("Status: Idle — Hold object within 25 cm for 3 seconds to activate camera.")
        image_box.empty()

    time.sleep(0.05)

App.run(user_loop=loop)

---

## How to Run and Test

1. Connect your laptop to the same Wi-Fi network as the UNO Q board.
2. In Arduino App Lab, click the Run button.
3. Observe the output in the bottom console tabs:
   - Python tab: Confirms "USB WEBCAM READY AT INDEX 2!" and Streamlit startup URLs.
   - Serial Monitor tab: Shows real-time distance measurements from the Modulino Distance sensor.
4. Open your web browser (Edge, Chrome, or Safari) and go to:
   http://<BOARD-IP>:7000
   (e.g., http://192.168.1.89:7000)
5. Testing the Trigger:
   - Place an object/hand 2 cm to 25 cm away from the distance sensor.
   - Keep it held in place for 3 seconds.
   - The Modulino Buzzer will beep, and the Streamlit UI will update from Idle to "STATUS: WEBCAM LIVE STREAMING" with real-time video!
   - Pull the object away past 25 cm—the camera feed will stop immediately.

---

## Troubleshooting Guide

- Web UI page shows "Can't reach this page": Ensure you are on the same Wi-Fi network as the UNO Q board and that you appended :7000 to the IP address.
- Camera stream freezes or doesn't start: Verify the USB webcam is plugged firmly into the board prior to hitting Run. Check the Python tab to verify index discovery (/dev/video2).
- Timer keeps resetting unexpectedly: Ensure the distance sensor lens is clear. Readings of 0 cm caused by optical contact are filtered in sketch.ino.
