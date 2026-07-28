
###################################Basic code 

# from arduino.app_utils import App
# import time

# def loop():
#     time.sleep(1)

# App.run(user_loop=loop)


###################################Camera Code
from arduino.app_utils import App, Bridge
from arduino.app_bricks.streamlit_ui import st
import cv2
import time

camera = None
camera_active = False

def get_working_camera():
    """Attempts to grab an active camera handle."""
    global camera
    if camera is not None and camera.isOpened():
        return camera
    
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
    global camera_active
    print(f"\n>>> [PYTHON RECEIVED BRIDGE CMD]: {status} <<<")
    if status == "START":
        print(">>> [PYTHON] ACTIVATING CAMERA FEED... <<<")
        camera_active = True
    elif status == "STOP":
        print(">>> [PYTHON] PAUSING CAMERA FEED... <<<\n")
        camera_active = False
    return True

# Register bridge listener
Bridge.provide("camera_cmd", on_camera_cmd)

# --- UI Setup ---
st.title("TrashTALK - Live WebCam Feed")

# Static UI placeholders to prevent duplicated text lines
status_box = st.empty()
image_box = st.empty()

# Pre-warm the camera handle once
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
                status_box.warning("Camera active but frame read failed. Retrying...")
    else:
        status_box.info("Status: Idle — Hold object within 10 cm for 3 seconds to activate camera.")
        image_box.empty()

    time.sleep(0.05)

App.run(user_loop=loop)
