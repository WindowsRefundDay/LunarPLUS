import ctypes
import cv2
import json
import math
import mss
import os
import sys
import time
import torch
from torch import compile
import numpy as np
import win32api
from termcolor import colored
from ultralytics import YOLO

# Cache frequently used functions and optimize imports
_get_system_metrics = ctypes.windll.user32.GetSystemMetrics
_send_input = ctypes.windll.user32.SendInput
_get_key_state = win32api.GetKeyState
_get_async_key_state = win32api.GetAsyncKeyState
PUL = ctypes.POINTER(ctypes.c_ulong)

# Auto Screen Resolution - compute once
SCREEN_RES_X = _get_system_metrics(0)
SCREEN_RES_Y = _get_system_metrics(1)
SCREEN_CENTER_X = SCREEN_RES_X >> 1  # Faster integer division
SCREEN_CENTER_Y = SCREEN_RES_Y >> 1

# Constants
AIM_HEIGHT = 10
CONFIDENCE_THRESHOLD = 0.45
USE_TRIGGER_BOT = True
MOUSE_LEFT = 0x01
MOUSE_RIGHT = 0x02
TARGET_THRESHOLD = 5
MOUSE_EVENT_LEFTDOWN = 0x0002
MOUSE_EVENT_LEFTUP = 0x0004
CV_WAITKEY_DELAY = 1  # Minimum delay for cv2.waitKey

# Pre-compute box constants
BOX_DEFAULT_SIZE = 350
BOX_SELF_DETECT_THRESHOLD = 15
BOX_SIDE_RATIO = 0.2
BOX_BOTTOM_RATIO = 1.2

class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

class Aimbot:
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    screen = mss.mss()
    pixel_increment = 1
    
    # Pre-allocate numpy arrays for frame processing
    frame_buffer = None
    
    # Load config once during class initialization
    try:
        with open("lib/config/config.json") as f:
            sens_config = json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        sens_config = {"targeting_scale": 1.0}
    
    aimbot_status = colored("ENABLED", 'green')
    _aimbot_status_cache = True

    def __init__(self, box_constant=BOX_DEFAULT_SIZE, collect_data=False, mouse_delay=0.0009):
        self.box_constant = box_constant
        self.box_center = box_constant / 2
        self.box_fifth = box_constant * BOX_SIDE_RATIO
        
        # Pre-allocate frame buffer
        Aimbot.frame_buffer = np.zeros((box_constant, box_constant, 3), dtype=np.uint8)
        
        # Training mode setup
        self.collect_data = collect_data
        if collect_data:
            self.data_dir = "lib/data/images"
            self.label_dir = "lib/data/labels"
            os.makedirs(self.data_dir, exist_ok=True)
            os.makedirs(self.label_dir, exist_ok=True)
            self.image_count = len(os.listdir(self.data_dir))
            print(colored("[INFO] Training mode activated. Press 'C' to capture screenshots.", "yellow"))
            print(colored("[INFO] Aim at the target and press 'C' to capture.", "yellow"))

        # Precompute detection box coordinates
        half_box = box_constant >> 1  # Faster integer division
        self.detection_box = {
            'left': SCREEN_CENTER_X - half_box,
            'top': SCREEN_CENTER_Y - half_box,
            'width': box_constant,
            'height': box_constant
        }

        print("[INFO] Loading the neural network model")
        self.model = YOLO('lib/best.pt')
        
        if torch.cuda.is_available():
            print(colored("CUDA ACCELERATION [ENABLED]", "green"))
            # Enable TensorRT optimization if available
            try:
                self.model.model = compile(self.model.model, mode="reduce-overhead")
                print(colored("TORCH COMPILE OPTIMIZATION [ENABLED]", "green"))
                # Set CUDA optimization flags
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception as e:
                print(f"Compile optimization failed: {e}")

        self.conf = CONFIDENCE_THRESHOLD
        self.iou = 0.80
        self.collect_data = collect_data
        self.mouse_delay = mouse_delay
        
        # Create window with optimized flags
        cv2.namedWindow("LunarPLUS Vision", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        
        print("\n[INFO] PRESS 'F1' TO TOGGLE AIMBOT\n[INFO] PRESS 'F2' TO QUIT")

    @staticmethod
    def update_status_aimbot():
        Aimbot._aimbot_status_cache = not Aimbot._aimbot_status_cache
        Aimbot.aimbot_status = colored("ENABLED" if Aimbot._aimbot_status_cache else "DISABLED", 
                                     'green' if Aimbot._aimbot_status_cache else 'red')
        sys.stdout.write("\033[K")
        print(f"[!] AIMBOT IS [{Aimbot.aimbot_status}]", end="\r")

    @staticmethod
    def left_click():
        ctypes.windll.user32.mouse_event(MOUSE_EVENT_LEFTDOWN)
        Aimbot.sleep(0.0001)
        ctypes.windll.user32.mouse_event(MOUSE_EVENT_LEFTUP)

    @staticmethod
    def sleep(duration, get_now=time.perf_counter):
        if duration <= 0: return
        end = get_now() + duration
        while get_now() < end:
            pass

    @classmethod
    def is_aimbot_enabled(cls):
        return cls._aimbot_status_cache

    @staticmethod
    def is_shooting():
        return _get_key_state(MOUSE_LEFT) < 0

    @staticmethod
    def is_targeted():
        return _get_key_state(MOUSE_RIGHT) < 0

    @staticmethod
    def is_target_locked(x, y):
        return abs(x - SCREEN_CENTER_X) <= TARGET_THRESHOLD and abs(y - SCREEN_CENTER_Y) <= TARGET_THRESHOLD

    def move_crosshair(self, x, y):
        if not Aimbot.is_targeted():
            return

        scale = self.sens_config["targeting_scale"]
        input_obj = Input(ctypes.c_ulong(0), self.ii_)
        
        for rel_x, rel_y in self.interpolate_coordinates_from_center((x, y), scale):
            self.ii_.mi = MouseInput(rel_x, rel_y, 0, 0x0001, 0, ctypes.pointer(self.extra))
            _send_input(1, ctypes.byref(input_obj), ctypes.sizeof(input_obj))
            if self.mouse_delay > 0:
                self.sleep(self.mouse_delay)

    def interpolate_coordinates_from_center(self, absolute_coordinates, scale):
        diff_x = (absolute_coordinates[0] - SCREEN_CENTER_X) * scale / self.pixel_increment
        diff_y = (absolute_coordinates[1] - SCREEN_CENTER_Y) * scale / self.pixel_increment
        
        length = int(math.hypot(diff_x, diff_y))  
        if length == 0:
            return
            
        unit_x = (diff_x / length) * self.pixel_increment
        unit_y = (diff_y / length) * self.pixel_increment
        
        x = y = sum_x = sum_y = 0
        for k in range(length):
            sum_x += x
            sum_y += y
            x, y = round(unit_x * k - sum_x), round(unit_y * k - sum_y)
            yield x, y

    def start(self):
        print("[INFO] Beginning screen capture")
        Aimbot.update_status_aimbot()
        last_time = time.perf_counter()
        frame_count = 0
        fps_update_time = last_time
        fps = 0  # Initialize fps variable
        
        try:
            while True:
                # Efficient frame capture
                frame = np.asarray(self.screen.grab(self.detection_box))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # FPS calculation with smoothing
                current_time = time.perf_counter()
                frame_count += 1
                
                if current_time - fps_update_time >= 0.5:  # Update FPS every 0.5 seconds
                    fps = frame_count / (current_time - fps_update_time)
                    frame_count = 0
                    fps_update_time = current_time
                
                # Use half precision and disable gradient computation
                with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                    boxes = self.model.predict(
                        source=frame,
                        verbose=False,
                        conf=self.conf,
                        iou=self.iou,
                        half=True,
                        agnostic_nms=True
                    )
                
                result = boxes[0]
                if result.boxes.xyxy.shape[0] > 0:
                    least_crosshair_dist = float('inf')
                    closest_detection = None
                    player_in_frame = False
                    box_center = self.box_constant >> 1  # Faster integer division

                    # Vectorized operations for box processing
                    for box in result.boxes.xyxy:
                        x1, y1, x2, y2 = map(int, box)
                        height = y2 - y1
                        
                        relative_head_X = (x1 + x2) >> 1
                        relative_head_Y = int(y1 + height * (1 - 1/AIM_HEIGHT))
                        
                        if x1 < BOX_SELF_DETECT_THRESHOLD or (x1 < self.box_fifth and y2 > self.box_constant/BOX_BOTTOM_RATIO):
                            player_in_frame = True
                            continue

                        dx = relative_head_X - box_center
                        dy = relative_head_Y - box_center
                        crosshair_dist = dx * dx + dy * dy
                        
                        if crosshair_dist < least_crosshair_dist:
                            least_crosshair_dist = crosshair_dist
                            closest_detection = {
                                "relative_head_X": relative_head_X,
                                "relative_head_Y": relative_head_Y
                            }

                    if closest_detection and self.is_aimbot_enabled():
                        self.move_crosshair(
                            closest_detection["relative_head_X"],
                            closest_detection["relative_head_Y"]
                        )
                        
                        if (USE_TRIGGER_BOT and 
                            self.is_target_locked(
                                closest_detection["relative_head_X"],
                                closest_detection["relative_head_Y"]
                            )):
                            self.left_click()

                # Efficient FPS display
                cv2.putText(frame, f"FPS: {int(fps)}", (5, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (113, 116, 244), 2, cv2.LINE_AA)
                
                # Optimized frame display
                cv2.imshow("LunarPLUS Vision", frame)
                
                # Check for quit keys
                if cv2.waitKey(1) & 0xFF in (ord('0'), ord('2')):
                    break
                
                # Check for F2 key separately for faster response
                if _get_async_key_state(0x71) & 0x8000:  # F2 key
                    break
                
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.clean_up()

    @staticmethod
    def clean_up():
        print("\n[INFO] F2 WAS PRESSED. QUITTING LUNARPLUS...")
        cv2.destroyAllWindows()
        Aimbot.screen.close()
        os._exit(1)  

if __name__ == "__main__": print("You are in the wrong directory and are running the wrong file; you must run lunar.py")
