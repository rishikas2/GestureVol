import cv2
import numpy as np
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import keyboard
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# Audio Control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]

# OpenCV with 75% larger frame size (1280x960)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

# Volume control
vol = volume.GetMasterVolumeLevel()
vol_bar = np.interp(vol, [min_vol, max_vol], [700, 200])  # Adjusted for new size
vol_per = np.interp(vol, [min_vol, max_vol], [0, 100])
volume_locked = False
last_toggle_time = 0
TOGGLE_COOLDOWN = 2

# Gesture control
last_gesture_time = 0
GESTURE_COOLDOWN = 0.5
SWIPE_THRESHOLD = 0.15
PINCH_THRESHOLD = 0.05  # Threshold for pinch gesture
last_mute_time = 0
MUTE_COOLDOWN = 1  # Cooldown for mute/unmute to prevent rapid toggling

def count_fingers(landmarks):
    finger_tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    finger_mcp = [
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.PINKY_MCP
    ]
    
    extended_fingers = 0
    
    # Check thumb separately
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    if thumb_tip.x < thumb_ip.x:
        extended_fingers += 1
    
    # Check other fingers
    for tip, mcp in zip(finger_tips, finger_mcp):
        if landmarks[tip].y < landmarks[mcp].y:
            extended_fingers += 1
    
    return extended_fingers

def detect_swipe(landmarks, prev_landmarks):
    if prev_landmarks is None:
        return None
    
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    prev_wrist = prev_landmarks[mp_hands.HandLandmark.WRIST]
    x_diff = wrist.x - prev_wrist.x
    
    if x_diff > SWIPE_THRESHOLD:
        return "right"
    elif x_diff < -SWIPE_THRESHOLD:
        return "left"
    return None

def detect_pinch(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    distance = np.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
    return distance < PINCH_THRESHOLD

prev_landmarks = None
muted = volume.GetMute()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    current_time = time.time()
    
    # Pink centered title
    text_size = cv2.getTextSize("GESTUREVOL", cv2.FONT_HERSHEY_SIMPLEX, 2, 4)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    cv2.putText(frame, "GESTUREVOL", (text_x, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 2, (180, 105, 255), 4)
    
    # Red volume status (top-left)
    status_text = f"VOLUME: {'LOCKED' if volume_locked else 'UNLOCKED'}"
    cv2.putText(frame, status_text, (30, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Mute status (top-left below volume status)
    mute_text = f"MUTE: {'ON' if muted else 'OFF'}"
    cv2.putText(frame, mute_text, (30, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Volume bar (right side with padding)
    bar_left = frame.shape[1] - 150  # Right side with margin
    cv2.rectangle(frame, (bar_left, 200), (bar_left + 50, 700), (0, 255, 0), 4)
    cv2.rectangle(frame, (bar_left, int(vol_bar)), (bar_left + 50, 700), (0, 255, 0), cv2.FILLED)
    cv2.putText(frame, f'{int(vol_per)}%', (bar_left - 10, 750), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    # Blue control panel (left side with padding)
    control_panel_x = 50
    control_panel_y = 120
    line_spacing = 40
    
    controls = [
        "3 Fingers: Toggle Volume Lock",
        "Spread fingers: Adjust Volume",
        "Pinch: Mute/Unmute",
        "Fist: Play/Pause",
        "Swipe Left: Previous Track",
        "Swipe Right: Next Track",
        "Press Q to quit"
    ]
    
    for i, control in enumerate(controls):
        y_pos = control_panel_y + (i * line_spacing)
        cv2.putText(frame, control, (control_panel_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    gesture_detected = None
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            finger_count = count_fingers(landmarks)
            swipe_direction = detect_swipe(landmarks, prev_landmarks) if prev_landmarks else None
            is_pinching = detect_pinch(landmarks)
            
            # Toggle volume lock with 3 fingers
            if finger_count == 3 and (current_time - last_toggle_time > TOGGLE_COOLDOWN):
                volume_locked = not volume_locked
                last_toggle_time = current_time
                time.sleep(0.3)  # Brief feedback pause
            
            # Mute/unmute with pinch gesture
            if is_pinching and (current_time - last_mute_time > MUTE_COOLDOWN):
                muted = not muted
                volume.SetMute(muted, None)
                last_mute_time = current_time
                time.sleep(0.3)  # Brief feedback pause
                cv2.putText(frame, "ACTIVE: Mute Toggle", (control_panel_x, 400), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # Gesture detection
            if finger_count == 0:
                gesture_detected = "play_pause"
                cv2.putText(frame, "ACTIVE: Play/Pause", (control_panel_x, 400), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            elif swipe_direction == "right":
                gesture_detected = "next_track"
                cv2.putText(frame, "ACTIVE: Next Track", (control_panel_x, 400), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            elif swipe_direction == "left":
                gesture_detected = "prev_track"
                cv2.putText(frame, "ACTIVE: Previous Track", (control_panel_x, 400), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # Volume control
            if not volume_locked and finger_count > 0 and not is_pinching:
                thumb = landmarks[mp_hands.HandLandmark.THUMB_TIP]
                index = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                finger_distance = np.hypot(thumb.x - index.x, thumb.y - index.y)
                vol = np.interp(finger_distance, [0.02, 0.3], [min_vol, max_vol])
                vol = np.clip(vol, min_vol, max_vol)
                vol_bar = np.interp(vol, [min_vol, max_vol], [700, 200])
                vol_per = np.interp(vol, [min_vol, max_vol], [0, 100])
                volume.SetMasterVolumeLevel(vol, None)
            
            prev_landmarks = landmarks
    
    # Execute gestures
    if gesture_detected and (current_time - last_gesture_time > GESTURE_COOLDOWN):
        if gesture_detected == "play_pause":
            keyboard.send("play/pause")
        elif gesture_detected == "next_track":
            keyboard.send("next track")
        elif gesture_detected == "prev_track":
            keyboard.send("previous track")
        last_gesture_time = current_time
    
    cv2.imshow('Media Control', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()