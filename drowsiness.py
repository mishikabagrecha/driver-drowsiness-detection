import cv2
import mediapipe as mp
import numpy as np
import pygame
import datetime

# --- Sound Setup ---
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
mono = np.array([
    4096 * np.sin(2 * np.pi * 440 * t / 44100)
    for t in range(44100)
], dtype=np.int16)
stereo = np.column_stack((mono, mono))
beep = pygame.sndarray.make_sound(stereo)

# --- EAR Calculation ---
def calculate_ear(eye_points):
    p2_p6 = np.linalg.norm(eye_points[1] - eye_points[5])
    p3_p5 = np.linalg.norm(eye_points[2] - eye_points[4])
    p1_p4 = np.linalg.norm(eye_points[0] - eye_points[3])
    if p1_p4 == 0:
        return 0.3
    return (p2_p6 + p3_p5) / (2.0 * p1_p4)

# --- MAR Calculation ---
def calculate_mar(mouth_points):
    p2_p8 = np.linalg.norm(mouth_points[1] - mouth_points[7])
    p3_p7 = np.linalg.norm(mouth_points[2] - mouth_points[6])
    p4_p6 = np.linalg.norm(mouth_points[3] - mouth_points[5])
    p1_p5 = np.linalg.norm(mouth_points[0] - mouth_points[4])
    if p1_p5 == 0:
        return 0.0
    return (p2_p8 + p3_p7 + p4_p6) / (2.0 * p1_p5)

# --- Landmark Indices ---
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH     = [61, 40, 37, 0, 267, 270, 291, 321, 405, 17, 181, 91]

# --- Setup MediaPipe ---
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "face_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# --- Setup Webcam ---
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("ERROR: Webcam not accessible. Try index 1 or 2.")
    exit()

print("Calibrating for 4 seconds... Look straight, keep eyes open, mouth closed.")

# --- Calibration Variables ---
calibration_ears   = []
calibration_mars   = []
calibration_done   = False
fps_estimate       = 20
calibration_limit  = 4 * fps_estimate  # 4 seconds
ear_threshold      = 0.22  # will update after calibration
mar_threshold      = 0.75  # will update after calibration

# --- State Variables ---
closed_frame_count = 0
yawn_frame_count   = 0
alert_playing      = False
startup_grace      = 30   # ignore first 30 frames after calibration

# --- Log ---
def log_event(event_type):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {event_type}"
    print(entry)
    with open("drowsiness_log.txt", "a") as f:
        f.write(entry + "\n")

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame.")
        break

    frame     = cv2.flip(frame, 1)
    h, w, _   = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results   = detector.detect(mp_image)

    # Dark top bar
    cv2.rectangle(frame, (0, 0), (w, 130), (30, 30, 30), -1)

    if results.face_landmarks:
        landmarks = results.face_landmarks[0]

        # Draw face mesh
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 1, (0, 200, 0), -1)

        # Get coordinates
        left_eye_pts = np.array([
            [landmarks[i].x * w, landmarks[i].y * h]
            for i in LEFT_EYE
        ])
        right_eye_pts = np.array([
            [landmarks[i].x * w, landmarks[i].y * h]
            for i in RIGHT_EYE
        ])
        mouth_pts = np.array([
            [landmarks[i].x * w, landmarks[i].y * h]
            for i in MOUTH
        ])

        # Draw eye dots blue
        for pt in left_eye_pts:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (255, 100, 0), -1)
        for pt in right_eye_pts:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (255, 100, 0), -1)

        # Draw mouth dots yellow
        for pt in mouth_pts:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 255), -1)

        # Calculate values
        left_ear  = calculate_ear(left_eye_pts)
        right_ear = calculate_ear(right_eye_pts)
        avg_ear   = (left_ear + right_ear) / 2.0
        mar       = calculate_mar(mouth_pts)

        # --- Calibration Phase ---
        if not calibration_done:
            calibration_ears.append(avg_ear)
            calibration_mars.append(mar)

            remaining = max(0, int((calibration_limit - len(calibration_mars)) / fps_estimate) + 1)

            cv2.putText(frame, f"Calibrating... {remaining}s",
                        (int(w/2) - 160, int(h/2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 255), 3)
            cv2.putText(frame, "Eyes open, mouth closed, face straight",
                        (int(w/2) - 230, int(h/2) + 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 255, 255), 2)

            if len(calibration_mars) >= calibration_limit:
                # Use minimum EAR seen during calibration minus small buffer
                avg_ear_cal   = np.mean(calibration_ears)
                avg_mar_cal   = np.mean(calibration_mars)

                # EAR threshold = your natural minimum - buffer
                ear_threshold = avg_ear_cal - 0.08

                # MAR threshold = your natural resting + buffer
                # Use max of resting MAR to avoid false yawn triggers
                max_mar_cal   = np.max(calibration_mars)
                mar_threshold = max_mar_cal + 0.10

                calibration_done = True
                print(f"\nCalibration Complete!")
                print(f"Your resting EAR: {avg_ear_cal:.2f} | Eye threshold set to: {ear_threshold:.2f}")
                print(f"Your resting MAR: {avg_mar_cal:.2f} | Max MAR: {max_mar_cal:.2f} | Yawn threshold set to: {mar_threshold:.2f}")
                print(f"Starting detection...\n")

            cv2.imshow("Drowsiness Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                beep.stop()
                break
            continue

        # --- Grace Period After Calibration ---
        if startup_grace > 0:
            startup_grace -= 1
            cv2.putText(frame, "Starting...",
                        (int(w/2) - 80, int(h/2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3)
            cv2.imshow("Drowsiness Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                beep.stop()
                break
            continue

        # --- Detection Phase ---
        drowsy  = False
        yawning = False

        # Eye closure
        if avg_ear < ear_threshold:
            closed_frame_count += 1
            if closed_frame_count >= 12:
                drowsy = True
                if closed_frame_count == 12:
                    log_event("DROWSINESS DETECTED - Eye closure")
        else:
            closed_frame_count = 0

        # Yawn detection
        if mar > mar_threshold:
            yawn_frame_count += 1
            if yawn_frame_count >= 8:
                yawning = True
                if yawn_frame_count == 8:
                    log_event("YAWNING DETECTED")
        else:
            yawn_frame_count = 0

        # --- Alert ---
        if drowsy or yawning:
            red_overlay = frame.copy()
            cv2.rectangle(red_overlay, (0, 0), (w, h), (0, 0, 200), -1)
            frame = cv2.addWeighted(frame, 0.6, red_overlay, 0.4, 0)

            alert_text = "DROWSY!" if drowsy else "YAWNING!"
            cv2.putText(frame, f"ALERT: {alert_text}",
                        (int(w/2) - 160, int(h/2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (255, 255, 255), 3)

            if not alert_playing:
                beep.play(-1)
                alert_playing = True
        else:
            if alert_playing:
                beep.stop()
                alert_playing = False

        # --- UI ---
        cv2.putText(frame, f"EAR: {avg_ear:.2f}  (min: {ear_threshold:.2f})",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}  (yawn: {mar_threshold:.2f})",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        status = "DROWSY" if (drowsy or yawning) else "AWAKE"
        color  = (0, 255, 0) if status == "AWAKE" else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}",
                    (20, 105), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)

        time_str = datetime.datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, time_str,
                    (w - 120, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (200, 200, 200), 2)

    else:
        cv2.putText(frame, "No Face Detected",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

    cv2.imshow("Drowsiness Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        beep.stop()
        break

cap.release()
cv2.destroyAllWindows()
print("Stopped. Log saved to drowsiness_log.txt")