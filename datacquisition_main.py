import cv2
import mediapipe as mp

# === PARAMÈTRES ===
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
video_source = 0  # Webcam
output_file = "hand_landmarks.csv"
# ==================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(video_source)

landmarks_c = (234, 63, 247)
connection_c = (240, 240, 240)
thickness = 3
circle_r = 2
opened = True
tosave = []

def keepHandLandmarks(hand_landmarks_list, tosave):
    if len(tosave) == 0:
        header = []
        for hand_id, landmarks in enumerate(hand_landmarks_list):
            for i, lm in enumerate(landmarks.landmark):
                header.append(f"HAND{hand_id}_LANDMARK{i}_X")
                header.append(f"HAND{hand_id}_LANDMARK{i}_Y")
        tosave.append(header)
    
    line = []
    for hand_id, landmarks in enumerate(hand_landmarks_list):
        for lm in landmarks.landmark:
            line.append(str(lm.x))
            line.append(str(lm.y))
    tosave.append(line)
    print("added hand landmarks to list")

def saveLandmarks(tosave, filename):
    with open(filename, "w") as f:
        for line in tosave:
            f.write(",".join(line) + "\n")
    print("saved landmarks to", filename)

with mp_hands.Hands(
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence,
    max_num_hands=2
) as hands:

    while opened:
        opened, image = cap.read()
        if not opened:
            break

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        output_img = image.copy()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    output_img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=landmarks_c, thickness=thickness, circle_radius=circle_r),
                    mp_drawing.DrawingSpec(color=connection_c, thickness=thickness, circle_radius=circle_r)
                )

        cv2.imshow("MediaPipe Hand Detection", output_img)
        key = cv2.waitKey(10)
        if key & 0xFF == ord("a"):
            keepHandLandmarks(results.multi_hand_landmarks, tosave)
        if key & 0xFF == ord("q") or key == 27:
            break

cap.release()
cv2.destroyAllWindows()
saveLandmarks(tosave, output_file)
