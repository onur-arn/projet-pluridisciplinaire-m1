import cv2
import mediapipe as mp

# === PARAMÈTRES ===
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
video_source = 0  # Webcam
output_file = "face_landmarks.csv"
# ==================

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(video_source)

landmarks_c = (0, 255, 0)
connection_c = (255, 0, 0)
thickness = 1
circle_r = 1
opened = True
tosave = []

def keepFaceLandmarks(face_landmarks_list, tosave):
    if len(tosave) == 0:
        header = []
        for face_id, landmarks in enumerate(face_landmarks_list):
            for i, lm in enumerate(landmarks.landmark):
                header.append(f"FACE{face_id}_LANDMARK{i}_X")
                header.append(f"FACE{face_id}_LANDMARK{i}_Y")
        tosave.append(header)
    
    line = []
    for face_id, landmarks in enumerate(face_landmarks_list):
        for lm in landmarks.landmark:
            line.append(str(lm.x))
            line.append(str(lm.y))
    tosave.append(line)
    print("Added face landmarks to list")

def saveLandmarks(tosave, filename):
    with open(filename, "w") as f:
        for line in tosave:
            f.write(",".join(line) + "\n")
    print("Saved landmarks to", filename)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence,
    refine_landmarks=True  # active iris landmarks
) as face_mesh:

    while opened:
        opened, image = cap.read()
        if not opened:
            break

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        output_img = image.copy()

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    output_img,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=landmarks_c, thickness=thickness, circle_radius=circle_r),
                    mp_drawing.DrawingSpec(color=connection_c, thickness=thickness, circle_radius=circle_r)
                )

        cv2.imshow("MediaPipe Face Mesh Detection", output_img)
        key = cv2.waitKey(10)
        if key & 0xFF == ord("a"):
            keepFaceLandmarks(results.multi_face_landmarks, tosave)
        if key & 0xFF == ord("q") or key == 27:
            break

cap.release()
cv2.destroyAllWindows()
saveLandmarks(tosave, output_file)
