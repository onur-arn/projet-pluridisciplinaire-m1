import cv2
import mediapipe as mp
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import time

# ───── Paramètres MediaPipe ─────
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
model_complexity = 1
video_source = 0

# ───── Chargement du modèle et du label encoder ─────
model = load_model("model.h5")
label_encoder = joblib.load("label_encoder.pkl")
print("Modèle et label_encoder chargés avec succès.")

# ───── Mapping des indices utiles ─────
index_to_name = {
    0: "NOSE_X", 1: "NOSE_Y", 22: "LEFT_SHOULDER_X", 23: "LEFT_SHOULDER_Y",
    24: "RIGHT_SHOULDER_X", 25: "RIGHT_SHOULDER_Y", 26: "LEFT_ELBOW_X", 27: "LEFT_ELBOW_Y",
    28: "RIGHT_ELBOW_X", 29: "RIGHT_ELBOW_Y", 30: "LEFT_WRIST_X", 31: "LEFT_WRIST_Y",
    32: "RIGHT_WRIST_X", 33: "RIGHT_WRIST_Y", 46: "LEFT_HIP_X", 47: "LEFT_HIP_Y",
    48: "RIGHT_HIP_X", 49: "RIGHT_HIP_Y", 50: "LEFT_KNEE_X", 51: "LEFT_KNEE_Y",
    52: "RIGHT_KNEE_X", 53: "RIGHT_KNEE_Y", 54: "LEFT_ANKLE_X", 55: "LEFT_ANKLE_Y",
    56: "RIGHT_ANKLE_X", 57: "RIGHT_ANKLE_Y", 67: "action"
}
indices_a_garder = list(index_to_name.keys())[:-1]  # Exclure "action"

# ───── Initialisation de MediaPipe ─────
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(video_source)

landmarks_c = (234, 63, 247)
connection_c = 240
thickness = 2
circle_r = 2

with mp_pose.Pose(min_detection_confidence=min_detection_confidence,
                  min_tracking_confidence=min_tracking_confidence,
                  model_complexity=model_complexity) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Traitement d’image
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        output_img = frame.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                output_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(landmarks_c, thickness, circle_r),
                mp_drawing.DrawingSpec(connection_c, thickness, circle_r)
            )

            # ───── Extraction des coordonnées normalisées ─────
            landmarks = results.pose_landmarks.landmark
            try:
                coords = []
                for i in indices_a_garder:
                    x = landmarks[i // 2].x if i % 2 == 0 else landmarks[i // 2].y
                    coords.append(float(x))

                # Construction du dictionnaire
                pose_dict = {index_to_name[i]: coords[indices_a_garder.index(i)] for i in indices_a_garder}

                center_x = (
                    pose_dict["LEFT_SHOULDER_X"] + pose_dict["RIGHT_SHOULDER_X"] +
                    pose_dict["LEFT_HIP_X"] + pose_dict["RIGHT_HIP_X"]
                ) / 4
                center_y = (
                    pose_dict["LEFT_SHOULDER_Y"] + pose_dict["RIGHT_SHOULDER_Y"] +
                    pose_dict["LEFT_HIP_Y"] + pose_dict["RIGHT_HIP_Y"]
                ) / 4

                scale_left = np.sqrt(
                    (pose_dict["LEFT_SHOULDER_X"] - pose_dict["LEFT_HIP_X"]) ** 2 +
                    (pose_dict["LEFT_SHOULDER_Y"] - pose_dict["LEFT_HIP_Y"]) ** 2
                )
                scale_right = np.sqrt(
                    (pose_dict["RIGHT_SHOULDER_X"] - pose_dict["RIGHT_HIP_X"]) ** 2 +
                    (pose_dict["RIGHT_SHOULDER_Y"] - pose_dict["RIGHT_HIP_Y"]) ** 2
                )
                scale = (scale_left + scale_right) / 2

                if scale != 0:
                    normalized_vec = []
                    for i in indices_a_garder:
                        name = index_to_name[i]
                        val = pose_dict[name]
                        if name.endswith("_X"):
                            normalized_val = (val - center_x) / scale
                        elif name.endswith("_Y"):
                            normalized_val = (val - center_y) / scale
                        else:
                            normalized_val = val
                        normalized_vec.append(normalized_val)

                    vec = np.array(normalized_vec, dtype=np.float32).reshape(1, -1)

                    prediction = model.predict(vec)
                    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
                    action = predicted_class[0]
                    print(f"Action prédite : {action}")

                    cv2.putText(output_img, f"Action: {action}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    print("Échelle nulle, normalisation ignorée.")
            except Exception as e:
                print(f"Erreur pendant la prédiction : {e}")

        cv2.imshow("Pose Detection + Prediction", output_img)

        key = cv2.waitKey(10)
        if key & 0xFF == ord("q") or key == 27:
            break

cap.release()
cv2.destroyAllWindows()
