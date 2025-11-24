import pickle
import cv2
import mediapipe as mp
import numpy as np

# -------------------------------
# Load trained model
# -------------------------------
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# -------------------------------
# Camera
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Unable to access the webcam.")

# -------------------------------
# Mediapipe config
# -------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,       # Real-time mode (tracking active)
    max_num_hands=1,               # Detect only 1 hand
    min_detection_confidence=0.5,  # Minimum confidence for initial detection
    min_tracking_confidence=0.5    # Confidence needed to track the hand
)

# label dict
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
               5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
               15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
               20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
               25: 'Z'}

# -------------------------------
# MAIN
# -------------------------------
print("🎥 Hand recognition started (press ESC to exit)")

while True:

    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame.")
        break

    # Convert BGR → RGB (required by Mediapipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(frame_rgb)

    # Storage for normalized landmark data
    data_aux = []
    x_vals, y_vals = [], []

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw the landmarks
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style()
        )

        # Collect coordinates
        for lm in hand_landmarks.landmark:
            x_vals.append(lm.x)
            y_vals.append(lm.y)

        # Normalize
        min_x, min_y = min(x_vals), min(y_vals)

        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min_x)
            data_aux.append(lm.y - min_y)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------
        try:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_char = labels_dict[int(prediction[0])]
        except Exception as e:
            predicted_char = "?"
            print("⚠️ Prediction error:", e)

        # ----------------------------------------------------
        # Draw bounding box around the hand
        # ----------------------------------------------------
        h, w, _ = frame.shape

        x1 = int(min_x * w) - 20
        y1 = int(min_y * h) - 20
        x2 = int(max(x_vals) * w) + 20
        y2 = int(max(y_vals) * h) + 20

        # Draw rectangle + text
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
        cv2.putText(frame,
                    predicted_char,
                    (x1, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 0),
                    3
                    )

    # Show the final frame
    cv2.imshow("Hand Recognition", frame)

    # Press ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()