import os
import pickle
import cv2
import mediapipe as mp
from tqdm import tqdm

# --------------------------------------------
# Mediapipe setup
# --------------------------------------------
mp_hands = mp.solutions.hands                  # Mediapipe hand detection module

hands = mp_hands.Hands(
    static_image_mode=True,                    # Treat each image independently (no tracking)
    max_num_hands=1,                           # Only one hand expected per image
    min_detection_confidence=0.5               # Minimum confidence for detecting a hand
)

# --------------------------------------------
# Data folder path
# --------------------------------------------
DATA_DIR = './data'

data = []
labels = []

# List classes ['A', 'B', 'C', ..., 'Z']
classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
print(f"Found classes: {classes}")

# --------------------------------------------
# Loop through each class folder
# --------------------------------------------
for class_name in classes:
    class_dir = os.path.join(DATA_DIR, class_name)
    images = os.listdir(class_dir)

    print(f"\nProcessing class: {class_name} ({len(images)} images)")

    # Loop through all images in the folder with a progress bar
    for img_name in tqdm(images):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Warning: Can't read image: {img_path}")
            continue

        # Convert BGR → RGB (Mediapipe expects RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract hand landmarks
        results = hands.process(img_rgb)

        # Skip images where no hand was detected
        if not results.multi_hand_landmarks:
            continue

        # Only one hand expected
        hand = results.multi_hand_landmarks[0]

        x_coords = [lm.x for lm in hand.landmark]
        y_coords = [lm.y for lm in hand.landmark]

        # Normalize using the minimum point
        min_x, min_y = min(x_coords), min(y_coords)

        normalized_landmarks = []
        for lm in hand.landmark:
            normalized_landmarks.append(lm.x - min_x)   # 42 values (21 landmarks × 2 coords)
            normalized_landmarks.append(lm.y - min_y)   # The alphabet letter (A-Z)

        data.append(normalized_landmarks)
        labels.append(class_name)

# --------------------------------------------
# Save the dataset
# --------------------------------------------
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("\n📦 Dataset saved as data.pickle")
print(f"Total samples: {len(data)}")