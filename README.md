# ✋ **ASL Hand Gesture Recognition**

This project uses **MediaPipe Hands** + **Machine Learning (Random Forest)** to recognize American Sign Language alphabet gestures (A–Z) from a webcam feed.

It includes:

- Data preprocessing

- Model training

- Real-time prediction using webcam

**Link** : [Official MediaPipe Face Mesh documentation](https://mediapipe.readthedocs.io/en/latest/solutions/hands.html)

## 📁 **Repository Structure**
```bash
├── README.md              # Project documentation
├── main.py                # Real-time hand gesture recognition (webcam)
├── create_data.py         # Create your own data
├── model.py               # Train the RandomForest model
├── clean_data.py          # Extract landmarks from images → data.pickle
├── data.pickle            # Preprocessed dataset (landmarks)
└── model.p                # Trained RandomForest model
```

## 🚀 **Features**

- Detects a hand using **MediaPipe**

- Extracts **21 hand landmarks** (x/y coordinates)

- **Normalizes** data for ML training

- Predicts ASL letters A–Z live from **webcam**

- **Lightweight** (no deep learning required)

- Works in **real time**

<img width="640" height="338" alt="Image" src="https://github.com/user-attachments/assets/d3b8d396-1e40-48e5-a14d-d1304e2f0ff0" />

## 🛠 **Installation**
1️⃣ Clone the repo
```bash
git clone https://github.com/MohamedAli1937/Sign-Language-Detection.git
```

2️⃣ Install dependencies
```bash
pip install opencv-python mediapipe scikit-learn numpy tqdm
```
## 📸 **Collecting Your Own Dataset**

Run the script to capture images for each alphabet letter:
```python
python clean_data.py
```


This processes your raw images inside `data/` and generates:
`data.pickle`

## 🧠 **Training the Model**

Train the **RandomForest classifier** using:
```python
python model.py
```

This produces: `model.p`

## 🎥 **Real-Time Hand Recognition**

Launch live prediction with your webcam:
```python
python main.py
```

Controls:
`ESC` → quit

## 🧱 **How It Works (Simplified)**

1️⃣ **MediaPipe** detects the hand

2️⃣ Extract **21 landmarks** → (x,y) → 42 features

3️⃣ **Normalize** landmarks relative to the **minimum x, y** (same during training + testing)

4️⃣ **RandomForest** predicts a class 0–25 → mapped to A–Z

## 📌 **Requirements**

**Python 3.8+** & **Webcam** & **Good lighting for best performance**

## 🙌 **Future Improvements**

- Add smoothing filter to stabilize predictions

- Add gesture recording for custom signs

- Convert to CNN for higher accuracy

- Build a simple Tkinter or web UI
