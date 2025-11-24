import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --------------------------------------------
# Load Dataset
# --------------------------------------------
with open("./data.pickle", "rb") as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict["data"])
labels = np.asarray(data_dict["labels"])

print(f"Dataset loaded:")
print(f" - Samples: {len(data)}")

# --------------------------------------------
# Train / Test Split
# --------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    data, labels,
    test_size=0.2,
    shuffle=True,
    stratify=labels,
    random_state=42
)

# --------------------------------------------
# RandomForest Model
# --------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    random_state=42,
    n_jobs=-1      # use all cores → faster
)

print("\nTraining model...")
model.fit(x_train, y_train)

# --------------------------------------------
# Evaluation
# --------------------------------------------
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n✅ Model evaluation:")
print(f"Accuracy: {accuracy * 100:.2f}%")

# --------------------------------------------
# Save Model
# --------------------------------------------
with open("model.p", "wb") as f:
    pickle.dump({"model": model}, f)

print("\n📦 Model saved as model.p")
