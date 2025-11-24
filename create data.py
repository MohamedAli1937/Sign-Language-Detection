import os
import cv2

# Create dataset folder if it doesn't exist
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Dataset parameters
number_of_classes = 26  # Number of classes (A-Z)
dataset_size = 100      # Number of images per class

# Initialize webcam
cap = cv2.VideoCapture(0)

# Loop over each class
for j in range(number_of_classes):

    # Create folder for the class if it doesn't exist
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    # ----------------------------------------
    # Wait for user to get ready
    # ----------------------------------------
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Press "Q" to start',
                    (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 0, 255),
                    3,
                    )
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) == ord('q'):
            break

    # ----------------------------------------
    # Capture images for the current class
    # ----------------------------------------
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        # Save image to class folder
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

# --------------------------------------------
# Release resources
# --------------------------------------------
cap.release()
cv2.destroyAllWindows()