import os
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_crop_faces(image_path, index):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        index += 1

    # Crop and save each face
    for i, (x, y, w, h) in enumerate(faces):
        # Calculating the dimensions to expand the cropped region
        expand_x = (112 - w) // 2
        expand_y = (112 - h) // 2
        # Adjusting to include nearby pixels
        x_start = max(0, x - expand_x)
        y_start = max(0, y - expand_y)
        x_end = min(image.shape[1], x + w + expand_x)
        y_end = min(image.shape[0], y + h + expand_y)

        cropped_face = image[y_start:y_end, x_start:x_end]

        output_path = os.path.join('train', f'image_{index}.jpg')
        cropped_face = cv2.GaussianBlur(cropped_face, (0, 0), sigmaX=2)
        cropped_face = cv2.resize(cropped_face, (112, 112))
        cv2.imwrite(output_path, cropped_face)


    return index

train_folder = 'test-full-faces'
index = 0
for filename in os.listdir(train_folder):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        image_path = os.path.join(train_folder, filename)
        index = detect_and_crop_faces(image_path, index)
