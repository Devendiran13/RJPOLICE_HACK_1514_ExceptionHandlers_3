import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from tensorflow.keras.models import load_model
from google.colab.patches import cv2_imshow

Mobilenet_model = MobileNetV2(weights='imagenet')
resnet50_model = ResNet50(weights='imagenet')

def preprocess_frame_mobilenet(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = mobilenet_preprocess_input(frame)
    return np.expand_dims(frame, axis=0)

def preprocess_frame_resnet(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = resnet_preprocess_input(frame)
    return np.expand_dims(frame, axis=0)

def detect_objects_mobilenet(frame):
    preprocessed_frame = preprocess_frame_mobilenet(frame)
    prediction = Mobilenet_model.predict(preprocessed_frame)
    return prediction

def detect_objects_resnet(frame):
    preprocessed_frame = preprocess_frame_resnet(frame)
    prediction = resnet50_model.predict(preprocessed_frame)
    return prediction

def is_suspicious(predictions_mobilenet, predictions_resnet, threshold=0.8):
    return np.max(predictions_mobilenet) > threshold or np.max(predictions_resnet) > threshold

def send_alert():
    print("Alert: Suspicious activity detected. Notify the police!")

cap = cv2.VideoCapture("/content/videoplayback.mp4")

frame_skip = 5 
frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    frame_count += 1

    if frame_count % frame_skip != 0:
        continue 
    predictions_mobilenet = detect_objects_mobilenet(frame)
    predictions_resnet = detect_objects_resnet(frame)

    if is_suspicious(predictions_mobilenet, predictions_resnet):
        send_alert()

    resized_frame = cv2.resize(frame, (640, 480))  # Resize the frame for display
    from google.colab.patches import cv2_imshow
    cv2_imshow(resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
