import cv2
import os
import requests

# ---------------------------
# Folder where script & models will be stored
# ---------------------------
base_path = r"C:\Users\hp\Desktop\VirtualAssistant"
os.makedirs(base_path, exist_ok=True)

# ---------------------------
# Model file paths
# ---------------------------
prototxt_path = os.path.join(base_path, "deploy.prototxt")
model_path = os.path.join(base_path, "res10_300x300_ssd_iter_140000.caffemodel")

# ---------------------------
# URLs to download models if missing
# ---------------------------
prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

# ---------------------------
# Function to download files
# ---------------------------
def download_file(url, path):
    print(f"Downloading {os.path.basename(path)}...")
    response = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"{os.path.basename(path)} downloaded successfully!")

# ---------------------------
# Check and download missing or corrupted files
# ---------------------------
def check_file(path, url):
    try:
        if not os.path.exists(path) or os.path.getsize(path) < 1000:  # simple size check
            download_file(url, path)
    except:
        download_file(url, path)

check_file(prototxt_path, prototxt_url)
check_file(model_path, model_url)

# ---------------------------
# Load the DNN model
# ---------------------------
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# ---------------------------
# Start webcam
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open webcam")

# Set higher webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    face_count = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.1:  # Lower threshold
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{confidence*100:.2f}%"
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv2.putText(frame, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            face_count += 1

    if face_count > 0:
        print(f"{face_count} face(s) detected!")

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")
