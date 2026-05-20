import cv2
import numpy as np
import os
import sys
import base64

# Class labels for MobileNet-SSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Colors for each class (BGR format)
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def detect_objects(image_base64, confidence_threshold=0.5):
    """
    Detect objects in an image using MobileNet-SSD

    Args:
        image_base64: Base64 encoded image string
        confidence_threshold: Minimum confidence for detection (0-1)

    Returns:
        Base64 encoded output image with detections
    """
    # Load the model
    print("[INFO] Loading model...")
    net = cv2.dnn.readNetFromCaffe(
        "/deploy.prototxt",
        "/mobilenet_ssd.caffemodel"
    )

    # Decode base64 image
    print("[INFO] Decoding base64 image...")
    try:
        img_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            print("[ERROR] Could not decode image")
            return []
    except Exception as e:
        print(f"[ERROR] Failed to decode base64 image: {e}")
        return []

    (h, w) = image.shape[:2]

    # Prepare the image for detection
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        0.007843,
        (300, 300),
        127.5
    )

    # Pass the blob through the network
    print("[INFO] Running detection...")
    net.setInput(blob)
    detections = net.forward()

    person_boxes = []

    # Loop over the detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = CLASSES[idx]
            print(f"[DETECTED] {label}: {confidence:.2f} at [{startX},{startY},{endX},{endY}]")
            if label == "person":
                person_boxes.append(f"{startX},{startY},{endX},{endY}")

    return person_boxes


def handler(params, context):
    if not "img" in params:
        return {}
    img = params["img"]

    response = {}

    person_boxes = detect_objects(img)

    response["Count"] = len(person_boxes)
    response["Detections"] = person_boxes
    response["Img"] = img

    return response