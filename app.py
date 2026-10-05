import cv2
from ultralytics import YOLO


# Load YOLOv8 Nano model
model = YOLO("yolov8n.pt")


# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


print("Webcam started.")
print("Press Q to exit.")


while True:

    # Read frame from webcam
    success, frame = cap.read()

    if not success:
        print("Error: Could not read frame.")
        break

    # Run YOLO object detection
    results = model(frame)

    # Process detection results
    for result in results:

        # Get detected boxes
        boxes = result.boxes

        for box in boxes:

            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Convert coordinates to integers
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            # Get confidence score
            confidence = float(box.conf[0])

            # Get class ID
            class_id = int(box.cls[0])

            # Get class name
            class_name = model.names[class_id]

            # Create label
            label = f"{class_name} {confidence:.2f}"

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Calculate label size
            (label_width, label_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )

            # Draw label background
            cv2.rectangle(
                frame,
                (x1, y1 - label_height - baseline),
                (x1 + label_width, y1),
                (0, 255, 0),
                -1
            )

            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

    # Display live detection result
    cv2.imshow("YOLOv8 Real-Time Object Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release webcam
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print("Application closed.")