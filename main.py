import cv2
import torch
from utils.preprocess import preprocess_image

# Load YOLOv7 model and set it to detect license plates only
model = torch.hub.load('WongKinYiu/yolov7', 'yolov7', pretrained=True)
model.classes = [license_plate_class_index]  # Replace with actual class index for plates
model.eval()  # Set the model to evaluation mode

def detect_and_display(model, video_source=0):
    """
    Capture video, run detection on each frame, and display results in real-time.
    
    Args:
        model (torch.nn.Module): YOLOv7 model for license plate detection.
        video_source (int/str): Video source (0 for default camera or path to video).

    Returns:
        None
    """
    # Initialize video capture (0 for default camera)
    cap = cv2.VideoCapture(video_source)
    
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        img_tensor = preprocess_image(frame)

        # Get model predictions
        results = model(img_tensor)

        # Draw bounding boxes on detected license plates
        for *xyxy, conf, cls in results.pred[0]:
            x1, y1, x2, y2 = map(int, xyxy)  # Bounding box coordinates
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw box

            # Display confidence score for detected plate
            label = f"Plate: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the frame with detections
        cv2.imshow("License Plate Detection", frame)

        # Exit if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close display window
    cap.release()
    cv2.destroyAllWindows()

# Run real-time detection
detect_and_display(model)
