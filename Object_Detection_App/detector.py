import torch
import cv2
from ultralytics import YOLO
import sys


class YOLODetector:
    def __init__(self, model_path=None):
        """Initialize the YOLO object detector"""
        print("Loading model...")
        try:
            # Load a pre-trained YOLOv5 model from torch hub
            self.model = torch.hub.load(
                "ultralytics/yolov5", "yolov5s", pretrained=True
            )
            print("Model loaded successfully!")

            # Set model to evaluation mode
            self.model.eval()

            # COCO dataset class names
            self.classes = self.model.names
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    def detect(self, image_path, conf_threshold=0.5):
        """
        Detect objects in an image

        Args:
            image_path: Path to the image or numpy array
            conf_threshold: Confidence threshold for detections

        Returns:
            Tuple of (processed image with bounding boxes, list of detections)
        """
        # Perform inference
        results = self.model(image_path)

        # Process results
        predictions = (
            results.xyxy[0].cpu().numpy()
        )  # xyxy format: x1, y1, x2, y2, confidence, class

        # Filter by confidence
        predictions = predictions[predictions[:, 4] >= conf_threshold]

        return self._process_results(image_path, predictions)

    def _process_results(self, image_path, predictions):
        """Process detection results and draw bounding boxes"""
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # If image is already a numpy array
            image = image_path

        # Create a copy for drawing
        drawn_image = image.copy()

        detections = []

        # Draw bounding boxes and labels
        for x1, y1, x2, y2, conf, cls in predictions:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = int(cls)
            class_name = self.classes[class_id]

            color = (0, 255, 0)  # Green color for bounding box

            # Draw rectangle
            cv2.rectangle(drawn_image, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            text = f"{class_name}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            (text_width, text_height) = cv2.getTextSize(
                text, font, font_scale, thickness
            )[0]
            cv2.rectangle(
                drawn_image,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1,
            )

            # Draw text
            cv2.putText(
                drawn_image, text, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness
            )

            # Add to detections list
            detections.append(
                {
                    "class": class_name,
                    "confidence": float(conf),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
            )

        return drawn_image, detections
