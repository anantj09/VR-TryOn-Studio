import os
import cv2
import numpy as np
from typing import List, Dict, Optional

class SegmentationService:
    def __init__(self):
        self.predictor = None
        self.model_type = "vit_h"
        # Resolve path to weights/sam_vit_h_4b8939.pth relative to this service
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.checkpoint_path = os.path.join(current_dir, "..", "weights", "sam_vit_h_4b8939.pth")

    def generate_mask(self, image_path: str, landmarks: List[Dict[str, float]]) -> Optional[str]:
        """
        Generates a high-quality binary silhouette mask for the human pose.
        Uses pose landmarks as point prompts to guide the geometric polygon mask generation.
        Saves the binary mask in the same temp directory and returns its filename.
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            h, w, c = image.shape
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Head Circle based on nose & ears
            if len(landmarks) > 0:
                nose = landmarks[0]
                l_ear = landmarks[7] if len(landmarks) > 7 else nose
                r_ear = landmarks[8] if len(landmarks) > 8 else nose
                head_center_x = int(nose["x"] * w)
                head_center_y = int(nose["y"] * h)
                head_radius = int(np.sqrt((l_ear["x"] - r_ear["x"])**2 + (l_ear["y"] - r_ear["y"])**2) * w * 1.3)
                if head_radius <= 0:
                    head_radius = int(h * 0.09)
                cv2.circle(binary_mask, (head_center_x, head_center_y), head_radius, 255, -1)

            # Neck, shoulders, hips, knees, ankles, wrists polygon
            poly_points = []
            
            # Left Arm: Left Shoulder (11) -> Left Elbow (13) -> Left Wrist (15)
            for idx in [11, 13, 15]:
                if idx < len(landmarks):
                    poly_points.append([int(landmarks[idx]["x"] * w), int(landmarks[idx]["y"] * h)])
            
            # Left Leg: Left Hip (23) -> Left Knee (25) -> Left Ankle (27) -> Left Heel (29) -> Left Foot Index (31)
            for idx in [23, 25, 27, 29, 31]:
                if idx < len(landmarks):
                    poly_points.append([int(landmarks[idx]["x"] * w), int(landmarks[idx]["y"] * h)])
                    
            # Right Leg: Right Foot Index (32) -> Right Heel (30) -> Right Ankle (28) -> Right Knee (26) -> Right Hip (24)
            for idx in [32, 30, 28, 26, 24]:
                if idx < len(landmarks):
                    poly_points.append([int(landmarks[idx]["x"] * w), int(landmarks[idx]["y"] * h)])
                    
            # Right Arm: Right Wrist (16) -> Right Elbow (14) -> Right Shoulder (12)
            for idx in [16, 14, 12]:
                if idx < len(landmarks):
                    poly_points.append([int(landmarks[idx]["x"] * w), int(landmarks[idx]["y"] * h)])

            if len(poly_points) > 2:
                pts = np.array(poly_points, dtype=np.int32)
                cv2.fillPoly(binary_mask, [pts], 255)
                
            # Post-process binary mask to make it smooth and seamless
            kernel = np.ones((5, 5), np.uint8)
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=3)
            binary_mask = cv2.GaussianBlur(binary_mask, (5, 5), 0)

            # Save binary mask
            file_dir, file_name = os.path.split(image_path)
            base_name, _ = os.path.splitext(file_name)
            mask_filename = f"{base_name}_mask.png"
            mask_path = os.path.join(file_dir, mask_filename)
            cv2.imwrite(mask_path, binary_mask)
            return mask_filename
        except Exception as e:
            print(f"[WARNING] Fallback mask generation failed: {str(e)}")
            return None

# Global instance of SegmentationService
segmentation_service = SegmentationService()
