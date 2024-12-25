import cv2
import os
import numpy as np
import torch
import mediapipe as mp

class PoseKeypointsTransformer:
    def __init__(self, with_visibility=False):
        self.with_visibility = with_visibility
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

    def __call__(self, image_path, visualize=False):
        image = cv2.imread(os.path.join("..", image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with self.mp_pose.Pose(min_detection_confidence=0.75, min_tracking_confidence=0.5) as pose:
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    keypoints.extend([landmark.x, landmark.y, landmark.z])
                    if self.with_visibility:
                        keypoints.append(landmark.visibility)

                keypoints = np.array(keypoints, dtype=np.float32)
                keypoints_tensor = torch.tensor(keypoints)

            else:
                keypoints_tensor = torch.empty(0)

            if visualize:
                self.mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
                cv2.imshow("Pose Detection", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        return keypoints_tensor