import cv2
import os
import numpy as np
import torch
import mediapipe as mp
import matplotlib.pyplot as plt

def get_bounding_box(image_path):
    mp_pose = mp.solutions.pose
    image = cv2.imread(os.path.join("..", image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(min_detection_confidence=0.75) as pose:
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            x_coords = [landmark.x for landmark in results.pose_landmarks.landmark]
            y_coords = [landmark.y for landmark in results.pose_landmarks.landmark]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min
            
            return np.array([x_center, y_center, width, height])
        else:
            return None
        
def visualize_bbox(image_path, bbox, label=None):
    image = cv2.imread(os.path.join("..", image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_height, image_width = image.shape[:2]
    
    x_center, y_center, width, height = bbox
    x_center *= image_width
    y_center *= image_height
    width *= image_width
    height *= image_height
    
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)
    
    image_with_bbox = cv2.rectangle(
        image.copy(), 
        (x_min, y_min), 
        (x_max, y_max), 
        color=(0, 255, 0),  
        thickness=2
    )
    
    if label:
        cv2.putText(
            image_with_bbox, 
            label, 
            (x_min, y_min - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 255, 0), 
            1, 
            cv2.LINE_AA
        )
    
    image_with_bbox = cv2.cvtColor(image_with_bbox, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image_with_bbox)
    plt.axis("off")
    plt.show()