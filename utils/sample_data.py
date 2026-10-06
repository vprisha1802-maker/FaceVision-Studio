"""
Sample Data Generator & Manager for Face Vision Studio.
Generates realistic sample portrait faces for instant testing without manual file uploads.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw


def generate_sample_portraits(output_dir: str):
    """
    Generate synthetic portrait faces with clear facial landmarks (eyes, smile, contours)
    to enable instant demonstration of Viola-Jones, Template Matching, DeepFace & FaceNet.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    samples = [
        {
            "filename": "sample_happy.png",
            "name": "Sample 1: Happy Portrait",
            "skin_color": (230, 195, 175),
            "hair_color": (45, 30, 20),
            "eye_color": (35, 75, 120),
            "expression": "happy",
            "bg_color": (30, 41, 59)
        },
        {
            "filename": "sample_surprised.png",
            "name": "Sample 2: Surprised Expression",
            "skin_color": (240, 205, 185),
            "hair_color": (160, 90, 40),
            "eye_color": (30, 95, 45),
            "expression": "surprised",
            "bg_color": (40, 30, 60)
        },
        {
            "filename": "sample_neutral.png",
            "name": "Sample 3: Professional Neutral",
            "skin_color": (215, 175, 150),
            "hair_color": (25, 25, 30),
            "eye_color": (50, 40, 30),
            "expression": "neutral",
            "bg_color": (20, 35, 45)
        }
    ]

    for item in samples:
        path = os.path.join(output_dir, item["filename"])
        if os.path.exists(path):
            continue

        width, height = 480, 560
        img = Image.new("RGB", (width, height), item["bg_color"])
        draw = ImageDraw.Draw(img)

        # Draw hair background
        draw.ellipse([100, 70, 380, 420], fill=item["hair_color"])
        
        # Neck
        draw.rectangle([200, 330, 280, 470], fill=(int(item["skin_color"][0] * 0.9), 
                                                   int(item["skin_color"][1] * 0.9), 
                                                   int(item["skin_color"][2] * 0.9)))
        
        # Shoulders / Clothes
        draw.ellipse([80, 440, 400, 700], fill=(70, 80, 110))

        # Face Oval
        face_box = [130, 110, 350, 400]
        draw.ellipse(face_box, fill=item["skin_color"])

        # Hair top & bangs
        draw.arc([115, 70, 365, 230], start=180, end=360, fill=item["hair_color"], width=30)
        draw.chord([120, 80, 360, 190], start=190, end=350, fill=item["hair_color"])

        # Eyebrows
        if item["expression"] == "surprised":
            draw.arc([165, 180, 225, 215], start=200, end=340, fill=(40, 30, 20), width=4)
            draw.arc([255, 180, 315, 215], start=200, end=340, fill=(40, 30, 20), width=4)
        else:
            draw.arc([165, 200, 225, 230], start=200, end=340, fill=(40, 30, 20), width=5)
            draw.arc([255, 200, 315, 230], start=200, end=340, fill=(40, 30, 20), width=5)

        # Eyes (Whites)
        draw.ellipse([170, 218, 220, 248], fill=(255, 255, 255), outline=(60, 50, 40), width=2)
        draw.ellipse([260, 218, 310, 248], fill=(255, 255, 255), outline=(60, 50, 40), width=2)
        
        # Eye Iris
        draw.ellipse([185, 222, 205, 244], fill=item["eye_color"])
        draw.ellipse([275, 222, 295, 244], fill=item["eye_color"])
        
        # Pupils
        draw.ellipse([191, 228, 199, 238], fill=(10, 10, 10))
        draw.ellipse([281, 228, 289, 238], fill=(10, 10, 10))
        
        # Eye light reflection
        draw.ellipse([188, 225, 192, 229], fill=(255, 255, 255))
        draw.ellipse([278, 225, 282, 229], fill=(255, 255, 255))

        # Nose bridge and tip
        draw.line([240, 225, 238, 280], fill=(190, 150, 130), width=3)
        draw.arc([230, 270, 250, 290], start=20, end=160, fill=(170, 130, 110), width=3)
        draw.arc([224, 276, 234, 288], start=80, end=200, fill=(180, 140, 120), width=2)
        draw.arc([246, 276, 256, 288], start=-20, end=100, fill=(180, 140, 120), width=2)

        # Mouth depending on expression
        if item["expression"] == "happy":
            # Big smile
            draw.chord([200, 300, 280, 355], start=0, end=180, fill=(190, 40, 60), outline=(140, 30, 45), width=2)
            draw.chord([208, 302, 272, 325], start=0, end=180, fill=(250, 250, 250)) # Teeth
        elif item["expression"] == "surprised":
            # Open 'O' mouth
            draw.ellipse([225, 308, 255, 350], fill=(140, 30, 40), outline=(100, 20, 30), width=2)
        else:
            # Neutral gentle lips
            draw.line([210, 325, 270, 325], fill=(160, 60, 70), width=4)
            draw.arc([215, 320, 265, 335], start=10, end=170, fill=(180, 80, 90), width=3)

        # Save to disk
        img.save(path, format="PNG")


def get_available_samples(samples_dir: str):
    """Retrieve list of sample images."""
    generate_sample_portraits(samples_dir)
    sample_files = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return [os.path.join(samples_dir, f) for f in sample_files]
