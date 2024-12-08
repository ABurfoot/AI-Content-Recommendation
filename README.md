## Overview
This project is a comprehensive content recommendation tool designed to provide users with personalized recommendations for images and videos. It leverages deep learning models to extract features from multimedia content and uses these features to offer tailored suggestions based on user preferences.

## Features
- Image and Video Support: Processes both image and video content.
- Efficient Feature Extraction: Utilizes EfficientNet for high-quality feature extraction.
- Personalized Recommendations: Adapts to user preferences through continuous learning.
- Resource Management: Includes intelligent memory and resource management for optimal performance.
- Interactive GUI: User-friendly interface with features such as video playback, likes, and skips.

## Requirements
Python 3.7+
PyTorch
torchvision
PIL (Pillow)
numpy
psutil
OpenCV (cv2)
tkinter

## Installation
1. Clone this repository: git clone https://github.com/ABurfoot/Deep-Learning-Content-Recommendation.git
2. Install the required packages:
   pip install torch
   pip install torchvision
   pip install Pillow
   pip install numpy
   pip install psutil
   pip install opencv-python

## Usage
1. Ensure your content (images and videos) is placed in a directory (e.g., content).
2. Run the program, replace <path_to_content_directory> with the path to your content folder: python main.py --content_path <path_to_content_directory>
3. Interact with the GUI to view, like, or skip recommended content.

## How it Works
### Data Processing
Content Loading: Scans the content directory for supported file types (.jpg, .png, .mp4, .avi, etc.).
Feature Extraction:
- Images: Processes images through EfficientNet to extract high-dimensional features.
- Videos: Samples frames from videos and processes them similarly.

### Recommendations
- User Preferences: Builds a profile based on user interactions (likes and dislikes).
- Similarity Scoring: Calculates cosine similarity between user preferences and available content.
- Exploration Strategy: Provides random recommendations when user preferences are insufficient.

### GUI Features
- Interactive Buttons: Like, skip, or dislike content.
- Video Playback: Supports frame-based video playback with controls.
- Dynamic Recommendations: Continuously updates the recommendation queue based on user activity.

## Key Components
EfficientNetFeatureExtractor: Extracts features from image content.
VideoFeatureExtractor: Processes video frames and applies temporal pooling for feature extraction.
RecommendationSystem:
- Manages content processing and feature extraction.
- Handles user preferences and provides recommendations.
ContentRecommenderGUI: Interactive GUI for displaying and interacting with recommendations.

## Limitations
- The stock market is inherently unpredictable and affected by many external factors.
- This tool should be used for educational purposes only and not for actual trading decisions.
- The accuracy of predictions can vary significantly depending on the stock and market conditions.

## Disclaimer
This software is for educational and research purposes only. It is not intended for commercial use. The creator Alexander Burfoot is not responsible for any outcomes resulting from the use of this tool.













