import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import os
import random
import numpy as np
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from collections import OrderedDict
from typing import Optional, List, Tuple
import logging
import sys
from pathlib import Path
import gc
import psutil
import cv2

# Create the logging system
def setup_logging():
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Generate log filename with timestamp
    log_file = f"logs/recommendation_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure logging format and handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

# Create logger instance
logger = setup_logging()

# Manages system resources and memory usage, handling cleanup operations and memory monitoring
class ResourceManager:
    def __init__(self, memory_threshold: float = 90.0):
        self.memory_threshold = memory_threshold
        self.running = True
        logger.info("Initializing Resource Manager")
    
    # Get current memory usage percentage
    def get_memory_usage(self) -> float:
        return psutil.Process().memory_percent()
    
    # Check if memory usage is above threshold
    def check_memory(self) -> bool:
        return self.get_memory_usage() > self.memory_threshold
    
    # Perform garbage collection and memory cleanup
    def cleanup(self):
        try:
            # Force garbage collection
            gc.collect()
            
            # Clear memory caches if needed
            if self.check_memory():
                logger.warning("High memory usage detected, clearing caches")
                return True
            return False
        except Exception as e:
            logger.error("Error during resource cleanup: %s", str(e), exc_info=True)
            return False
    
    # Clean up resources on shutdown
    def shutdown(self):
        logger.info("Shutting down Resource Manager")
        self.running = False
        self.cleanup()

# Handles video processing operations including frame extraction and feature computation
class VideoProcessor:
    def __init__(self, max_cache_size: int = 50, total_frames_to_sample: int = 10):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        self.frame_cache = OrderedDict()
        self.max_cache_size = max_cache_size
        self.total_frames_to_sample = total_frames_to_sample
        self.resource_manager = ResourceManager()
        logger.info("Initializing Video Processor...")

    # Clear the video frame cache
    def clear_cache(self):
        try:
            self.frame_cache.clear()
            self.resource_manager.cleanup()
        except Exception as e:
            logger.error(f"Error clearing video cache: {str(e)}")

    # Extracts evenly spaced frames across the video duration
    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame indices to sample
            if total_frames <= self.total_frames_to_sample:
                # If video is short, just take all frames
                frame_indices = range(total_frames)
            else:
                # Calculate evenly spaced frame indices
                step = total_frames // self.total_frames_to_sample
                frame_indices = range(0, total_frames, step)[:self.total_frames_to_sample]

            frames = []
            current_frame = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if current_frame in frame_indices:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)

                    if len(frames) >= self.total_frames_to_sample:
                        break

                current_frame += 1

                # Memory check periodically
                if current_frame % 30 == 0 and self.resource_manager.check_memory():
                    self.clear_cache()

            cap.release()
            return frames

        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {str(e)}")
            return []

    # Processes sampled video frames
    def preprocess_video(self, video_path: str) -> Optional[torch.Tensor]:
        try:
            if video_path in self.frame_cache:
                self.frame_cache.move_to_end(video_path)
                return self.frame_cache[video_path]

            # Extract frames
            frames = self.extract_frames(video_path)
            if not frames:
                return None

            # Process frames
            processed_frames = []
            for frame in frames:
                image = Image.fromarray(frame)
                tensor = self.transform(image).unsqueeze(0)
                processed_frames.append(tensor)

            if not processed_frames:
                return None

            # Combine frames into single tensor
            video_tensor = torch.cat(processed_frames, dim=0)

            # Update cache
            if len(self.frame_cache) >= self.max_cache_size:
                self.frame_cache.popitem(last=False)
            self.frame_cache[video_path] = video_tensor

            return video_tensor

        except Exception as e:
            logger.error(f"Error preprocessing video {video_path}: {str(e)}")
            return None

    # Gets video metadata including frame count, fps, etc.
    def get_video_info(self, video_path: str) -> Optional[dict]:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None

            info = {
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
            }
            cap.release()
            return info

        except Exception as e:
            logger.error(f"Error getting video info for {video_path}: {str(e)}")
            return None
    
class ImageProcessor:
    def __init__(self, max_cache_size: int = 100):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        self.image_cache = OrderedDict()
        self.max_cache_size = max_cache_size
        self.resource_manager = ResourceManager()
    
    # Clear the image cache
    def clear_cache(self):
        self.image_cache.clear()
        self.resource_manager.cleanup()

    # Processes images for neural network input with caching    
    def preprocess_image(self, image_path: str) -> Optional[torch.Tensor]:
        # Check if resources need to be cleaned up
        if self.resource_manager.check_memory():
            logger.warning("Memory usage high, clearing image cache")
            self.clear_cache()
            
        if image_path in self.image_cache:
            self.image_cache.move_to_end(image_path)
            return self.image_cache[image_path]
        
        try:
            image = Image.open(image_path).convert('RGB')
            tensor = self.transform(image).unsqueeze(0)
            
            if len(self.image_cache) >= self.max_cache_size:
                self.image_cache.popitem(last=False)
            
            self.image_cache[image_path] = tensor
            return tensor
        except Exception as e:
            logger.error("Error preprocessing image %s: %s", image_path, str(e), exc_info=True)
            return None

# Neural network feature extractor using EfficientNet architecture for image feature extraction
class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(EfficientNetFeatureExtractor, self).__init__()
        logger.info("Initializing EfficientNet Feature Extractor...")
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-1])
        
        self.feature_cache = OrderedDict()
        self.max_cache_size = 1000

    # Performs the forward pass through the neural network to extract image features
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.efficientnet(x).squeeze()

    # Stores extracted features in cache to avoid recomputing for frequently accessed images
    def cache_features(self, image_path: str, features: torch.Tensor):
        if len(self.feature_cache) >= self.max_cache_size:
            self.feature_cache.popitem(last=False)
        self.feature_cache[image_path] = features.cpu()

    # Retrieves previously cached features for an image if available
    def get_cached_features(self, image_path: str) -> Optional[torch.Tensor]:
        if image_path in self.feature_cache:
            self.feature_cache.move_to_end(image_path)
            return self.feature_cache[image_path]
        return None

# Video feature extraction using EfficientNet with temporal processing
class VideoFeatureExtractor(nn.Module):
    def __init__(self):
        super(VideoFeatureExtractor, self).__init__()
        logger.info("Initializing Video Feature Extractor...")
        
        # Load base model
        self.efficientnet = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-1])
        
        # Add temporal processing layers
        self.temporal_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_pool = nn.AdaptiveAvgPool1d(1)
        
        # Initialize caching
        self.feature_cache = OrderedDict()
        self.max_cache_size = 1000
        self.resource_manager = ResourceManager()

    # Process frames through the network
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Process each frame
        frame_features = []
        for i in range(batch_size):
            # Extract frame features
            frame = x[i].unsqueeze(0)
            features = self.efficientnet(frame)
            frame_features.append(features)
        
        # Combine frame features
        frame_features = torch.cat(frame_features, dim=0)
        
        # Temporal pooling
        pooled_features = self.temporal_pool(frame_features)
        
        # Final feature pooling
        video_features = self.feature_pool(
            pooled_features.squeeze(-1).squeeze(-1).unsqueeze(0)
        )
        
        return video_features.squeeze()

    # Store video features in cache
    def cache_features(self, video_path: str, features: torch.Tensor):
        try:
            if len(self.feature_cache) >= self.max_cache_size:
                self.feature_cache.popitem(last=False)
            self.feature_cache[video_path] = features.cpu()
        except Exception as e:
            logger.error(f"Error caching video features: {str(e)}")

    # Retrieve cached video features
    def get_cached_features(self, video_path: str) -> Optional[torch.Tensor]:
        try:
            if video_path in self.feature_cache:
                self.feature_cache.move_to_end(video_path)
                return self.feature_cache[video_path]
            return None
        except Exception as e:
            logger.error(f"Error retrieving cached features: {str(e)}")
            return None

    # Clear feature cache
    def clear_cache(self):
        try:
            self.feature_cache.clear()
            if self.resource_manager.check_memory():
                self.resource_manager.cleanup()
        except Exception as e:
            logger.error(f"Error clearing feature cache: {str(e)}")
    
class RecommendationSystem:
    def __init__(self, content_directory: str):
        logger.info("=== Initializing Recommendation System ===")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)
        
        # Initialize components
        self.resource_manager = ResourceManager()
        self.image_processor = ImageProcessor(max_cache_size=100)
        self.video_processor = VideoProcessor(max_cache_size=50)
        
        # Initialize feature extractors
        self.image_feature_extractor = EfficientNetFeatureExtractor().to(self.device)
        self.video_feature_extractor = VideoFeatureExtractor().to(self.device)
        self.image_feature_extractor.eval()
        self.video_feature_extractor.eval()
        
        # Data storage
        self.content_directory = content_directory
        self.content_features = {}
        self.content_types = {}  # Tracks whether content is image or video
        self.user_preferences = {}
        
        logger.info("Starting content processing...")
        self._extract_content_features()

    # Cleans up system resources and caches    
    def cleanup_resources(self):
        try:
            # Clear caches
            self.image_processor.clear_cache()
            self.video_processor.clear_cache()
            if hasattr(self.image_feature_extractor.feature_cache, 'clear'):
                self.image_feature_extractor.feature_cache.clear()
            self.video_feature_extractor.clear_cache()
            
            # Clear CUDA cache if using GPU
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Run garbage collection
            self.resource_manager.cleanup()
            
        except Exception as e:
            logger.error("Error during resource cleanup: %s", str(e), exc_info=True)

    # Extracts features from all content in the directory    
    def _extract_content_features(self):
        # Log directory contents for debugging
        try:
            all_files = os.listdir(self.content_directory)
            logger.info(f"Directory contents: {all_files}")
        except Exception as e:
            logger.error(f"Error reading directory: {str(e)}")

        # Get all supported files
        files = [f for f in os.listdir(self.content_directory) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov'))]
        
        total_files = len(files)
        logger.info(f"Found {total_files} supported files: {files}")
        
        processed_count = 0
        for filename in files:
            filepath = os.path.join(self.content_directory, filename)
            try:
                # Check memory and cleanup if needed
                if self.resource_manager.check_memory():
                    logger.warning("High memory usage, cleaning up...")
                    self.cleanup_resources()
                
                # Determine content type
                is_video = filename.lower().endswith(('.mp4', '.avi', '.mov'))
                self.content_types[filename] = 'video' if is_video else 'image'
                
                if is_video:
                    # Process video content
                    cached_features = self.video_feature_extractor.get_cached_features(filepath)
                    if cached_features is not None:
                        self.content_features[filename] = cached_features.numpy()
                        processed_count += 1
                        continue
                    
                    # Process new video
                    video_tensor = self.video_processor.preprocess_video(filepath)
                    if video_tensor is not None:
                        video_tensor = video_tensor.to(self.device)
                        with torch.no_grad():
                            features = self.video_feature_extractor(video_tensor)
                        self.content_features[filename] = features.cpu().numpy()
                        self.video_feature_extractor.cache_features(filepath, features)
                        processed_count += 1
                else:
                    # Process image content
                    cached_features = self.image_feature_extractor.get_cached_features(filepath)
                    if cached_features is not None:
                        self.content_features[filename] = cached_features.numpy()
                        processed_count += 1
                        continue
                    
                    # Process new image
                    tensor = self.image_processor.preprocess_image(filepath)
                    if tensor is not None:
                        tensor = tensor.to(self.device)
                        with torch.no_grad():
                            features = self.image_feature_extractor(tensor)
                        self.content_features[filename] = features.cpu().numpy()
                        self.image_feature_extractor.cache_features(filepath, features)
                        processed_count += 1
                
                if processed_count % 10 == 0:
                    logger.info("Processed %d/%d files", processed_count, total_files)
                    
            except Exception as e:
                logger.error("Error processing %s: %s", filename, str(e), exc_info=True)
                continue
        
        logger.info("Successfully processed %d files", processed_count)

    # Updates user preferences based on interactions    
    def update_user_preferences(self, user_id: str, content_id: str, enjoyed: bool):
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        self.user_preferences[user_id][content_id] = enjoyed
        
        # Clean up resources if needed
        if self.resource_manager.check_memory():
            self.cleanup_resources()

    # Gets personalized recommendations for a user    
    def get_recommendation(self, user_id: str, num_recommendations: int = 5) -> List[str]:
        try:
            if user_id not in self.user_preferences or len(self.user_preferences[user_id]) < 3:
                return self._exploration_strategy(num_recommendations)
            
            try:
                user_profile = self._compute_user_profile(user_id)
                similarities = {}
                
                for content_id, features in self.content_features.items():
                    if content_id not in self.user_preferences[user_id]:
                        # Ensure features match user profile shape
                        flat_features = features.flatten()
                        
                        # Adjust lengths if necessary
                        min_length = min(len(user_profile), len(flat_features))
                        user_profile_adj = user_profile[:min_length]
                        flat_features_adj = flat_features[:min_length]
                        
                        # Calculate cosine similarity with zero division protection
                        norm_user = np.linalg.norm(user_profile_adj)
                        norm_features = np.linalg.norm(flat_features_adj)
                        
                        if norm_user > 0 and norm_features > 0:
                            similarity = np.dot(user_profile_adj, flat_features_adj) / (norm_user * norm_features)
                            similarities[content_id] = float(similarity)
                        else:
                            similarities[content_id] = 0.0
                
                if not similarities:
                    return self._exploration_strategy(num_recommendations)
                
                # Sort by similarity and get top recommendations
                recommendations = sorted(
                    similarities.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:num_recommendations]
                
                recommended_items = [content_id for content_id, _ in recommendations]
                
                # If not enough recommendations, pad with random items
                if len(recommended_items) < num_recommendations:
                    available_items = [item for item in self.content_features.keys() 
                                    if item not in recommended_items and 
                                    item not in self.user_preferences[user_id]]
                    if available_items:
                        random_items = random.sample(available_items, 
                                                min(num_recommendations - len(recommended_items), 
                                                    len(available_items)))
                        recommended_items.extend(random_items)
                
                return recommended_items
                
            except Exception as e:
                logger.error(f"Error in recommendation calculation: {str(e)}")
                return self._exploration_strategy(num_recommendations)
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return self._exploration_strategy(num_recommendations)
    
    # Computes user profile based on liked content    
    def _compute_user_profile(self, user_id: str) -> np.ndarray:
        try:
            enjoyed_contents = [
                content_id for content_id, enjoyed in self.user_preferences[user_id].items() 
                if enjoyed
            ]
            
            if not enjoyed_contents:
                # Get the shape from an existing content feature
                sample_feature = next(iter(self.content_features.values()))
                return np.zeros_like(sample_feature).flatten()
            
            # Initialize with the first feature to get the correct shape
            first_feature = self.content_features[enjoyed_contents[0]]
            target_shape = first_feature.shape
            
            # Process each feature to match the target shape
            processed_features = []
            for content_id in enjoyed_contents:
                feature = self.content_features[content_id]
                # Ensure feature has the same shape as the target
                if feature.shape != target_shape:
                    # Reshape feature to match target
                    feature = np.resize(feature, target_shape)
                processed_features.append(feature.flatten())
            
            # Convert to numpy array and compute mean
            features_array = np.array(processed_features)
            profile = np.mean(features_array, axis=0)
            
            return profile
            
        except Exception as e:
            logger.error(f"Error computing user profile: {str(e)}")
            print(f"Debug - Feature shapes: {[self.content_features[c].shape for c in enjoyed_contents]}")
            # Return zero vector with correct dimension
            sample_feature = next(iter(self.content_features.values()))
            return np.zeros_like(sample_feature).flatten()

    # Provides random recommendations for new users    
    def _exploration_strategy(self, num_recommendations: int) -> List[str]:
        available_items = list(self.content_features.keys())
        return random.sample(available_items, min(num_recommendations, len(available_items)))
        
    # Get content type (image or video)
    def get_content_type(self, content_id: str) -> str:
        return self.content_types.get(content_id, 'unknown')

# Main GUI class that handles the visual presentation and user interaction for the content recommendation system
class ContentRecommenderGUI:
    def __init__(self, recommendation_system: RecommendationSystem):
            logger.info("Initializing GUI...")
            self.root = tk.Tk()
            self.root.title("")
            
            self.recommendation_system = recommendation_system
            self.user_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.viewed_content = set()
            self.recommendation_queue = []
            self.current_content = None
            
            # Video playback state
            self.video_playing = False
            self.current_video_cap = None
            self.video_thread = None
            
            # Display cache and resource management
            self.display_cache = OrderedDict()
            self.max_display_cache = 20
            self.resource_manager = ResourceManager()
            
            # Register cleanup on window close
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            self.setup_gui()
    
    # Initializes and configures all GUI elements including the main window, frames, buttons, and controls.
    def setup_gui(self):
            # Configure the main window with larger default size
            self.root.minsize(1024, 768)
            
            # Configure main window grid weights
            self.root.grid_rowconfigure(0, weight=1)
            self.root.grid_columnconfigure(0, weight=1)
            
            # Main frame with padding
            main_frame = ttk.Frame(self.root, padding="20")
            main_frame.grid(row=0, column=0, sticky="nsew")
            
            # Configure main frame grid weights
            main_frame.grid_rowconfigure(0, weight=1)
            main_frame.grid_columnconfigure(1, weight=1)
            
            # Empty columns for centering with less weight
            main_frame.grid_columnconfigure(0, weight=0)
            main_frame.grid_columnconfigure(2, weight=0)
            
            # Content frame - starts with default size but will adjust
            self.content_frame = ttk.Frame(main_frame)
            self.content_frame.grid(row=0, column=1, pady=10)
            
            self.content_label = ttk.Label(self.content_frame)
            self.content_label.place(relx=0.5, rely=0.5, anchor="center")
            
            # Video controls frame - centered
            self.video_controls = ttk.Frame(main_frame)
            self.video_controls.grid(row=1, column=1, pady=5)
            
            # Play button
            self.play_button = ttk.Button(
                self.video_controls,
                text="▶",
                command=self.toggle_video,
                width=5
            )
            self.play_button.pack(side=tk.LEFT, padx=5)
            
            # Progress bar
            self.video_progress = ttk.Progressbar(
                self.video_controls,
                length=300,
                mode='determinate'
            )
            self.video_progress.pack(side=tk.LEFT, padx=5)
            
            # Hide video controls initially
            self.video_controls.grid_remove()
            
            # Interaction buttons - centered
            button_frame = ttk.Frame(main_frame)
            button_frame.grid(row=2, column=1, pady=10)
            
            self.dislike_button = ttk.Button(
                button_frame,
                text="✕",
                command=self.dislike_content,
                width=10
            )
            self.skip_button = ttk.Button(
                button_frame,
                text="↷",
                command=self.skip_content,
                width=10
            )
            self.like_button = ttk.Button(
                button_frame,
                text="♥",
                command=self.like_content,
                width=10
            )
            
            self.dislike_button.pack(side=tk.LEFT, padx=5)
            self.skip_button.pack(side=tk.LEFT, padx=5)
            self.like_button.pack(side=tk.LEFT, padx=5)
            
            # Initialize feed
            self.update_recommendation_queue()
            self.show_next_content()
    
    # Clears display cache and stops any playing video
    def clear_display_cache(self):
        logger.debug("Clearing display cache")
        self.display_cache.clear()
        self.resource_manager.cleanup()
        self.stop_video()

    # Stops video playback and resets controls
    def stop_video(self):
        self.video_playing = False
        if self.current_video_cap is not None:
            self.current_video_cap.release()
            self.current_video_cap = None
        if hasattr(self, 'play_button'):
            self.play_button.configure(text="▶")
            self.video_progress['value'] = 0

    # Toggles video playback state
    def toggle_video(self):
        if self.video_playing:
            self.video_playing = False
            self.play_button.configure(text="▶")
        else:
            self.video_playing = True
            self.play_button.configure(text="⏸")
            self.play_video()

    # Handles video playback and frame updates with proper sizing
    def play_video(self):
        if not self.current_content or not self.video_playing:
            return
            
        content_path = os.path.join(
            self.recommendation_system.content_directory,
            self.current_content
        )
        
        if self.current_video_cap is None:
            self.current_video_cap = cv2.VideoCapture(content_path)
        
        try:
            ret, frame = self.current_video_cap.read()
            if ret:
                # Convert frame to RGB and create PhotoImage
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Use the stored video dimensions
                if hasattr(self, 'video_dimensions'):
                    frame_pil = frame_pil.resize(self.video_dimensions, Image.Resampling.LANCZOS)
                
                # Update display
                photo = ImageTk.PhotoImage(frame_pil)
                self.content_label.configure(image=photo)
                self.content_label.image = photo
                
                # Update progress bar
                total_frames = self.current_video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                current_frame = self.current_video_cap.get(cv2.CAP_PROP_POS_FRAMES)
                progress = (current_frame / total_frames) * 100
                self.video_progress['value'] = progress
                
                if self.video_playing:
                    self.root.after(33, self.play_video)
            else:
                self.current_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.video_playing = False
                self.play_button.configure(text="▶")
                
        except Exception as e:
            logger.error(f"Error during video playback: {str(e)}")
            self.stop_video()

    # Updates the queue of recommended content
    def update_recommendation_queue(self):
        logger.debug("Updating recommendation queue")
        try:
            if self.resource_manager.check_memory():
                self.clear_display_cache()
            
            self.recommendation_queue = []
            
            if len(self.viewed_content) >= 3:
                recommendations = self.recommendation_system.get_recommendation(self.user_id)
                new_recommendations = [r for r in recommendations if r not in self.viewed_content]
                self.recommendation_queue.extend(new_recommendations[:5])
            else:
                all_content = list(self.recommendation_system.content_features.keys())
                available = [c for c in all_content if c not in self.viewed_content]
                if available:
                    random.shuffle(available)
                    self.recommendation_queue.extend(available[:5])
            
            logger.debug("Queue updated with %d items", len(self.recommendation_queue))
            
        except Exception as e:
            logger.error(f"Error updating recommendation queue: {str(e)}", exc_info=True)

    # Shows the next content item
    def show_next_content(self):
        self.disable_buttons()
        logger.debug("Showing next content")
        
        try:
            self.stop_video()
            
            if self.resource_manager.check_memory():
                self.clear_display_cache()
            
            if not self.recommendation_queue:
                self.update_recommendation_queue()
            
            if not self.recommendation_queue:
                logger.warning("No more content available")
                self.show_message("No more content available!")
                self.enable_buttons()
                return
            
            self.current_content = self.recommendation_queue.pop(0)
            content_path = os.path.join(
                self.recommendation_system.content_directory,
                self.current_content
            )
            
            content_type = self.recommendation_system.get_content_type(self.current_content)
            
            # Initialize video capture
            if content_type == 'video':
                cap = cv2.VideoCapture(content_path)
                if not cap.isOpened():
                    logger.error(f"Could not open video: {content_path}")
                    return
                
                # Get video dimensions
                video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Read the first frame for thumbnail
                ret, frame = cap.read()
                if ret:
                    # Convert frame to RGB and create thumbnail
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    
                    # Get window dimensions and resize thumbnail
                    window_width = max(self.root.winfo_width(), 1024)
                    window_height = max(self.root.winfo_height(), 768)
                    
                    # Calculate available space with padding
                    available_width = window_width - 100
                    available_height = window_height - 200
                    
                    # Calculate scaling factors based on video dimensions
                    width_scale = available_width / video_width
                    height_scale = available_height / video_height
                    
                    # Use the smaller scaling factor to ensure the frame fits
                    scale = min(width_scale, height_scale)
                    
                    # Calculate new dimensions
                    new_width = int(video_width * scale)
                    new_height = int(video_height * scale)
                    
                    # Store dimensions for video playback
                    self.video_dimensions = (new_width, new_height)
                    
                    # Resize thumbnail to match video dimensions
                    frame_pil = frame_pil.resize(self.video_dimensions, Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(frame_pil)
                    
                    # Update display
                    self.content_label.configure(image=photo)
                    self.content_label.image = photo
                    
                    # Update content frame size
                    self.content_frame.configure(width=new_width, height=new_height)
                
                # Reset video to beginning and cleanup
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                cap.release()
                
                # Show video controls
                self.video_controls.grid()
                self.play_button.configure(text="▶")
                self.video_progress['value'] = 0
                
            else:
                # Image handling code
                self.video_controls.grid_remove()
                photo = self.load_and_resize_image(content_path)
                if photo is not None:
                    self.content_label.configure(image=photo)
                    self.content_label.image = photo
            
            self.viewed_content.add(self.current_content)
            
            if len(self.recommendation_queue) < 3:
                self.update_recommendation_queue()
                    
        except Exception as e:
            logger.error(f"Error showing next content: {str(e)}", exc_info=True)
            self.current_content = None
        
        finally:
            self.enable_buttons()

    # Loads and resizes an image for display without cropping
    def load_and_resize_image(self, image_path: str) -> Optional[ImageTk.PhotoImage]:
        try:
            if self.resource_manager.check_memory():
                self.clear_display_cache()
            
            cache_key = str(image_path)
            if cache_key in self.display_cache:
                self.display_cache.move_to_end(cache_key)
                return self.display_cache[cache_key]
            
            print(f"Loading image from: {image_path}")
            
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None
                
            # Load and convert image
            image = Image.open(image_path)
            image = image.convert('RGB')
            
            # Get original dimensions
            orig_width, orig_height = image.size
            print(f"Original dimensions: {orig_width}x{orig_height}")
            
            # Get window size or use minimum size if window isn't ready
            window_width = max(self.root.winfo_width(), 1024)
            window_height = max(self.root.winfo_height(), 768)
            
            # Calculate available space with padding
            available_width = window_width - 100
            available_height = window_height - 200
            
            # Ensure minimum display size
            available_width = max(available_width, 600)
            available_height = max(available_height, 400)
            
            # Calculate scaling factors for both dimensions
            width_scale = available_width / orig_width
            height_scale = available_height / orig_height
            
            # Use the smaller scaling factor to ensure the image fits both dimensions
            scale = min(width_scale, height_scale)
            
            # Calculate new dimensions
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            
            print(f"New dimensions: {new_width}x{new_height}")
            
            # Resize image
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(resized_image)
            
            # Update content frame size to match image
            self.content_frame.configure(width=new_width, height=new_height)
            
            # Cache the image
            if len(self.display_cache) >= self.max_display_cache:
                self.display_cache.popitem(last=False)
            self.display_cache[cache_key] = photo
            
            return photo
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            print(f"Error loading image: {str(e)}")
            return None

    # Enables user interaction buttons
    def enable_buttons(self):
        for button in (self.like_button, self.dislike_button, self.skip_button):
            button.state(['!disabled'])

    # Disables user interaction buttons
    def disable_buttons(self):
        for button in (self.like_button, self.dislike_button, self.skip_button):
            button.state(['disabled'])

    # Updates displayed statistics
    def update_stats(self):
        stats_text = (
            f"Views: {self.stats['views']} | "
            f"Likes: {self.stats['likes']} | "
            f"Images: {self.stats['image_views']} | "
            f"Videos: {self.stats['video_views']}"
        )
        self.stats_label.config(text=stats_text)

    # Handles user liking content
    def like_content(self):
        if self.current_content:
            self.recommendation_system.update_user_preferences(
                self.user_id,
                self.current_content,
                True
            )
            self.show_next_content()

    # Handles user disliking content
    def dislike_content(self):
        if self.current_content:
            self.recommendation_system.update_user_preferences(
                self.user_id,
                self.current_content,
                False
            )
            self.show_next_content()

    # Handles user skipping content
    def skip_content(self):
        if self.current_content:
            self.show_next_content()

    # Shows a popup message
    def show_message(self, message: str):
        msg_window = tk.Toplevel(self.root)
        msg_window.title("Message")
        ttk.Label(msg_window, text=message, padding=20).pack()
        ttk.Button(msg_window, text="OK", command=msg_window.destroy).pack(pady=10)

    # Handles application shutdown
    def on_closing(self):
        if not hasattr(self, '_closing'):
            self._closing = True
            logger.info("Closing application")
            try:
                # Cleanup resources
                self.clear_display_cache()
                self.recommendation_system.cleanup_resources()
                self.resource_manager.shutdown()
                
                # Destroy window
                if self.root.winfo_exists():
                    self.root.destroy()
                
            except Exception as e:
                logger.error("Error during cleanup: %s", str(e), exc_info=True)
                if self.root.winfo_exists():
                    self.root.destroy()

    # Starts the GUI main loop
    def run(self):
        logger.info("Starting GUI main loop")
        self.root.mainloop()
    
# Main entry point for the application. Handles command line arguments, initializes the recommendation system and GUI, and manages the application lifecycle
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Content Recommendation System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--content_path",
        type=str,
        default="C:/Users/burfo/Documents/ai_content_recommendation/content",
        help="Path to the content directory containing images and videos"
    )
    
    parser.add_argument(
        "--max_frames",
        type=int,
        default=100,
        help="Maximum number of frames to extract from videos"
    )
    
    parser.add_argument(
        "--frame_sample_rate",
        type=int,
        default=1,
        help="Sample every nth frame from videos"
    )
    
    args = parser.parse_args()
    
    gui = None
    
    try:
        # Verify content directory exists
        if not os.path.exists(args.content_path):
            logger.error("Directory not found: %s", args.content_path)
            print(f"""Please create a directory named 'content' in the same location as this script and add some content files:""")
            return
        
        # Check for required packages
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV (cv2) is required for video support. Please install it using:")
            print("\npip install opencv-python\n")
            return
        
        # Initialize system
        logger.info("Initializing recommendation system...")
        recommendation_system = RecommendationSystem(args.content_path)
        
        # Launch GUI
        logger.info("Launching GUI...")
        gui = ContentRecommenderGUI(recommendation_system)
        
        # Register cleanup handler
        import atexit
        def cleanup():
            if gui is not None and not hasattr(gui, '_closing'):
                gui.on_closing()
        atexit.register(cleanup)
        
        # Start application
        gui.run()
        
    except Exception as e:
        logger.error("Fatal error: %s", str(e), exc_info=True)
        # Show error in popup if GUI exists
        if gui is not None:
            gui.show_message(f"An error occurred:\n{str(e)}\n\nCheck the log file for details.")
    
    finally:
        # Ensure cleanup
        if gui is not None and not hasattr(gui, '_closing'):
            try:
                logger.info("Performing final cleanup...")
                gui.on_closing()
            except Exception as e:
                logger.error("Error in final cleanup: %s", str(e), exc_info=True)

# Entry point
if __name__ == "__main__":
    main()