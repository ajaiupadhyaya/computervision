#!/usr/bin/env python3
"""
Setup script for NBA Computer Vision Analysis Project
Downloads required large files and sets up the environment
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path

def download_file(url, filename):
    """Download a file with progress indicator"""
    print(f"📥 Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"✅ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

def main():
    print("🏀 NBA Computer Vision Analysis Project Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("requirements.txt"):
        print("❌ Please run this script from the nba_cv_2025_finals directory")
        sys.exit(1)
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Download YOLOv8 model
    model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    model_file = "yolov8n.pt"
    
    if not os.path.exists(model_file):
        if not download_file(model_url, model_file):
            print("❌ Failed to download YOLOv8 model. Please download manually:")
            print(f"   wget {model_url}")
            sys.exit(1)
    else:
        print("✅ YOLOv8 model already exists")
    
    # Check for video file
    video_file = "game1_highlights.mp4"
    if not os.path.exists(video_file):
        print(f"⚠️  Video file '{video_file}' not found")
        print("   Please add your video file to the project directory")
        print("   or update the video_path in detect_and_log.py")
    else:
        print("✅ Video file found")
    
    # Install requirements
    print("\n📦 Installing Python dependencies...")
    os.system("pip install -r requirements.txt")
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Add your video file (or update video_path in detect_and_log.py)")
    print("2. Run: python detect_and_log.py")
    print("3. Follow the analysis pipeline in the README")

if __name__ == "__main__":
    main() 