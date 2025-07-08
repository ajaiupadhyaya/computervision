#!/usr/bin/env python3
"""
Run all NBA Computer Vision Analysis scripts in sequence
"""

import os
import sys
import subprocess
import time

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"🏀 {description}")
    print(f"📝 Running: {script_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(f"✅ {script_name} completed successfully")
        if result.stdout:
            print("Output:", result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_name} failed with error:")
        print(f"Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"❌ {script_name} not found")
        return False

def main():
    print("🏀 NBA Computer Vision Analysis - Full Pipeline")
    print("This will run all analysis scripts in sequence")
    
    # Check if we're in the right directory
    if not os.path.exists("detect_and_log.py"):
        print("❌ Please run this script from the nba_cv_2025_finals directory")
        sys.exit(1)
    
    # Check for required files
    required_files = [
        ("yolov8n.pt", "YOLOv8 model"),
        ("game1_highlights.mp4", "Video file")
    ]
    
    missing_files = []
    for file, description in required_files:
        if not os.path.exists(file):
            missing_files.append(f"{file} ({description})")
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease run setup.py first or add the missing files manually")
        sys.exit(1)
    
    # Define the analysis pipeline
    pipeline = [
        ("detect_and_log.py", "Player Detection & Tracking"),
        ("possession_tracker.py", "Possession Analysis"),
        ("shot_chart.py", "Shot Chart Generation"),
        ("shot_difficulty_model.py", "Shot Difficulty Analysis"),
        ("heatmap_generator.py", "Movement Analytics")
    ]
    
    start_time = time.time()
    success_count = 0
    
    for script, description in pipeline:
        if run_script(script, description):
            success_count += 1
        else:
            print(f"\n❌ Pipeline stopped due to failure in {script}")
            break
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print("📊 Pipeline Summary")
    print(f"{'='*60}")
    print(f"✅ Completed: {success_count}/{len(pipeline)} scripts")
    print(f"⏱️  Total time: {duration:.1f} seconds")
    
    if success_count == len(pipeline):
        print("\n🎉 All analyses completed successfully!")
        print("\nGenerated files:")
        output_files = [
            "game1_detections.csv",
            "possessions_summary.csv", 
            "shot_difficulty_output.csv",
            "movement_zone_summary.csv",
            "possession_timeline.png",
            "game1_shot_chart.png",
            "xfg_scatter.png",
            "movement_heatmap.png",
            "movement_clusters.png"
        ]
        
        for file in output_files:
            if os.path.exists(file):
                size = os.path.getsize(file)
                print(f"   ✅ {file} ({size:,} bytes)")
            else:
                print(f"   ❌ {file} (missing)")
    else:
        print(f"\n⚠️  Pipeline incomplete. {len(pipeline) - success_count} scripts failed.")

if __name__ == "__main__":
    main() 