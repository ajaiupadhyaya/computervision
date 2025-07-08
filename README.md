# NBA Computer Vision Analysis Project

A computer vision project that analyzes NBA basketball game footage to extract player movement data, track possessions, generate shot charts, and analyze shot difficulty using machine learning.

## 🏀 Project Overview

This project processes NBA basketball video to:
- **Player Detection & Tracking**: Use YOLO object detection to identify and track players
- **Possession Analysis**: Automatically segment game footage into distinct possessions
- **Shot Location Mapping**: Generate shot charts showing estimated shooting locations
- **Shot Difficulty Modeling**: Predict expected field goal percentage (xFG%) based on shot distance and defender proximity
- **Movement Analytics**: Create heatmaps and clustering analysis of player movement patterns

## 📁 Project Structure

```
nba_cv_2025_finals/
├── detect_and_log.py          # Main detection script using YOLOv8
├── possession_tracker.py      # Possession analysis and timeline generation
├── shot_chart.py             # Basketball court visualization with shot locations
├── shot_difficulty_model.py  # ML model for shot difficulty prediction
├── heatmap_generator.py      # Player movement heatmaps and clustering
├── court_utils.py            # Helper functions for court visualization
├── requirements.txt          # Python dependencies
├── yolov8n.pt               # YOLOv8 model (not tracked in git)
├── game1_highlights.mp4     # Source video (not tracked in git)
├── game1_detections.csv     # Raw detection data
├── possessions_summary.csv  # Possession timing and statistics
├── shot_difficulty_output.csv # Shot analysis with xFG predictions
├── movement_zone_summary.csv # Player movement clustering results
├── possession_timeline.png  # Timeline showing possession changes
├── game1_shot_chart.png     # Basketball court with shot locations
├── xfg_scatter.png          # Shot difficulty visualization
├── movement_heatmap.png     # Player movement density heatmap
└── movement_clusters.png    # K-means clustering of movement zones
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd computervision
   ```

2. **Install dependencies**
   ```bash
   cd nba_cv_2025_finals
   pip install -r requirements.txt
   ```

3. **Download required files** (not tracked in git due to size)
   ```bash
   # Download YOLOv8 model
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   
   # Add your video file (rename to game1_highlights.mp4)
   # or update the video path in detect_and_log.py
   ```

### Usage

Run the analysis pipeline in order:

1. **Extract player detections**
   ```bash
   python detect_and_log.py
   ```

2. **Analyze possessions**
   ```bash
   python possession_tracker.py
   ```

3. **Generate shot chart**
   ```bash
   python shot_chart.py
   ```

4. **Analyze shot difficulty**
   ```bash
   python shot_difficulty_model.py
   ```

5. **Create movement analytics**
   ```bash
   python heatmap_generator.py
   ```

## 📊 Output Files

### Data Files
- `game1_detections.csv` - Raw player detection coordinates
- `possessions_summary.csv` - Possession timing and statistics
- `shot_difficulty_output.csv` - Shot analysis with xFG predictions
- `movement_zone_summary.csv` - Player movement clustering results

### Visualizations
- `possession_timeline.png` - Timeline showing possession changes
- `game1_shot_chart.png` - Basketball court with estimated shot locations
- `xfg_scatter.png` - Shot difficulty visualization with xFG% color coding
- `movement_heatmap.png` - Player movement density heatmap
- `movement_clusters.png` - K-means clustering of player movement zones

## 🔧 Configuration

### Video Settings
- Update `video_path` in `detect_and_log.py` to point to your video file
- Adjust `start_seconds` to skip to desired timestamp
- Modify confidence thresholds as needed

### Court Dimensions
- Update `COURT_WIDTH` and `COURT_HEIGHT` in analysis scripts to match your video resolution
- Adjust `HOOP_X` and `HOOP_Y` coordinates for accurate shot analysis

### Model Parameters
- Change `k` in `heatmap_generator.py` to adjust number of movement zones
- Modify thresholds in `possession_tracker.py` for possession detection sensitivity

## 🛠️ Technical Stack

- **Computer Vision**: YOLOv8 (Ultralytics), OpenCV
- **Machine Learning**: Scikit-learn (Logistic Regression, K-means clustering)
- **Data Processing**: Pandas, NumPy, SciPy
- **Visualization**: Matplotlib, Seaborn
- **Progress Tracking**: tqdm

## 📝 Notes

- Large files (video, model) are excluded from git via `.gitignore`
- The project assumes 30fps video - adjust frame rate calculations if needed
- Shot detection uses heuristics - results may need manual validation
- Court coordinates are estimated - calibrate for accurate shot analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and research purposes.