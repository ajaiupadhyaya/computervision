# Example Outputs

This document shows what the expected outputs should look like when you run the NBA Computer Vision Analysis pipeline.

## 📊 Data Files

### game1_detections.csv
Raw player detection data with columns:
- `frame`: Frame number from video
- `label`: Detection label (usually "person")
- `conf`: Confidence score (0-1)
- `x1, y1, x2, y2`: Bounding box coordinates

Example:
```csv
frame,label,conf,x1,y1,x2,y2
88,person,0.85,120,200,180,280
88,person,0.92,300,150,360,230
89,person,0.87,125,205,185,285
...
```

### possessions_summary.csv
Possession timing and statistics:
- `possession_id`: Unique possession identifier
- `start_frame, end_frame`: Frame range for possession
- `duration_frames`: Number of frames in possession
- `duration_sec`: Duration in seconds
- `avg_spread`: Average player spacing during possession

Example:
```csv
possession_id,start_frame,end_frame,duration_frames,avg_spread,duration_sec
1,88,245,157,85.2,5.23
2,246,412,166,92.1,5.53
3,413,578,165,78.9,5.50
...
```

### shot_difficulty_output.csv
Shot analysis with expected field goal percentage:
- `frame`: Frame number
- `shot_distance_ft`: Distance to hoop in feet
- `defender_distance_ft`: Distance to nearest defender
- `make`: Simulated make/miss (0/1)
- `xFG`: Expected field goal percentage

Example:
```csv
frame,shot_distance_ft,defender_distance_ft,make,xFG
108,18.5,4.2,1,0.72
128,22.1,2.8,0,0.45
148,15.3,6.1,1,0.81
...
```

### movement_zone_summary.csv
Player movement clustering results:
- `zone`: Cluster identifier
- `x_norm_mean, y_norm_mean`: Average position in normalized coordinates
- `x_norm_std, y_norm_std`: Position standard deviation
- `count`: Number of detections in zone

Example:
```csv
zone,x_norm_mean,y_norm_std,x_norm_mean,y_norm_std,count
0,25.3,8.2,15.1,6.8,1247
1,12.8,4.1,28.9,5.2,892
2,37.2,6.5,22.1,7.3,1103
...
```

## 🖼️ Visualization Files

### possession_timeline.png
Line chart showing player spread over time with vertical lines marking possession changes.

### game1_shot_chart.png
Basketball court diagram with red dots showing estimated shot locations.

### xfg_scatter.png
Scatter plot with shot distance vs defender distance, colored by expected field goal percentage.

### movement_heatmap.png
Density heatmap showing where players spent most time on the court.

### movement_clusters.png
Basketball court with colored dots showing different movement zones/clusters.

## 📈 Expected Performance

### Processing Times (approximate)
- **detect_and_log.py**: 2-5 minutes per minute of video (depends on hardware)
- **possession_tracker.py**: 10-30 seconds
- **shot_chart.py**: 5-15 seconds
- **shot_difficulty_model.py**: 10-30 seconds
- **heatmap_generator.py**: 30-60 seconds

### File Sizes (approximate)
- `game1_detections.csv`: 1-5 MB (depends on video length)
- `possessions_summary.csv`: 1-10 KB
- `shot_difficulty_output.csv`: 1-5 KB
- `movement_zone_summary.csv`: 1-2 KB
- PNG files: 100-500 KB each

### Quality Metrics
- **Detection Confidence**: Aim for >0.6 average confidence
- **Possession Detection**: Should identify 20-40 possessions per game quarter
- **Shot Detection**: May need manual validation for accuracy
- **Movement Zones**: 4-8 zones typically provide good coverage

## 🔧 Troubleshooting

### Common Issues
1. **Low detection confidence**: Check video quality, lighting, camera angle
2. **Too many/few possessions**: Adjust thresholds in `possession_tracker.py`
3. **Inaccurate shot locations**: Calibrate court coordinates in analysis scripts
4. **Missing visualizations**: Ensure matplotlib backend is working correctly

### Performance Tips
- Use GPU acceleration if available (CUDA)
- Process shorter video segments for testing
- Adjust confidence thresholds based on video quality
- Consider downsampling video for faster processing 