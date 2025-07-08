#!/bin/bash

# Initialize Git repository and make first commit
echo "🏀 Initializing NBA Computer Vision Project Git Repository"
echo "=================================================="

# Initialize git repository
git init

# Add all files (except those in .gitignore)
git add .

# Make initial commit
git commit -m "Initial commit: NBA Computer Vision Analysis Project

- Player detection and tracking with YOLOv8
- Possession analysis and timeline generation  
- Shot chart generation and difficulty modeling
- Movement analytics with heatmaps and clustering
- Comprehensive documentation and setup scripts

Large files (video, model) excluded via .gitignore"

echo "✅ Git repository initialized successfully!"
echo ""
echo "Next steps:"
echo "1. Create a repository on GitHub"
echo "2. Add remote origin: git remote add origin <your-repo-url>"
echo "3. Push to GitHub: git push -u origin main"
echo ""
echo "To continue on another computer:"
echo "1. Clone the repository"
echo "2. cd nba_cv_2025_finals"
echo "3. python setup.py"
echo "4. Add your video file"
echo "5. python run_all.py" 