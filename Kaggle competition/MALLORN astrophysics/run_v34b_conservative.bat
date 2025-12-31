@echo off
call C:\Users\alexy\miniconda3\Scripts\activate.bat nlp-gpu
cd /d "C:\Users\alexy\Documents\Claude_projects\Kaggle competition\MALLORN astrophysics"
python scripts\train_v34b_conservative.py
