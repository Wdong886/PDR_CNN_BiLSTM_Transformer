# Pedestrian Dead Reckoning (PDR) Trajectory Estimation

## Project Overview
This repository implements Pedestrian Dead Reckoning (PDR) with error correction using regression models. It calculates distance errors and heading deviations, trains regression models to predict these errors, and estimates pedestrian trajectories. Pre-trained models are included for immediate use.

## Setup Instructions
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

## Workflow
Workflow
Follow these steps to reproduce the trajectory estimation:

1. **Generate Dead Reckoning Errors**
Calculate distance errors and heading deviations under fixed windows:
   ```bash
   python src/ImprovePDR_error.py
   
2. **Train Regression Models (Optional)**
Pre-trained models are provided, but you can retrain if needed:

   ```bash
      python regressionFinal/distance/distance_regression.py
      python regressionFinal/direction_regression/direction_Regression.py
   
**Pre-trained models locations**:  
- Distance model: `regressionFinal/distance/`  
- Heading model: `regressionFinal/direction_regression/`  

3. **Run Trajectory Estimation**
Estimate pedestrian motion trajectories:

   ```bash
   python demo.py
