# Multimodal Prediction of Alzheimer's Disease

## Overview

This project implements a multimodal approach for predicting Alzheimer's Disease using machine learning and deep learning techniques. It was developed as part of Washington University in St. Louis's CSE 419A: Introduction to AI for Health course. The project utilizes the OASIS-1 dataset to analyze various imaging and clinical data for early detection and prediction of Alzheimer's Disease.

## Features

- Multimodal data integration from OASIS-1 dataset
- Deep learning models for image analysis
- Machine learning models for clinical data analysis
- Feature importance analysis using SHAP values
- Comprehensive performance evaluation metrics
- Visual analysis of model predictions
- Combined classifier leveraging both imaging and clinical data

## Core Files

### Notebooks and Analysis
- `OASIS_1_Notebook_Aadarsha.ipynb` - Main implementation notebook with data processing, model training, and evaluation

### Documentation
- `Report/neurips_2024.pdf` - Detailed project report in NeurIPS format
- `Final Demo - OASIS 1 - 419A.pptx` - Project presentation slides

### Visualizations
- `Report/cnn_model.png` - CNN architecture visualization
- `Report/final_classifier.png` - Final classifier architecture and results

## Technologies

- Python 3.12
- TensorFlow/Keras
- PyTorch
- Scikit-learn
- XGBoost
- SHAP (SHapley Additive exPlanations)
- Pandas & NumPy
- Matplotlib & Seaborn
- Jupyter Notebooks

## Getting Started

1. **Set up the environment**:
```sh
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. **Data Preparation**:
   - Download the OASIS-1 dataset
   - Place the dataset in the appropriate directory
   - Follow the data preprocessing steps in the notebook

3. **Running the Analysis**:
   - Open `OASIS_1_Notebook_Aadarsha.ipynb` in Jupyter Notebook
   - Follow the notebook cells sequentially for complete analysis

## Dependencies

Key dependencies include:
- tensorflow (2.18.0)
- torch (2.5.1)
- scikit-learn
- xgboost (2.1.3)
- numpy (2.0.2)
- pandas
- matplotlib (3.9.2)
- seaborn
- shap (0.46.0)

Full dependencies are listed in `requirements.txt`

## Project Structure

```
├── OASIS_1_Notebook_Aadarsha.ipynb    # Main implementation notebook
├── requirements.txt                    # Project dependencies
├── Report/                            # Project documentation
│   ├── neurips_2024.pdf              # Detailed project report
│   ├── cnn_model.png                 # Model architecture visualization
│   └── final_classifier.png          # Final results visualization
└── Final Demo - OASIS 1 - 419A.pptx  # Presentation slides
```

## Acknowledgments

- Developed for CSE 419A at Washington University in St. Louis
- OASIS-1 Dataset contributors
- Course instructors and teaching assistants