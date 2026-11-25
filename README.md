# Breast Cancer Detection Using Logistic Regression

A machine learning project that classifies breast tumours as benign or malignant
from nine cell-level features (Wisconsin Breast Cancer dataset, 683 samples).

## Results
- Test accuracy: 95.62%
- 10-fold cross-validation accuracy: 96.70% (std 1.97%)
- 6 errors on 137 test samples (3 false positives, 3 false negatives)

## Files
- `cancer.py` : training, evaluation, saving the model and predicting on new data
- `cancer_data.csv` : dataset
- `new_datasets.csv` : sample new data for prediction
- `new_dataset_with_predictions.csv` : output predictions
- `logistic_model.pkl` : saved trained model

## How to run
pip install -r requirements.txt
python cancer.py

This is an educational project and is not for real medical diagnosis.