import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# 1. Load data
dataset = pd.read_csv('cancer_data.csv')
print(dataset.shape)

feature_cols = list(dataset.columns[1:-1])   # the 9 feature columns
X = dataset[feature_cols].values
y = dataset['Class'].values

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# 3. Train
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# 4. Evaluate
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

scores = cross_val_score(classifier, X_train, y_train, cv=10)
print('CV accuracy: {:.2f} % (std {:.2f} %)'.format(scores.mean()*100, scores.std()*100))

# 5. Save the model
joblib.dump(classifier, 'logistic_model.pkl')
joblib.dump(feature_cols, 'feature_columns.pkl')
print('Model saved')

# 6. Predict on new data
model = joblib.load('logistic_model.pkl')
cols = joblib.load('feature_columns.pkl')

new_dataset = pd.read_csv('new_datasets.csv')
new_dataset = new_dataset.drop(
    columns=[c for c in ['Sample code number', 'Class'] if c in new_dataset.columns])
X_new = new_dataset[cols]

pred = model.predict(X_new.values)
prob = model.predict_proba(X_new.values)[:, 1]

result = X_new.copy()
result['Predicted_Class'] = pred
result['Predicted_Label'] = np.where(pred == 4, 'Malignant', 'Benign')
result['Malignant_Probability'] = prob.round(3)
result.to_csv('new_dataset_with_predictions.csv', index=False)
print(result[['Predicted_Label', 'Malignant_Probability']])