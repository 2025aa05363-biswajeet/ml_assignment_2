# ML Assignment 2 - Streamlit App

# Problem Statement

Heart disease is the leading cause of death in the United States, claiming approximately 647,000 lives annually and affecting millions of people. It often develops due to risk factors such as high blood pressure, high cholesterol, smoking, aging, genetics, diabetes, and unhealthy lifestyle habits. Many individuals remain unaware of their condition until severe events like heart attacks occur, highlighting the need for early detection. Current diagnosis relies on surveys and medical tests, which may not always detect risk early. Therefore, developing a predictive system using patient risk factors and clinical data can help identify high-risk individuals early, enabling preventive care and reducing mortality and healthcare costs.

# Dataset description
The Heart Disease Health Indicators Dataset is a large, cleaned dataset derived from the CDC’s BRFSS 2015 survey, containing 253,680 records and 22 health-related features. It includes clinical, behavioral, and demographic variables such as blood pressure, cholesterol, BMI, smoking, physical activity, age, and income. The dataset is designed for binary classification, with the target variable indicating the presence or absence of heart disease. It is widely used for predictive modeling, risk factor analysis, and early detection of heart disease using machine learning techniques.

# Models Used


| ML Model Name              | Accuracy | F1 Score | Precision | Recall   | MCC      | AUC      |
|----------------------------|----------|----------|-----------|----------|----------|----------|
| Logistic Regression       | 0.907304 | 0.208382 | 0.546820  | 0.128717 | 0.233127 | 0.558774 |
| Decision Tree             | 0.907304 | 0.208382 | 0.546820  | 0.128717 | 0.233127 | 0.558774 |
| k-Nearest Neighbors (KNN)| 0.896385 | 0.222682 | 0.385363  | 0.156581 | 0.198541 | 0.565216 |
| Naive Bayes              | 0.818236 | 0.359850 | 0.270084  | 0.538989 | 0.289052 | 0.693232 |
| Random Forest (Ensemble) | 0.902791 | 0.182903 | 0.449878  | 0.114785 | 0.190843 | 0.550044 |
| XGBoost (Ensemble)      | 0.906023 | 0.188287 | 0.519249  | 0.114993 | 0.212187 | 0.551922 |



## Model Performance Observations

| ML Model Name              | Observation about model performance |
|----------------------------|-------------------------------------|
| Logistic Regression       | Achieved high accuracy (90.73%) and good precision (54.68%), but very low recall (12.87%), indicating it misses many heart disease cases. Moderate MCC and AUC suggest limited ability to distinguish between classes. |
| Decision Tree             | Shows identical performance to Logistic Regression, with high accuracy but poor recall. This indicates the model is biased toward the majority class and is not effective in identifying positive heart disease cases. |
| k-Nearest Neighbors (kNN)| Slightly lower accuracy (89.64%) but better recall than Logistic Regression and Decision Tree. However, precision and MCC are moderate, indicating limited reliability in prediction. |
| Naive Bayes              | Lowest accuracy (81.82%) but highest recall (53.90%) and best AUC (0.693). This model is most effective at identifying heart disease cases, though it produces more false positives. Best overall model for early detection. |
| Random Forest (Ensemble) | High accuracy (90.28%) and good precision (44.99%), but very low recall (11.48%). Despite being an ensemble model, it struggles to detect positive cases effectively. |
| XGBoost (Ensemble)      | High accuracy (90.60%) and strong precision (51.92%), but low recall (11.49%). Shows slightly better MCC than Random Forest, but still limited in detecting heart disease cases. |
