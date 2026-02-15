import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, recall_score, roc_auc_score, roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler


def predict_with_model(model, model_name, input_data):
    X_test = input_data.drop(columns=['HeartDiseaseorAttack'])
    Y_test = input_data['HeartDiseaseorAttack']
    X_test_scaled = StandardScaler().fit_transform(X_test)
    try:
        Y_pred = model[model_name].predict(X_test_scaled)
        accuracy = accuracy_score(Y_test, Y_pred)
        f1 = f1_score(Y_test, Y_pred)
        precision = precision_score(Y_test, Y_pred)
        recall = recall_score(Y_test, Y_pred)
        roc_auc = roc_auc_score(Y_test, Y_pred)
        mcc = matthews_corrcoef(Y_test, Y_pred)

        confussion_matrix = confusion_matrix(Y_test, Y_pred)
        st.write(f"Accuracy: {accuracy}")
        st.write(f"F1 Score: {f1}")
        st.write(f"Precision: {precision}")
        st.write(f"Recall: {recall}")
        st.write(f"ROC AUC: {roc_auc}")
        st.write(f"MCC: {mcc}")
        # Create plot
        fig, ax = plt.subplots()
        sns.heatmap(confussion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        # Show in Streamlit
        st.pyplot(fig)

        predictions = model[model_name].predict(X_test_scaled)
        st.write("Predictions:")
        predictions_df = pd.DataFrame(predictions, columns=['Predicted_HeartDiseaseorAttack'])
        final_df = pd.concat([input_data.reset_index(drop=True), predictions_df], axis=1)
        st.dataframe(final_df)
    except Exception as e:
        st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    st.title("Heart Disease Prediction")
    # Load the trained model
    with open('all_models.joblib', 'rb') as f:
        model = joblib.load(f)
    model_name = st.selectbox("Select a model", options=list(model.keys()))
    st.write(f"You selected: {model_name}")

    test_csv = pd.read_csv('test_dataset.csv')
    test_csv = test_csv.to_csv(index=False)

    st.download_button(
    label="Download Test CSV",
    data=test_csv,
    file_name="test_data.csv",
    mime="text/csv",
    use_container_width=True
)

    csv_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if csv_file is not None:
        input_data = pd.read_csv(csv_file)
        st.write("Input Data:")
        st.dataframe(input_data)
    
        if st.button("Predict"):
            predict_with_model(model, model_name, input_data)
         
