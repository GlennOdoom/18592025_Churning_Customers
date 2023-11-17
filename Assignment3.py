import streamlit as st
import pandas as pd
import joblib
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the pre-trained model
model_path = 'Churn_Mod.pkl'
with open(model_path, 'rb') as model_file:
    model = joblib.load(model_file)

# Function to preprocess input data
def preprocess_input(data, label_encoder_dict=None, scaler=None):
    if label_encoder_dict is None:
        label_encoder_dict = {}
        scaler = StandardScaler()

    # Encode categorical columns
    for column in data.select_dtypes(include=['object']).columns:
        if column not in label_encoder_dict:
            label_encoder_dict[column] = LabelEncoder()
        data[column] = label_encoder_dict[column].fit_transform(data[column])

    # Scale numerical features
    numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    return data, label_encoder_dict, scaler

# Function to make predictions
def predict_churn(data, label_encoder_dict, scaler):
    # Preprocess the input data
    processed_input, _, _ = preprocess_input(data, label_encoder_dict, scaler)

    # Ensure that the input features match the order and format used during training
    selected_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'customerID', 'Contract',
                         'OnlineSecurity', 'PaymentMethod', 'TechSupport', 'OnlineBackup', 'PaperlessBilling']

    # Reorder and select the features used during training
    processed_input = processed_input[selected_features]

    # Make predictions
    predictions = model.predict(processed_input)
    print("Raw Predictions:", predictions)

    # Debugging confidence
    confidence = predictions[0][0]
    print("Confidence:", confidence)

    # Assuming you have a threshold for classifying churn or no churn
    churn_prediction = "Churn" if confidence >= 0.4 else "No Churn"
    return churn_prediction, confidence

# Function to get user input
def get_user_input():
    # Create input form
    st.sidebar.header("Enter Customer Information")

    # Example input fields (customize based on your dataset and features)
    user_input = {
        'tenure': st.sidebar.slider('Tenure (months)', min_value=0, max_value=72, step=1, value=24),
        'MonthlyCharges': st.sidebar.slider('Monthly Charges', min_value=0, max_value=120, step=1, value=50),
        'TotalCharges': st.sidebar.slider('Total Charges', min_value=0, max_value=8000, step=1, value=2000),
        'customerID': st.sidebar.text_input('Customer ID'),
        'Contract': st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two year']),
        'OnlineSecurity': st.sidebar.selectbox('Online Security', ['Yes', 'No']),
        'PaymentMethod': st.sidebar.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']),
        'TechSupport': st.sidebar.selectbox('Tech Support', ['Yes', 'No']),
        'OnlineBackup': st.sidebar.selectbox('Online Backup', ['Yes', 'No']),
        'PaperlessBilling': st.sidebar.selectbox('Paperless Billing', ['Yes', 'No']),
    }

    return pd.DataFrame([user_input])

# Function to display results
def display_results(prediction, confidence):
    st.subheader("Churn Prediction Result:")
    st.write(f"The model predicts: **{prediction}** with a confidence of {confidence:.2%}")

# Streamlit app
def main():
    st.title("Customer Churn Prediction App")

    # Get user input
    user_input = get_user_input()

    # Preprocess input data
    user_input, label_encoder_dict, scaler = preprocess_input(user_input)

    # Make predictions
    prediction, confidence = predict_churn(user_input, label_encoder_dict, scaler)

    # Display results
    display_results(prediction, confidence)


# Run the app
if __name__ == "__main__":
    main()
