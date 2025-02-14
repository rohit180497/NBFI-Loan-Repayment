
from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import mlflow
import joblib
from mlflow.tracking import MlflowClient
import json
import threading
from azureml.core import Workspace
from db_manager import store_prediction

# Ensure full output is displayed
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.expand_frame_repr', False)  # Prevent column wrapping

app = Flask(__name__)

# Connect to Azure ML Workspace
ws = Workspace.from_config()  # This reads from config.json

# Set MLflow Tracking URI to Azure ML Workspace
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

print("MLflow Connected to Azure ML!")


# Initialize MLflow Client
client = MlflowClient()

# Define Model Name
MODEL_NAME = "NBFI-loan-defaulter-prediction-logistic-regression"

#Fetch Latest Model Version
latest_model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")

if not latest_model_versions:
    raise ValueError(f"No registered model found for {MODEL_NAME}")

#Get latest version dynamically
latest_model_version = max([int(m.version) for m in latest_model_versions])
latest_run_id = [m.run_id for m in latest_model_versions if int(m.version) == latest_model_version][0]

print(f"Latest Model Version: {latest_model_version} | Run ID: {latest_run_id}")

# Function to Load Model from MLflow
def load_model():   
    try:
        model_uri = f"models:/{MODEL_NAME}/{latest_model_version}"
        model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded successfully from Azure MLflow!")
        return model
    except Exception as e:
        print(f"Error loading model from MLflow: {e}")
        return None

# Function to Load Scaler from MLflow Artifacts
def load_scaler():
    try:
        scaler_path = client.download_artifacts(latest_run_id, "Logistic_regression/Logistic_regression_scaler.pkl")
        scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully from Azure MLflow!")
        return scaler
    except Exception as e:
        print(f"Error loading scaler from MLflow: {e}")
        return None
    
# Load Model & Scaler
model = load_model()
scaler = load_scaler()
# Define expected columns for the model after encoding

expected_columns = [
    'Score_Source_2', 'Score_Source_3', 'Employed_Years', 'Car_Owned',
    'Annuity_Income_Ratio', 'ID_Years', 'Credit_to_Age_Ratio',
    'Phone_Change', 'Age_Years', 'Loan_Duration', 'House_Own',
    'Client_Gender_Male', 'Client_Education_Graduation dropout',
    'Client_Education_Junior secondary', 'Client_Education_Secondary',
    'Loan_Contract_Type_RL', 'Client_Permanent_Match_Tag_Yes',
    'Client_Income_Type_Govt Job', 'Client_Income_Type_Other',
    'Client_Income_Type_Retired', 'Client_Income_Type_Service',
    'Client_Housing_Type_Home', 'Client_Housing_Type_Municipal',
    'Client_Housing_Type_Office', 'Client_Housing_Type_Rental',
    'Client_Housing_Type_Shared', 'Client_Marital_Status_M',
    'Client_Marital_Status_S', 'Client_Marital_Status_W'
]
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form

        raw_input_json = json.dumps(request.form.to_dict(), default=str)

        # Convert form data to DataFrame
        input_data = pd.DataFrame([[
            int(form_data["ID_Years"]),
            float(form_data["credit_amount"]) / int(form_data["age_years"]), #Credit_to_Age_Ratio
            int(form_data["car_owned"]),
            int(form_data["house_own"]),
            float(form_data["score_source_2"]),
            float(form_data["score_source_3"]),
            int(form_data["phone_change"]),
            float(form_data["loan_annuity"]) / float(form_data["client_income"]),  # Annuity_Income_Ratio
            float(form_data["credit_amount"]) / float(form_data["loan_annuity"]),  # Loan_Duration
            int(form_data["age_years"]),
            int(form_data["employed_years"]),
            form_data["client_income_type"],
            form_data["client_education"],
            form_data["client_marital_status"],
            form_data["client_gender"],
            form_data["loan_contract_type"],
            form_data["client_housing_type"],
            int(form_data["client_permanent_match_tag"])
        ]], columns=[
            "ID_Years","Credit_to_Age_Ratio","Car_Owned", "House_Own", "Score_Source_2", "Score_Source_3", "Phone_Change", "Annuity_Income_Ratio", 
            "Loan_Duration", "Age_Years", "Employed_Years",
            "Client_Income_Type", "Client_Education", "Client_Marital_Status", "Client_Gender", 
            "Loan_Contract_Type", "Client_Housing_Type", "Client_Permanent_Match_Tag"
        ])

        # One-Hot Encoding for categorical variables
        categorical_features = [
            "Client_Income_Type", "Client_Education", "Client_Marital_Status",
            "Client_Gender", "Loan_Contract_Type", "Client_Housing_Type", "Client_Permanent_Match_Tag"
        ]
        input_data = pd.get_dummies(input_data, columns=categorical_features, prefix=categorical_features)

        # Ensure all expected columns are present (fill missing with 0)
        for col in expected_columns:
            if col not in input_data.columns:
                input_data[col] = 0

        # Reorder columns to match model's expected order
        filter_data = input_data[expected_columns]

        # Convert numerical columns to float64
        numerical_columns = ['Score_Source_2', 'Score_Source_3', 'Employed_Years', 'Car_Owned',
       'Annuity_Income_Ratio', 'ID_Years', 'Credit_to_Age_Ratio',
       'Phone_Change', 'Age_Years', 'Loan_Duration', 'House_Own']
        
        # Ensure transformation output is assigned correctly
        scaled_values = scaler.transform(filter_data[numerical_columns])  # Transform numerical columns
        filter_data = filter_data.copy()
        filter_data.loc[:, numerical_columns] = pd.DataFrame(scaled_values, columns=numerical_columns, index=filter_data.index)

        # filter_data[numerical_columns] = pd.DataFrame(scaled_values, columns=numerical_columns, index=filter_data.index)

       
        # Convert categorical one-hot encoded columns to boolean
        boolean_columns = [
            "Client_Income_Type_Govt Job", "Client_Income_Type_Other", "Client_Income_Type_Retired",
            "Client_Income_Type_Service", "Client_Education_Graduation dropout", 
            "Client_Education_Junior secondary", "Client_Education_Secondary",
            "Client_Marital_Status_M", "Client_Marital_Status_S", "Client_Marital_Status_W",
            "Client_Gender_Male", "Loan_Contract_Type_RL", "Client_Housing_Type_Home",
            "Client_Housing_Type_Municipal", "Client_Housing_Type_Office", "Client_Housing_Type_Rental",
            "Client_Housing_Type_Shared", "Client_Permanent_Match_Tag_Yes"
        ]

        filter_data.loc[:, boolean_columns] = filter_data[boolean_columns].astype(bool)
        print(filter_data.iloc[:,:])
        # filter_data[boolean_columns] = filter_data[boolean_columns].astype(bool)
        print(model)

        # Make prediction
        # Make prediction
        if model is not None:
            prediction_proba = model.predict_proba(filter_data)[0]  # Extract the array 
            proba_default = round(float(prediction_proba[1]) * 100, 2)  # Probability of default
            proba_no_default = round(float(prediction_proba[0]) * 100, 2)  # Probability of non-default

            # Define Message Based on Probability
            if proba_default > 50:
                predicted_class = 1  # High Risk: Default
                result_message = f"🚨 High Risk: {proba_default}% probability of loan default."
            else:
                predicted_class = 0  # Low Risk: No Default
                result_message = f"✅ Low Risk: {proba_no_default}% probability of timely repayment."

            print("Returning result:", result_message)  

            # Return Response to Frontend First
            response = jsonify({"prediction": result_message})

            latest_run_id = None
            try:
                latest_run = mlflow.search_runs(order_by=["start_time desc"]).iloc[0]  # Fetch latest run
                latest_run_id = latest_run["run_id"]
            except Exception as e:
                print(f"Error fetching MLflow run ID: {e}")

            # Validate JSON Data Before Storing
            try:
                json_data = input_data.to_json(orient="records")
                print("Storing JSON Data:", json_data)  # Debugging
            except Exception as e:
                print(f"Error converting input data to JSON: {e}")

            # Store Prediction in Azure SQL using a Background Thread
            def async_store():
                try:
                    store_prediction(
                    run_id=latest_run_id,
                    client_id=form_data["id"],  # Extract from request payload
                    raw_input=raw_input_json,  # Raw input
                    processed_input=filter_data.to_dict(orient="records"),  # Preprocessed before prediction
                    prediction_prob=proba_default,
                    predicted_class=predicted_class
                )
                    print("Prediction stored successfully in Azure SQL!")
                except Exception as e:
                    print(f" Failed to store prediction: {e}")

            threading.Thread(target=async_store).start()

            return response  # Ensure this runs before storing

        else:
            return jsonify({"prediction": "Error: Model not loaded"})

    except Exception as e:
        return jsonify({"error": str(e)})
    

if __name__ == "__main__":
    app.run(debug=True)

