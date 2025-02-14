
from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
# Ensure full output is displayed
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.expand_frame_repr', False)  # Prevent column wrapping

app = Flask(__name__)

# Load trained model
def load_model():
    try:
        with open("model/logistic_model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
# Load scaler
def load_scaler():
    try:
        scaler = pd.read_pickle("model/scaler.pkl")
        return scaler
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return None

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
        input_data = input_data[expected_columns]

        # Convert numerical columns to float64
        numerical_columns = ['Score_Source_2', 'Score_Source_3', 'Employed_Years', 'Car_Owned',
       'Annuity_Income_Ratio', 'ID_Years', 'Credit_to_Age_Ratio',
       'Phone_Change', 'Age_Years', 'Loan_Duration', 'House_Own']
        
        # Ensure transformation output is assigned correctly
        scaled_values = scaler.transform(input_data[numerical_columns])  # Transform numerical columns
        input_data[numerical_columns] = pd.DataFrame(scaled_values, columns=numerical_columns, index=input_data.index)

       
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

        input_data[boolean_columns] = input_data[boolean_columns].astype(bool)

        # Make prediction
        if model is not None:
            prediction_proba = model.predict_proba(input_data)[0]  # Extract the array            
            # Convert to float & round to 2 decimal places
            proba_default = round(float(prediction_proba[1]) * 100, 2)
            proba_no_default = round(float(prediction_proba[0]) * 100, 2)

                     
            # Define message based on probability
            if proba_default > 50:
                result = f"🚨 High Risk: {proba_default}% probability of loan default."
            else:
                result = f"✅ Low Risk: {proba_no_default}% probability of timely repayment."
        else:
            result = "Error: Model not loaded"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

