import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('models/trained_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.set_page_config(page_title="Employee Salary Predictor", layout="centered")
st.title("💼 Employee Salary Prediction App")
st.markdown("Predict your expected salary based on qualifications and job profile.")

# --- Form inputs ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        job_title = st.selectbox("Job Title", ['Software Engineer', 'Data Scientist', 'DevOps Engineer', 'QA Engineer',
                                               'Product Manager', 'UI/UX Designer', 'Mobile Developer', 'Machine Learning Engineer',
                                               'Security Analyst', 'Cloud Architect', 'Business Analyst', 'Technical Writer',
                                               'Blockchain Developer', 'Frontend Developer', 'Backend Developer', 'Database Administrator'])

        education = st.selectbox("Education Level", ['Bachelors', 'Masters', 'PhD'])
        experience = st.slider("Years of Experience", 0.0, 20.0, 2.0, step=0.1)
        job_level = st.selectbox("Job Level", ['Junior', 'Mid', 'Senior'])
        certification = st.radio("Has Certification?", ['Yes', 'No'])

    with col2:
        location = st.selectbox("Job Location", ['Bangalore', 'Hyderabad', 'Pune', 'Chennai', 'Delhi', 'Mumbai', 'Kolkata',
                                                 'Ahmedabad', 'Jaipur', 'Noida', 'Gurgaon', 'Coimbatore', 'Indore', 'Nagpur', 'Thiruvananthapuram'])

        company_size = st.selectbox("Company Size", ['Small', 'Medium', 'Large'])
        industry = st.selectbox("Industry", ['Tech', 'Finance', 'Healthcare'])
        employment_type = st.selectbox("Employment Type", ['Full-time', 'Contract', 'Intern'])
        remote_work = st.radio("Remote Work?", ['Yes', 'No'])
        university_tier = st.selectbox("University Tier", ['Tier 1', 'Tier 2', 'Tier 3'])

    submitted = st.form_submit_button("Predict Salary")

# --- Prediction logic ---
if submitted:
    # Build input dataframe
    input_dict = {
        'Years_of_Experience': [experience],
        'Experience_Squared': [experience**2],
        'Has_Certification': [1 if certification == 'Yes' else 0]
    }

    # Add all one-hot encodable fields (set 1 or 0)
    features = [
        job_title, education, location, company_size, industry,
        employment_type, remote_work, university_tier, job_level
    ]

    base_df = pd.get_dummies(pd.DataFrame([{
        'Job_Title': job_title,
        'Education_Level': education,
        'Location': location,
        'Company_Size': company_size,
        'Industry': industry,
        'Employment_Type': employment_type,
        'Remote_Work': remote_work,
        'University_Tier': university_tier,
        'Job_Level': job_level
    }]), drop_first=True)

    full_input = pd.DataFrame(input_dict)
    input_final = pd.concat([full_input, base_df], axis=1)

    # Ensure all model input columns are present
    model_features = joblib.load('models/feature_names.pkl')
    for col in model_features:
        if col not in input_final.columns:
            input_final[col] = 0  # fill missing columns with 0

    input_final = input_final[model_features]
    input_scaled = scaler.transform(input_final)

    # Predict
    salary_pred = model.predict(input_scaled)[0]
    st.success(f"💰 Estimated Salary: ₹ {int(salary_pred):,}")
