import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

model_placement = joblib.load('Placement_Prediction.pkl')
model_salary = joblib.load('Salary_Prediction.pkl')

def main():

    st.set_page_config(
        page_title="Placement and Salary Prediction",
        layout="wide"
    )

    st.title("Placement and Salary Prediction")
    st.write("Predict placement status and estimated salary based on student performance")

    st.sidebar.header("Information")
    st.sidebar.write(
        "Aplikasi ini memprediksi apakah siswa telah mendapat pekerjaan dan berapa gajinya"
    )

    with st.form("prediction_form"):

        st.subheader("Student Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            ssc = st.number_input("SSC Percentage", 50, 95, 72)
            hsc = st.number_input("HSC Percentage", 50, 94, 72)
            degree = st.number_input("Degree Percentage", 55, 89, 72)
            cgpa = st.number_input("CGPA", 5.5, 9.8, 7.7)

        with col2:
            entrance = st.number_input("Entrance Exam Score", 40, 99, 69)
            tech = st.number_input("Technical Skill Score", 40, 99, 70)
            soft = st.number_input("Soft Skill Score", 40, 99, 69)
            internship = st.number_input("Internship Count", 0, 4, 2)
            projects = st.number_input("Live Projects", 0, 5, 3)

        with col3:
            experience = st.number_input("Work Experience (Months)", 0, 24, 12)
            cert = st.number_input("Certifications", 0, 5, 2)
            attendance = st.number_input("Attendance (%)", 60, 99, 80)
            backlogs = st.number_input("Backlogs", 0, 5, 3)
            extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])

        submit = st.form_submit_button("Predict")

    data = {
        'gender': gender,
        'ssc_percentage': int(ssc),
        'hsc_percentage': int(hsc),
        'degree_percentage': int(degree),
        'cgpa': float(cgpa),
        'entrance_exam_score': int(entrance),
        'technical_skill_score': int(tech),
        'soft_skill_score': int(soft),
        'internship_count': int(internship),
        'live_projects': int(projects),
        'work_experience_months': int(experience),
        'certifications': int(cert),
        'attendance_percentage': int(attendance),
        'backlogs': int(backlogs),
        'extracurricular_activities': extra
    }

    df = pd.DataFrame([data])

    if submit:

        st.subheader("Prediction Result")

        # LANGSUNG PAKAI MODEL (TANPA API)
        placement = model_placement.predict(df)[0]

        colA, colB = st.columns(2)

        if placement == 1:
            colA.success("Placement: YES")

            salary = model_salary.predict(df)[0]
            colB.info(f"Estimated Salary: {salary:.2f} LPA")

        else:
            colA.error("Placement: NO")
            colB.warning("Estimated Salary: 0 (Not Placed)")

        with st.expander("Show Input Data"):
            st.dataframe(df)

# tidak dipakai
def make_prediction(features):
    response = requests.post("http://127.0.0.1:8000/predict", json=features)
    prediction = response.json()["prediction"]
    return prediction

if __name__ == '__main__':
    main()

#python -m streamlit run app.py
