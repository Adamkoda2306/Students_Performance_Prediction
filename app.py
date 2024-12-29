import joblib
import pandas as pd
from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
from flask_cors import CORS


load_dotenv()

app = Flask(__name__)
CORS(app)

#Get Environment Variables
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')

# Load the model
with open("student_performance_model.pkl", "rb") as file:
    model = joblib.load(file)
    
print(f"Loaded model type: {type(model)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        attendance_rate = float(request.form['AttendanceRate'])
        study_hours_per_week = float(request.form['StudyHoursPerWeek'])
        previous_grade = float(request.form['PreviousGrade'])
        extracurricular_activities = int(request.form['ExtracurricularActivities'])
        parental_support = int(request.form['ParentalSupport'])
        
        # Prepare data as a DataFrame
        new_data = {
            'AttendanceRate': attendance_rate,
            'StudyHoursPerWeek': study_hours_per_week,
            'PreviousGrade': previous_grade,
            'ExtracurricularActivities': extracurricular_activities,
            'ParentalSupport': parental_support
        }
        data_df = pd.DataFrame([new_data])
        
        # Predict using the model
        predicted_score = model.predict(data_df)[0]
        
        return render_template(
            'index.html', 
            prediction_text=f"Predicted Final Grade: {predicted_score:.2f}"
        )
    except Exception as e:
        return render_template(
            'index.html', 
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
