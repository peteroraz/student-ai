from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import csv
import io

app = Flask(__name__)
model = joblib.load('student_model.pkl')

def predict_student(attendance, study_hours, prev_score,
                    family_income, internet, teacher_inter):
    features = np.array([[attendance, study_hours, prev_score,
                          family_income, internet, teacher_inter]])
    prediction  = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    recommendations = []
    if prediction == 0:
        if attendance    < 70: recommendations.append("Address attendance urgently")
        if study_hours   < 2:  recommendations.append("Enrol in after-school study program")
        if teacher_inter < 4:  recommendations.append("Schedule one-on-one teacher sessions")
        if prev_score    < 50: recommendations.append("Arrange peer tutoring support")

    return {
        'prediction':       int(prediction),
        'pass_probability': round(float(probability[1]) * 100, 1),
        'fail_probability': round(float(probability[0]) * 100, 1),
        'recommendations':  recommendations
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data   = request.get_json()
    result = predict_student(
        float(data['attendance']),
        float(data['study_hours']),
        float(data['prev_score']),
        float(data['family_income']),
        int(data['internet']),
        float(data['teacher_inter'])
    )
    return jsonify(result)

@app.route('/predict_bulk', methods=['POST'])
def predict_bulk():
    file    = request.files['file']
    stream  = io.StringIO(file.stream.read().decode('utf-8'))
    reader  = csv.DictReader(stream)
    results = []

    for row in reader:
        try:
            result = predict_student(
                float(row['attendance']),
                float(row['study_hours']),
                float(row['prev_score']),
                float(row['family_income']),
                int(row['internet']),
                float(row['teacher_inter'])
            )
            result['name'] = row.get('name', 'Unknown')
            results.append(result)
        except Exception as e:
            results.append({'name': row.get('name', 'Unknown'), 'error': str(e)})

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)