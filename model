import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

np.random.seed(42)
n = 200

attendance    = np.random.randint(40, 101, n)
study_hours   = np.random.uniform(0.5, 6, n)
prev_score    = np.random.randint(20, 100, n)
family_income = np.random.randint(1, 6, n)
internet      = np.random.randint(0, 2, n)
teacher_inter = np.random.uniform(1, 10, n)

X = np.column_stack([attendance, study_hours, prev_score,
                     family_income, internet, teacher_inter])

score     = (attendance * 0.3 + study_hours * 4 + prev_score * 0.3 +
             family_income * 1.5 + internet * 3 + teacher_inter * 1.2)
y         = (score > np.percentile(score, 35)).astype(int)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, 'student_model.pkl')
print("Model trained and saved as student_model.pkl")