from flask import Flask, render_template, request
import joblib
import os
import numpy as np
import pickle

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/result", methods=['POST', 'GET'])
def result():
    Gender = int(request.form['Gender'])
    Age = int(request.form['Age'])
    JobSatisfaction = int(request.form['JobSatisfaction'])
    DistanceFromHome = int(request.form['DistanceFromHome'])
    WorkLifeBalance = int(request.form['WorkLifeBalance'])
    OverTime = int(request.form['OverTime'])
    TotalWorkingYears = int(request.form['TotalWorkingYears'])
    PerformanceRating = int(request.form['PerformanceRating'])
    YearsSinceLastPromotion = int(request.form['YearsSinceLastPromotion'])
    MaritalStatus = int(request.form['MaritalStatus'])
    MonthlyIncome = int(request.form['MonthlyIncome'])
    PercentSalaryHike = int(request.form['PercentSalaryHike'])

    x = np.array([Gender, Age, JobSatisfaction, DistanceFromHome, WorkLifeBalance, OverTime, TotalWorkingYears,
                 PerformanceRating, YearsSinceLastPromotion, MaritalStatus, MonthlyIncome, PercentSalaryHike])\
        .reshape(1, -1)

    scaler_path = os.path.join('D:/Task 100 Projects/Employee Attrition Analysis/Employee Attrition Analysis/Attrition',
                               'models/scaler.pkl')
# scaler = None
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    x = scaler.transform(x)

    model_path = os.path.join('D:/Task 100 Projects/Employee Attrition Analysis/Employee Attrition Analysis/Attrition',
                              'models/gb_model.sav')
    gb = joblib.load(model_path)

    predict = gb.predict(x)

    if predict == 0:
        return render_template("NoAttrition.html")
    else:
        return render_template("AttritionOccurs.html")


# if __name__ == "__main__":
#     app.run(debug=True, port=7384)
if __name__ == "__main__":
    app.run(debug=True)
