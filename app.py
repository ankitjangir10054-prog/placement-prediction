
from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model
model = joblib.load(
    os.path.join(BASE_DIR, "placement_model.joblib")
)

# Load scaler
scaler = joblib.load(
    os.path.join(BASE_DIR, "scaler.joblib")
)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    study_hours = float(request.form["study_hours"])
    attendance = float(request.form["attendance"])
    sleep_hours = float(request.form["sleep_hours"])
    internet_usage = float(request.form["internet_usage"])
    assignments_completed = float(request.form["assignments_completed"])
    previous_score = float(request.form["previous_score"])
    exam_score = float(request.form["exam_score"])

    # Validate daily hours
    total_hours = study_hours + sleep_hours + internet_usage

    if total_hours > 24:
        return render_template(
            "index.html",
            prediction_text=f"""
            ❌ Invalid Input!<br><br>
            Study Hours + Sleep Hours + Internet Usage
            = {total_hours:.0f} hours.<br><br>
            Total cannot exceed 24 hours in one day.
            """
        )

    features = [[
        study_hours,
        attendance,
        sleep_hours,
        internet_usage,
        assignments_completed,
        previous_score,
        exam_score
    ]]

    # Scale input
    scaled_features = scaler.transform(features)

    # Prediction
    prediction = model.predict(scaled_features)[0]

    # Probability
    probability = model.predict_proba(scaled_features)[0]

    placement_prob = probability[1] * 100
    not_placement_prob = probability[0] * 100

    if prediction == 1:
        result = f"""
        🎉 Congratulations! Student is likely to be Placed.<br><br>
        Placement Probability: {placement_prob:.2f}%<br>
        Not Placement Probability: {not_placement_prob:.2f}%
        """
    else:
        result = f"""
        ❌ Student is likely NOT to be Placed.<br><br>
        Placement Probability: {placement_prob:.2f}%<br>
        Not Placement Probability: {not_placement_prob:.2f}%
        """

    return render_template(
        "index.html",
        prediction_text=result
    )
if __name__ == "__main__":
    app.run(debug=True)

