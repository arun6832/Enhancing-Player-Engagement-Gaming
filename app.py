from flask import Flask, render_template, request
import numpy as np
import pickle

# Step 1: Initialize Flask app
app = Flask(__name__)

# Step 2: Load the trained model and scaler
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Step 3: Define route for the home page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get input values from the form
            sessions_per_week = float(request.form["SessionsPerWeek"])
            avg_session_duration = float(request.form["AvgSessionDurationMinutes"])
            play_time_hours = float(request.form["PlayTimeHours"])
            in_game_purchases = float(request.form["InGamePurchases"])
            achievements_unlocked = float(request.form["AchievementsUnlocked"])
            player_level = float(request.form["PlayerLevel"])

            # Combine inputs into a NumPy array
            user_input = np.array([[sessions_per_week, avg_session_duration, play_time_hours,
                                    in_game_purchases, achievements_unlocked, player_level]])

            # Scale the input
            user_input_scaled = scaler.transform(user_input)

            # Predict the output
            prediction = model.predict(user_input_scaled)
            engagement_levels = ["Low", "Medium", "High"]  # Assuming 0, 1, 2 for labels
            predicted_level = engagement_levels[prediction[0]]

            # Return the result
            return render_template("index.html", prediction=predicted_level)

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html", prediction=None)

# Step 4: Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
