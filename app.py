from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model & columns
model = joblib.load('spacex_model.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html', prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        flight_number = int(request.form['FlightNumber'])
        payload_mass = float(request.form['PayloadMass'])
        orbit = request.form['Orbit']
        launch_site = request.form['LaunchSite']
        block = int(request.form['Block'])
        reused_count = int(request.form['ReusedCount'])

        # Prepare data dictionary
        input_data = {
            'FlightNumber': flight_number,
            'PayloadMass': payload_mass,
            'Block': block,
            'ReusedCount': reused_count,
            'Flights': 1,  # dummy if your model expects
            'Orbit_' + orbit: 1,
            'LaunchSite_' + launch_site: 1
        }

        # Create DataFrame with all columns as model expects
        df = pd.DataFrame(columns=model_columns)
        df.loc[0] = 0
        for col, val in input_data.items():
            if col in df.columns:
                df.at[0, col] = val

        # Predict
        prediction = model.predict(df)[0]
        output = "✅ Successful Landing" if prediction == 1 else "❌ Landing Failed"

        return render_template('index.html', prediction_text=output)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
