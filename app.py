from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import joblib

# Initialize the Flask application
app = Flask(__name__)

# Load the "brain" and "translators" we saved earlier
model = tf.keras.models.load_model('churn_model.h5')
le = joblib.load('label_encoder.pkl')
ct = joblib.load('column_transformer.pkl')
sc = joblib.load('scaler.pkl')

# Create a simple homepage route just to check if the app is awake
@app.route('/', methods=['GET'])
def home():
    return "Your Churn Prediction API is successfully running!"

# Create the prediction route where data will be sent
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Receive the JSON data from the user
        data = request.get_json()
        
        # 2. Extract features in the EXACT order your original X matrix had them
        features = [[
            data['CreditScore'],
            data['Geography'],
            data['Gender'],
            data['Age'],
            data['Tenure'],
            data['Balance'],
            data['NumOfProducts'],
            data['HasCrCard'],
            data['IsActiveMember'],
            data['EstimatedSalary']
        ]]
        
        # Convert to a numpy array so our tools can read it
        features = np.array(features, dtype=object)
        
        # 3. Apply the exact same translations we did during training
        features[:, 2] = le.transform(features[:, 2]) # Translate Gender to 0 or 1
        features = ct.transform(features)             # One-Hot Encode Geography
        features = sc.transform(features)             # Scale all numbers down
        
        # 4. Make the prediction using the brain
        prediction = model.predict(features)
        
        # If the probability is > 0.5, they churn (1), else they stay (0)
        churn_result = int(prediction[0][0] > 0.5)
        
        # 5. Send the answer back to the user
        return jsonify({
            'churn_probability': float(prediction[0][0]),
            'will_churn': bool(churn_result)
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# Start the server
if __name__ == '__main__':
    app.run(debug=True)