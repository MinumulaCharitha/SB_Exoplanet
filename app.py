import pandas as pd
import numpy as np
import shap
from flask import Flask, request, jsonify, render_template
import joblib
from flask_sqlalchemy import SQLAlchemy

# ---------------- Flask App Setup ----------------
app = Flask(__name__)

# ---------------- Database Configuration ----------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///exoplanets.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ---------------- API Key for GET requests ----------------
API_KEY = "mysecretkey"

# ---------------- Database Model ----------------
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    planet_name = db.Column(db.String(100))
    predicted_class = db.Column(db.Integer)
    model_used = db.Column(db.String(10))

    def __repr__(self):
        return f'<Prediction {self.planet_name} - {self.predicted_class}>'

# ---------------- Load Models & Preprocessor ----------------
xgb_model = joblib.load('xgb_model.pkl')
svm_model = joblib.load('svm_model.pkl')
preprocessor = joblib.load('preprocessor_fixed.pkl')

# SHAP Explainer (created ONCE)
explainer = shap.TreeExplainer(xgb_model)

# ---------------- Feature Lists ----------------
num_features = preprocessor.transformers_[0][2]
cat_features = preprocessor.transformers_[1][2]
all_features = list(num_features) + list(cat_features)

print("Numerical features:", num_features)
print("Categorical features:", cat_features)

# ---------------- Home Route ----------------
@app.route('/')
def home():
    return render_template('index.html')

# ---------------- Prediction API ----------------
@app.route('/predict_habitability', methods=['POST'])
def predict_habitability():
    try:
        # --- Step 1: Get input JSON ---
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # --- Step 2: Convert input to DataFrame ---
        input_df = pd.DataFrame([data])

        # --- Step 3: Fill missing columns ---
        for col in all_features:
            if col not in input_df.columns:
                input_df[col] = 0 if col in num_features else 'missing'

        # --- Step 4: Reorder columns ---
        input_df = input_df[all_features]

        # --- Step 5: Transform features ---
        X_input = preprocessor.transform(input_df)

        # --- Step 6: Choose model ---
        model_name = data.get('model', 'xgb').lower()
        if model_name == 'xgb':
            model = xgb_model
        elif model_name == 'svm':
            model = svm_model
        else:
            return jsonify({'error': f'Model "{model_name}" not supported'}), 400

        # --- Step 7: Make prediction ---
        prediction = int(model.predict(X_input)[0])

        # --- Step 8: Prediction probabilities ---
        try:
            prediction_proba = model.predict_proba(X_input).tolist()
        except AttributeError:
            prediction_proba = "Not available"

        # ---------- SHAP EXPLANATION (ONLY FOR XGBOOST) ----------
        shap_features = {}

        if model_name == 'xgb':
            # Convert sparse to dense
            if hasattr(X_input, "toarray"):
                X_dense = X_input.toarray()
            else:
                X_dense = X_input

            feature_names = preprocessor.get_feature_names_out()
            X_dense_df = pd.DataFrame(X_dense, columns=feature_names)

            shap_values = explainer.shap_values(X_dense_df)

            # Handle multiclass
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            shap_contrib = (
                pd.Series(shap_values[0], index=feature_names)
                .sort_values(key=abs, ascending=False)
                .head(5)
            )

            shap_features = shap_contrib.to_dict()

        # --- Step 9: Save prediction to DB ---
        planet_name = data.get('P_NAME', 'Unknown')
        new_pred = Prediction(
            planet_name=planet_name,
            predicted_class=prediction,
            model_used=model_name
        )
        db.session.add(new_pred)
        db.session.commit()

        # --- Step 10: Return response ---
        return jsonify({
            'success': True,
            'planet_name': planet_name,
            'predicted_class': prediction,
            'prediction_proba': prediction_proba,
            'shap_explanation': shap_features
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ---------------- Retrieve all predictions ----------------
@app.route('/predictions', methods=['GET'])
def get_predictions():
    api_key = request.headers.get('x-api-key')
    if api_key != API_KEY:
        return jsonify({'error': 'Unauthorized', 'success': False}), 401

    all_preds = Prediction.query.all()
    results = []
    for pred in all_preds:
        results.append({
            'id': pred.id,
            'planet_name': pred.planet_name,
            'predicted_class': pred.predicted_class,
            'model_used': pred.model_used
        })

    return jsonify({'success': True, 'predictions': results})

# ---------------- Run App ----------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
