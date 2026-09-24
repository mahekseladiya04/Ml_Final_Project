import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best_fraud_model.joblib')
METADATA_PATH = os.path.join(os.path.dirname(__file__), 'models', 'model_metadata.json')
EDA_PATH = os.path.join(os.path.dirname(__file__), 'models', 'eda_insights.json')

model_pipeline = None
model_metadata = {}
eda_insights = {}

def load_artifacts():
    global model_pipeline, model_metadata, eda_insights
    if os.path.exists(MODEL_PATH):
        model_pipeline = joblib.load(MODEL_PATH)
        print(f"[SERVER] Loaded model from {MODEL_PATH}")

    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            model_metadata = json.load(f)

    if os.path.exists(EDA_PATH):
        with open(EDA_PATH, 'r') as f:
            eda_insights = json.load(f)

load_artifacts()

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'AutoFraudShield ML Backend',
        'model_loaded': model_pipeline is not None,
        'model_name': model_metadata.get('best_model', 'Gradient Boosting')
    })

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    return jsonify(model_metadata)

@app.route('/api/eda-stats', methods=['GET'])
def get_eda_stats():
    return jsonify(eda_insights)

@app.route('/api/sample-claims', methods=['GET'])
def get_sample_claims():
    samples = [
        {
            'name': 'Legitimate Commuter Claim',
            'badge': 'Verified Low Risk',
            'data': {
                'age_of_driver': 45,
                'gender': 'M',
                'marital_status': 1,
                'safety_rating': 88,
                'annual_income': 65000.0,
                'high_education': 1,
                'address_change': 0,
                'property_status': 'Own',
                'claim_day_of_week': 'Wednesday',
                'accident_site': 'Highway',
                'past_num_of_claims': 0,
                'witness_present': 1,
                'liab_prct': 10,
                'channel': 'Broker',
                'police_report': 1,
                'age_of_vehicle': 3,
                'vehicle_category': 'Medium',
                'vehicle_price': 32000.0,
                'vehicle_color': 'gray',
                'total_claim': 8500.0,
                'injury_claim': 1200.0,
                'policy_deductible': 1000,
                'annual_premium': 1250.0,
                'days_open': 4.5,
                'form_defects': 1,
                'claim_month': 6,
                'claim_day': 14
            }
        },
        {
            'name': 'Moderate Inspection Claim',
            'badge': 'Audit Recommended',
            'data': {
                'age_of_driver': 28,
                'gender': 'F',
                'marital_status': 0,
                'safety_rating': 65,
                'annual_income': 38000.0,
                'high_education': 0,
                'address_change': 1,
                'property_status': 'Rent',
                'claim_day_of_week': 'Friday',
                'accident_site': 'Local',
                'past_num_of_claims': 2,
                'witness_present': 0,
                'liab_prct': 50,
                'channel': 'Phone',
                'police_report': 0,
                'age_of_vehicle': 7,
                'vehicle_category': 'Compact',
                'vehicle_price': 18500.0,
                'vehicle_color': 'silver',
                'total_claim': 14200.0,
                'injury_claim': 4800.0,
                'policy_deductible': 500,
                'annual_premium': 980.0,
                'days_open': 9.2,
                'form_defects': 4,
                'claim_month': 11,
                'claim_day': 22
            }
        },
        {
            'name': 'High-Risk Fraud Claim',
            'badge': 'Critical Alert',
            'data': {
                'age_of_driver': 22,
                'gender': 'M',
                'marital_status': 0,
                'safety_rating': 42,
                'annual_income': 18000.0,
                'high_education': 0,
                'address_change': 1,
                'property_status': 'Rent',
                'claim_day_of_week': 'Sunday',
                'accident_site': 'Parking Lot',
                'past_num_of_claims': 4,
                'witness_present': 0,
                'liab_prct': 95,
                'channel': 'Online',
                'police_report': 0,
                'age_of_vehicle': 11,
                'vehicle_category': 'Large',
                'vehicle_price': 14000.0,
                'vehicle_color': 'black',
                'total_claim': 36500.0,
                'injury_claim': 19500.0,
                'policy_deductible': 500,
                'annual_premium': 680.0,
                'days_open': 14.8,
                'form_defects': 9,
                'claim_month': 12,
                'claim_day': 28
            }
        }
    ]
    return jsonify(samples)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Parse fields
        age_of_driver = float(data.get('age_of_driver', 35))
        gender = str(data.get('gender', 'M')).strip().upper()
        marital_status = float(data.get('marital_status', 1))
        safety_rating = float(data.get('safety_rating', 75))
        annual_income = float(data.get('annual_income', 50000.0))
        high_education = float(data.get('high_education', 1))
        address_change = float(data.get('address_change', 0))
        property_status = str(data.get('property_status', 'Own')).strip().capitalize()
        claim_day_of_week = str(data.get('claim_day_of_week', 'Wednesday')).strip()
        accident_site = str(data.get('accident_site', 'Highway')).strip()
        past_num_of_claims = float(data.get('past_num_of_claims', 0))
        witness_present = float(data.get('witness_present', 1))
        liab_prct = float(data.get('liab_prct', 20))
        channel = str(data.get('channel', 'Broker')).strip()
        police_report = float(data.get('police_report', 1))
        age_of_vehicle = float(data.get('age_of_vehicle', 4))
        vehicle_category = str(data.get('vehicle_category', 'Medium')).strip()
        vehicle_price = float(data.get('vehicle_price', 25000.0))
        vehicle_color = str(data.get('vehicle_color', 'silver')).strip().lower()
        total_claim = float(data.get('total_claim', 15000.0))
        injury_claim = float(data.get('injury_claim', 3000.0))
        policy_deductible = float(data.get('policy_deductible', 1000))
        annual_premium = float(data.get('annual_premium', 1200.0))
        days_open = float(data.get('days_open', 7.5))
        form_defects = float(data.get('form_defects', 2))
        claim_month = int(data.get('claim_month', 6))
        claim_day = int(data.get('claim_day', 15))

        # Computed features
        injury_to_total_ratio = injury_claim / (total_claim + 1e-5)
        claim_to_price_ratio = total_claim / (vehicle_price + 1e-5)
        income_to_price_ratio = annual_income / (vehicle_price + 1e-5)

        row = {
            'age_of_driver': age_of_driver,
            'gender': gender,
            'marital_status': marital_status,
            'safety_rating': safety_rating,
            'annual_income': annual_income,
            'high_education': high_education,
            'address_change': address_change,
            'property_status': property_status,
            'claim_day_of_week': claim_day_of_week,
            'accident_site': accident_site,
            'past_num_of_claims': past_num_of_claims,
            'witness_present': witness_present,
            'liab_prct': liab_prct,
            'channel': channel,
            'police_report': police_report,
            'age_of_vehicle': age_of_vehicle,
            'vehicle_category': vehicle_category,
            'vehicle_price': vehicle_price,
            'vehicle_color': vehicle_color,
            'total_claim': total_claim,
            'injury_claim': injury_claim,
            'policy deductible': policy_deductible,
            'annual premium': annual_premium,
            'days open': days_open,
            'form defects': form_defects,
            'claim_month': claim_month,
            'claim_day': claim_day,
            'injury_to_total_ratio': injury_to_total_ratio,
            'claim_to_price_ratio': claim_to_price_ratio,
            'income_to_price_ratio': income_to_price_ratio
        }

        df_input = pd.DataFrame([row])

        if model_pipeline is None:
            load_artifacts()
            if model_pipeline is None:
                return jsonify({'error': 'Model pipeline not loaded on server'}), 500

        # Standard model classification prediction & probability
        raw_pred = int(model_pipeline.predict(df_input)[0])
        probs = model_pipeline.predict_proba(df_input)[0]
        fraud_prob = float(probs[1])
        legit_prob = float(probs[0])
        
        if fraud_prob < 0.28:
            risk_level = 'Low Risk'
            risk_color = '#10b981' # emerald
            recommendation = 'Straight-Through Processing (STP) approved. No fraudulent indicators detected.'
        elif fraud_prob < 0.52:
            risk_level = 'Moderate Attention'
            risk_color = '#f59e0b' # amber
            recommendation = 'Secondary desk review recommended. Verify vehicle damage assessment and witness statement.'
        else:
            risk_level = 'Critical Fraud Alert'
            risk_color = '#ef4444' # red
            recommendation = 'Flag for Special Investigation Unit (SIU). Conduct physical damage inspection and interview claimant.'

        # Determine Specific Risk Drivers
        risk_factors = []
        if injury_to_total_ratio > 0.45:
            risk_factors.append(f'Elevated personal injury ratio ({injury_to_total_ratio*100:.1f}% of total payout).')
        if claim_to_price_ratio > 1.2:
            risk_factors.append(f'Claim sum (${total_claim:,.2f}) significantly exceeds vehicle value (${vehicle_price:,.2f}).')
        if form_defects >= 5:
            risk_factors.append(f'Abnormal documentation errors ({int(form_defects)} defects flagged).')
        if days_open > 12.0:
            risk_factors.append(f'Prolonged claim settlement duration ({days_open:.1f} days open).')
        if annual_income < 25000:
            risk_factors.append('Low reported income relative to claim value and policy premium.')
        if past_num_of_claims >= 3:
            risk_factors.append(f'High frequency of prior claims ({int(past_num_of_claims)} previous claims).')
        if witness_present == 0 and police_report == 0:
            risk_factors.append('Lack of official corroboration (no police report and no witnesses).')

        if not risk_factors:
            risk_factors.append('Claim attributes fall within normal standard actuarial thresholds.')

        response_payload = {
            'prediction': raw_pred,
            'prediction_label': 'Fraudulent Claim' if raw_pred == 1 else 'Legitimate Claim',
            'fraud_probability': round(fraud_prob * 100, 2),
            'legitimate_probability': round(legit_prob * 100, 2),
            'confidence_score': round(max(fraud_prob, legit_prob) * 100, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'recommendation': recommendation,
            'risk_factors': risk_factors,
            'calculated_ratios': {
                'injury_to_total_ratio': round(injury_to_total_ratio, 4),
                'claim_to_price_ratio': round(claim_to_price_ratio, 4),
                'income_to_price_ratio': round(income_to_price_ratio, 4)
            }
        }

        return jsonify(response_payload)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[SERVER] Starting AutoFraudShield Backend on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
