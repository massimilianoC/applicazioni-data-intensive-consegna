
# Applicazione Web Flask per Predizione Consumi Energetici
# Studente: Massimiliano Camillucci - Matricola: 0001087333
# Corso: Programmazione di Applicazioni Data Intensive A.A. 2024/2025

from flask import Flask, render_template, request, jsonify
import sys
import os
import joblib
import json
import numpy as np
import pandas as pd
import random

app = Flask(__name__)

model = None
scaler = None
encoders = None
metadata = None
feature_ranges = None

def load_model_components():
    global model, scaler, encoders, metadata, feature_ranges
    
    try:
        required_files = [
            'energy_prediction_model.pkl',
            'scaler.pkl', 
            'label_encoders.pkl',
            'model_metadata.json',
            'feature_ranges.json'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            raise FileNotFoundError(f"File mancanti: {missing_files}")
        
        model = joblib.load('energy_prediction_model.pkl')
        scaler = joblib.load('scaler.pkl')
        encoders = joblib.load('label_encoders.pkl')
        
        with open('model_metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        with open('feature_ranges.json', 'r', encoding='utf-8') as f:
            feature_ranges = json.load(f)
        
        print("Modello e preprocessori caricati con successo!")
        print(f"Modello: {metadata.get('model_type', 'N/A')}")
        print(f"R²: {metadata.get('performance', {}).get('test_r2', 'N/A'):.4f}")
        return True
        
    except Exception as e:
        print(f"Errore nel caricamento: {e}")
        print("Assicurati di aver eseguito prima il notebook per generare i file del modello")
        return False

model_loaded = load_model_components()

@app.route('/')
def home():
    """Form per inserimento dati"""
    if not model_loaded:
        return """
        <html><body style="font-family: Arial; text-align: center; padding: 50px;">
        <h1>Errore Caricamento Modello</h1>
        <p>I file del modello non sono stati trovati.</p>
        <p>Esegui prima il notebook Jupyter per generare i file necessari.</p>
        <hr>
        <small>Programmazione di Applicazioni Data Intensive - A.A. 2024/2025</small>
        </body></html>
        """
    
    return render_template('index.html', 
                         metadata=metadata, 
                         feature_ranges=feature_ranges,
                         feature_descriptions=metadata.get('feature_descriptions', {}))

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint previsioni"""
    if not model_loaded:
        return jsonify({
            'success': False, 
            'error': 'Modello non caricato. Esegui prima il notebook.'
        })
    
    try:
        input_data = {}
        
        for feature in metadata['numeric_features']:
            value = request.form.get(feature)
            if value is None or value == '':
                return jsonify({
                    'success': False, 
                    'error': f'Valore mancante per {feature}'
                })
            try:
                input_data[feature] = float(value)
            except ValueError:
                return jsonify({
                    'success': False, 
                    'error': f'Valore non valido per {feature}: {value}'
                })
        
        for feature in metadata['categorical_features']:
            value = request.form.get(feature)
            if value is None or value == '':
                input_data[feature] = feature_ranges[feature]['values'][0]
            else:
                if value not in feature_ranges[feature]['values']:
                    return jsonify({
                        'success': False, 
                        'error': f'Valore non valido per {feature}: {value}'
                    })
                input_data[feature] = value
        
        try:
            input_df = pd.DataFrame([input_data])
            input_df = input_df[metadata['feature_names']]
        except KeyError as e:
            return jsonify({
                'success': False, 
                'error': f'Errore struttura dati: {e}'
            })
        
        input_processed = input_df.copy()
        
        for feature in metadata['categorical_features']:
            if feature in encoders:
                try:
                    encoded_value = encoders[feature].transform([input_data[feature]])[0]
                    input_processed[feature] = encoded_value
                except ValueError as e:
                    return jsonify({
                        'success': False, 
                        'error': f'Errore encoding {feature}: {e}'
                    })
        
        try:
            numeric_features_array = input_processed[metadata['numeric_features']].values
            scaled_features = scaler.transform(numeric_features_array)
            input_processed[metadata['numeric_features']] = scaled_features
        except Exception as e:
            return jsonify({
                'success': False, 
                'error': f'Errore scaling: {e}'
            })
        
        try:
            prediction = model.predict(input_processed)[0]
            
            if np.isnan(prediction) or np.isinf(prediction):
                return jsonify({
                    'success': False, 
                    'error': 'Predizione non valida'
                })
            
        except Exception as e:
            return jsonify({
                'success': False, 
                'error': f'Errore predizione: {e}'
            })
        
        rmse = metadata['performance']['test_rmse']
        confidence_interval = [
            max(0, prediction - rmse),  
            prediction + rmse
        ]
        
        return jsonify({
            'success': True,
            'prediction': round(float(prediction), 2),
            'confidence_interval': [round(float(ci), 2) for ci in confidence_interval],
            'input_data': input_data,
            'model_info': {
                'type': metadata['model_type'],
                'r2_score': metadata['performance']['test_r2'],
                'rmse': metadata['performance']['test_rmse']
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': f'Errore generale: {str(e)}'
        })

@app.route('/random_data')
def random_data():
    """Generare dati casuali"""
    if not model_loaded:
        return jsonify({
            'success': False, 
            'error': 'Modello non caricato. Esegui prima il notebook.'
        })
    
    try:
        random_input = {}
        
        for feature in metadata['numeric_features']:
            if feature in feature_ranges:
                mean = feature_ranges[feature]['mean']
                std = feature_ranges[feature]['std']
                min_val = feature_ranges[feature]['min']
                max_val = feature_ranges[feature]['max']
                value = np.random.normal(mean, std/2)
                value = max(min_val, min(max_val, value))
                random_input[feature] = round(float(value), 4)
        
        for feature in metadata['categorical_features']:
            if feature in feature_ranges:
                random_input[feature] = random.choice(feature_ranges[feature]['values'])
        
        return jsonify({
            'success': True, 
            'data': random_input,
            'info': 'Dati generati nei range realistici del dataset'
        })
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': f'Errore generazione dati casuali: {str(e)}'
        })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint non trovato'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Errore interno del server'}), 500

if __name__ == '__main__':
    print("Avvio server Flask ")
    print("Apri il browser su: http://localhost:5000")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Errore avvio server: {e}")
        print("Verifica che la porta 5000 non sia già in uso")
