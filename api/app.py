from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
import cv2
from PIL import Image
import io
from scipy import stats
from skimage import feature
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

app = Flask(__name__)
CORS(app)  # Activer CORS pour le frontend

# Chemins des modèles
MODEL_DIR = '/home/models'

# Initialiser Spark session
spark = None

# Charger les modèles au démarrage
models = {}
scalers = {}

def init_spark():
    """Initialise une session Spark"""
    global spark
    try:
        spark = SparkSession.builder \
            .appName("RetinopathyAPI") \
            .master("local[1]") \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "2g") \
            .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse") \
            .config("spark.driver.host", "127.0.0.1") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("ERROR")
        print("Spark session initialisée avec succès")
        return True
    except Exception as e:
        print(f"Erreur d'initialisation Spark: {e}")
        return False

def load_models():
    """Charge tous les modèles Spark sauvegardés"""
    global models
    
    if not init_spark():
        print("Impossible d'initialiser Spark")
        return
    
    spark_models = [
        "logistic_regression",
        "random_forest",
        "svm"
    ]

    for model_name in spark_models:
        model_path = os.path.join(MODEL_DIR, f"spark_{model_name}")
        if os.path.exists(model_path):
            try:
                models[model_name] = PipelineModel.load(model_path)
                print(f"Modèle Spark chargé: {model_name}")
            except Exception as e:
                print(f"Erreur chargement {model_name}: {e}")

    print(f"\nModèles Spark chargés: {len(models)}")

# Charger les modèles au démarrage
load_models()

@app.route('/models', methods=['GET'])
def get_models():
    """Liste des modèles disponibles"""
    return jsonify({
        'models': list(models.keys()),
        'count': len(models)
    })

def extract_image_features(image_array):
    try:
        img_resized = cv2.resize(image_array, (256, 256))
        
        features = []
        
        for i in range(3):
            channel = img_resized[:,:,i]
            features.append(np.mean(channel))
            features.append(np.std(channel))
            features.append(stats.skew(channel.flatten()))
        
        img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
        for i in range(3):
            channel = img_hsv[:,:,i]
            features.append(np.mean(channel))
            features.append(np.std(channel))
            features.append(stats.skew(channel.flatten()))
        
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        
        hist, _ = np.histogram(img_gray.flatten(), bins=10, range=(0, 256))
        hist = hist.astype(float) / hist.sum()
        features.extend(hist)
        
        features.append(img_gray.std())
        features.append(img_gray.mean())
        features.append(float(img_gray.min()))
        features.append(float(img_gray.max()))
        
        features.append(stats.entropy(hist))
        
        moments = cv2.HuMoments(cv2.moments(img_gray)).flatten()
        for m in moments:
            features.append(-np.sign(m) * np.log10(np.abs(m) + 1e-10))
        
        edges = cv2.Canny(img_gray, 100, 200)
        features.append(np.sum(edges > 0) / edges.size)
        features.append(np.mean(edges))
        
        return np.array(features)
        
    except Exception as e:
        print(f"Erreur extraction features: {e}")
        return None

@app.route('/predict-image', methods=['POST'])
def predict_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Aucune image fournie'}), 400
        
        if 'model' not in request.form:
            return jsonify({'error': 'Modèle non spécifié'}), 400
        
        model_name = request.form['model']
        image_file = request.files['image']
        
        if model_name not in models:
            return jsonify({
                'error': f'Modèle {model_name} non disponible',
                'available_models': list(models.keys())
            }), 404
        
        # Lire l'image
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image.convert('RGB'))
        
        # Extraire les features
        features = extract_image_features(image_array)
        
        from pyspark.ml.linalg import Vectors
        features_vector = Vectors.dense(features.tolist())
        df = spark.createDataFrame([(features_vector,)], ['features'])
        
        # Faire la prédiction
        model = models[model_name]
        predictions = model.transform(df)
        
        prediction_row = predictions.select('prediction').collect()[0]
        prediction = int(prediction_row['prediction'])
        print(f"DEBUG: Prediction={prediction}")
        
        is_positive = int(prediction) >= 1
        
        # Résultat
        result = {
            'model': model_name,
            'prediction': 1 if is_positive else 0,  # 0=Négatif, 1=Positif
            'class_predicted': int(prediction),  # Classe originale (0-4)
            'diagnosis': 'Rétinopathie diabétique détectée' if is_positive else 'Pas de rétinopathie diabétique',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results', methods=['GET'])
def get_results():
    """
    Obtenir les résultats de comparaison des modèles
    """
    try:
        results = {}
        
        # Charger les résultats Spark
        spark_results_path = os.path.join(MODEL_DIR, 'spark_results.csv')
        if os.path.exists(spark_results_path):
            spark_df = pd.read_csv(spark_results_path)
            results['spark'] = spark_df.to_dict('records')
        
        # Charger les résultats sklearn
        sklearn_results_path = os.path.join(MODEL_DIR, 'sklearn_results.csv')
        if os.path.exists(sklearn_results_path):
            sklearn_df = pd.read_csv(sklearn_results_path)
            results['sklearn'] = sklearn_df.to_dict('records')
        
        # Charger la comparaison
        comparison_path = os.path.join(MODEL_DIR, 'comparison_results.csv')
        if os.path.exists(comparison_path):
            comparison_df = pd.read_csv(comparison_path)
            results['comparison'] = comparison_df.to_dict('records')
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print(f"Modèles disponibles: {list(models.keys())}")
    print("API disponible sur: http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
