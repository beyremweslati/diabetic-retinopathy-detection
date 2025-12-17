# Détection de Rétinopathie Diabétique avec Apache Spark

Projet de machine learning utilisant **Apache Spark MLlib** pour détecter la rétinopathie diabétique à partir d'images de rétine, déployé avec Docker et servi via une API Flask.

## Description

Ce projet implémente un système complet de détection de rétinopathie diabétique utilisant l'apprentissage automatique distribué :

- **Apache Spark MLlib** pour l'entraînement distribué de modèles (Régression Logistique, Random Forest, SVM)
- **API REST Flask** pour servir les prédictions
- **Extraction de caractéristiques** à partir d'images de rétine (statistiques RGB, HSV, détection de contours, moments de Hu)
- **Docker Compose** pour un déploiement et une orchestration faciles
- **Interface web** pour la comparaison de modèles et les prédictions

Le système entraîne des modèles de classification binaire pour détecter la présence de rétinopathie diabétique à partir de caractéristiques d'images de rétine prétraitées.

## 📊 Dataset

**Source :** Dataset de Détection de Rétinopathie Diabétique (https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data)

**Total d'images :** 3662 images de rétine en couleur  
**Classes :** 5 niveaux de sévérité

- No DR (Sain) : 1805 images
- Mild (Léger) : 370 images
- Moderate (Modéré) : 999 images
- Severe (Sévère) : 193 images
- Proliferative DR (Prolifératif) : 295 images

**Classification binaire :** Les images sont étiquetées comme saines (0) ou présentant une rétinopathie diabétique (1-4).

**Caractéristiques :** 42 caractéristiques extraites par image incluant :

- Statistiques de couleur RGB et HSV
- Caractéristiques d'histogramme
- Détection de contours (Canny)
- Moments de Hu
- Métriques de contraste et de luminosité

## Démarrage

### 1. Cloner le projet

```bash
git clone https://github.com/beyremweslati/diabetic-retinopathy-detection.git
```

### 2. Construire les Images Docker

```bash
docker compose build
```

Cela construira toutes les images nécessaires :

- Cluster Spark (master + 2 workers)
- Environnement Jupyter Notebook
- Serveur API Flask
- Serveur web frontend

### 3. Démarrer Tous les Services

```bash
docker compose up -d
```

## Entraînement des Modèles

### Option 1 : Utiliser les Notebooks Jupyter

1. Accéder à Jupyter sur **http://localhost:8888**
2. Naviguer vers `notebooks/`
3. Exécuter les notebooks dans l'ordre :
   - `data_preparation.ipynb` - Extrait les caractéristiques des images
   - `train_spark_models.ipynb` - Entraîne les modèles Spark MLlib
   - `train_sklearn_models.ipynb` - Entraîne les modèles scikit-learn (optionnel, pour comparaison)

## 🌐 Points d'Accès

| Service             | URL                   | Description                        |
| ------------------- | --------------------- | ---------------------------------- |
| **Frontend**        | http://localhost:3000 | Interface web pour les prédictions |
| **API Flask**       | http://localhost:5000 | Endpoints de l'API REST            |
| **Jupyter**         | http://localhost:8888 | Environnement de notebooks         |
| **Spark Master UI** | http://localhost:8080 | Monitoring du cluster Spark        |
| **Spark Worker 1**  | http://localhost:8081 | Statut du worker                   |
| **Spark Worker 2**  | http://localhost:8082 | Statut du worker                   |

Trois modèles de machine learning sont entraînés avec **Apache Spark MLlib** :

| Modèle                    | Description                   |
| ------------------------- | ----------------------------- |
| **Régression Logistique** | Classification linéaire       |
| **Random Forest**         | Apprentissage par ensemble    |
| **SVM Linéaire**          | Machine à vecteurs de support |

Tous les modèles utilisent :

- **StandardScaler** pour la normalisation des caractéristiques
- **Validation croisée** pour l'ajustement des hyperparamètres
- **Classification binaire** (sain vs. rétinopathie diabétique)

## 📊 Endpoints de l'API

### Endpoints Principaux

```bash
GET  /models              # Lister les modèles disponibles
POST /predict-image       # Prédire à partir d'une image uploadée
GET  /results             # Obtenir les résultats d'entraînement
```

## 📁 Structure du Projet

```
retinopathie/
├── docker-compose.yml          # Orchestration des services
├── Dockerfile.spark            # Image du cluster Spark
├── Dockerfile.jupyter          # Environnement Jupyter
├── Dockerfile.flask            # Image de l'API Flask
├── Dockerfile.frontend         # Image du frontend web
│
├── dataset2/                   # Répertoire du dataset
│   ├── train.csv               # Labels des images
│   ├── colored_images/         # Images de rétine (5 classes)
│   └── processed/              # Caractéristiques traitées (train/val/test)
│
├── notebooks/                  # Notebooks Jupyter
│   ├── data_preparation.ipynb
│   ├── train_spark_models.ipynb
│   └── train_sklearn_models.ipynb
│
├── models/                     # Modèles entraînés
│   ├── spark_logistic_regression/
│   ├── spark_random_forest/
│   ├── spark_svm/
│   └── spark_results.csv
│
├── api/                        # API Flask
│   └── app.py
│
└── frontend/                   # Interface web
    ├── index.html
    ├── styles.css
    └── script.js
```

## 🛠️ Stack Technologique

- **Apache Spark 3.5.0** - Calcul distribué
- **PySpark MLlib** - Algorithmes de machine learning
- **Flask** - Framework d'API REST
- **Docker & Docker Compose** - Conteneurisation
- **Jupyter Notebook** - Développement interactif
- **OpenCV & scikit-image** - Traitement d'images
- **NumPy & Pandas** - Manipulation de données
