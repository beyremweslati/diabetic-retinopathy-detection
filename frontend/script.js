const API_URL = '/api';

let availableModels = [];
let selectedImage = null;

document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    loadModels();
    loadResults();
    setupEventListeners();
});

function initializeTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.dataset.tab;

            // Remove active class from all
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // Add active class to selected
            button.classList.add('active');
            document.getElementById(tabId).classList.add('active');

            // Load data if needed
            if (tabId === 'results' || tabId === 'comparison') {
                loadResults();
            }
        });
    });
}

async function loadModels() {
    try {
        const response = await fetch(`${API_URL}/models`);
        const data = await response.json();

        availableModels = data.models;

        const select = document.getElementById('model-select');
        select.innerHTML = '';

        if (availableModels.length === 0) {
            select.innerHTML = '<option value="">Aucun modèle disponible</option>';
            return;
        }

        availableModels.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = formatModelName(model);
            select.appendChild(option);
        });

        // Select first model by default
        if (availableModels.length > 0) {
            select.value = availableModels[0];
        }

    } catch (error) {
        console.error('Erreur lors du chargement des modèles:', error);
        document.getElementById('model-select').innerHTML = '<option value="">Erreur de chargement</option>';
    }
}

function formatModelName(name) {
    return name
        .replace('spark_', '')
        .replace('_', ' ')
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

function setupEventListeners() {
    const predictBtn = document.getElementById('predict-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const imageInput = document.getElementById('image-input');

    if (predictBtn) {
        predictBtn.addEventListener('click', makePrediction);
    }

    if (uploadBtn && imageInput) {
        uploadBtn.addEventListener('click', (e) => {
            e.preventDefault();
            imageInput.click();
        });

        imageInput.addEventListener('change', handleImageUpload);
    }
}

function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    selectedImage = file;

    // Show preview
    const preview = document.getElementById('image-preview');
    const previewImg = document.getElementById('preview-img');
    const imageName = document.getElementById('image-name');

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        imageName.textContent = file.name;
        preview.style.display = 'block';

        document.getElementById('predict-btn').disabled = false;
    };
    reader.readAsDataURL(file);
}

// Make Prediction
async function makePrediction() {
    const modelSelect = document.getElementById('model-select');
    const predictBtn = document.getElementById('predict-btn');
    const resultCard = document.getElementById('prediction-result');

    const model = modelSelect.value;

    if (!model) {
        alert('Veuillez sélectionner un modèle');
        return;
    }

    if (!selectedImage) {
        alert('Veuillez télécharger une image');
        return;
    }

    // Show loading
    predictBtn.disabled = true;
    predictBtn.querySelector('.btn-text').style.display = 'none';
    predictBtn.querySelector('.btn-loader').style.display = 'inline';
    resultCard.style.display = 'none';

    try {
        // Create FormData to send image
        const formData = new FormData();
        formData.append('image', selectedImage);
        formData.append('model', model);

        const response = await fetch(`${API_URL}/predict-image`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Erreur HTTP: ${response.status}`);
        }

        const result = await response.json();
        displayPredictionResult(result);

    } catch (error) {
        console.error('Erreur lors de la prédiction:', error);
        alert('Erreur lors de la prédiction. Veuillez réessayer.');
    } finally {
        predictBtn.disabled = false;
        predictBtn.querySelector('.btn-text').style.display = 'inline';
        predictBtn.querySelector('.btn-loader').style.display = 'none';
    }
}

// Display Prediction Result
function displayPredictionResult(result) {
    const resultCard = document.getElementById('prediction-result');
    const badge = document.getElementById('result-badge');
    const testResult = document.getElementById('test-result');
    const modelUsed = document.getElementById('model-used');

    // Set model name
    modelUsed.textContent = formatModelName(result.model);

    // Set test result - Simple Positif/Négatif
    if (result.prediction === 1) {
        badge.textContent = 'Positif';
        badge.className = 'result-badge positive';
        testResult.textContent = 'Test Positif - Rétinopathie Diabétique Détectée';
        testResult.style.color = '#f44336';
        testResult.style.fontWeight = '600';
    } else {
        badge.textContent = 'Négatif';
        badge.className = 'result-badge negative';
        testResult.textContent = 'Test Négatif - Pas de Rétinopathie Diabétique';
        testResult.style.color = '#4caf50';
        testResult.style.fontWeight = '600';
    }

    // Show result card
    resultCard.style.display = 'block';
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}



// Load Results
async function loadResults() {
    try {
        const response = await fetch(`${API_URL}/results`);
        const data = await response.json();

        // Display Spark results
        if (data.spark) {
            displayModelResults(data.spark, 'spark-results');
        }

        // Display sklearn results
        if (data.sklearn) {
            displayModelResults(data.sklearn, 'sklearn-results');
        }

        // Display comparison
        if (data.comparison) {
            displayComparison(data.comparison);
        }

    } catch (error) {
        console.error('Erreur lors du chargement des résultats:', error);
        document.getElementById('spark-results').innerHTML = '<div class="loading">Erreur de chargement</div>';
        document.getElementById('sklearn-results').innerHTML = '<div class="loading">Erreur de chargement</div>';
    }
}

// Display Model Results
function displayModelResults(results, containerId) {
    const container = document.getElementById(containerId);

    if (!results || results.length === 0) {
        container.innerHTML = '<div class="loading">Aucun résultat disponible</div>';
        return;
    }

    let html = '';

    results.forEach(result => {
        html += `
            <div class="metric-card">
                <div class="metric-name">${result.model}</div>
                <div class="metric-grid">
                    <div class="metric-item">
                        <span class="metric-label">Accuracy</span>
                        <span class="metric-value">${(result.accuracy * 100).toFixed(2)}%</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Precision</span>
                        <span class="metric-value">${(result.precision * 100).toFixed(2)}%</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Recall</span>
                        <span class="metric-value">${(result.recall * 100).toFixed(2)}%</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">F1-Score</span>
                        <span class="metric-value">${(result.f1_score * 100).toFixed(2)}%</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">AUC-ROC</span>
                        <span class="metric-value">${(result.auc_roc * 100).toFixed(2)}%</span>
                    </div>
                    ${result.training_time ? `
                    <div class="metric-item">
                        <span class="metric-label">Temps</span>
                        <span class="metric-value">${result.training_time.toFixed(2)}s</span>
                    </div>
                    ` : ''}
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

// Display Comparison
function displayComparison(comparison) {
    const container = document.getElementById('comparison-table');
    const insightsContainer = document.getElementById('insights');

    if (!comparison || comparison.length === 0) {
        container.innerHTML = '<div class="loading">Aucune comparaison disponible</div>';
        return;
    }

    // Create table
    let html = `
        <table>
            <thead>
                <tr>
                    <th>Modèle</th>
                    <th>Framework</th>
                    <th>Accuracy</th>
                    <th>F1-Score</th>
                    <th>AUC-ROC</th>
                    <th>Temps (s)</th>
                </tr>
            </thead>
            <tbody>
    `;

    comparison.forEach(row => {
        html += `
            <tr>
                <td>${row.Model}</td>
                <td>${row.Framework}</td>
                <td>${(row.Accuracy * 100).toFixed(2)}%</td>
                <td>${(row['F1-Score'] * 100).toFixed(2)}%</td>
                <td>${(row['AUC-ROC'] * 100).toFixed(2)}%</td>
                <td>${row['Training Time'].toFixed(2)}</td>
            </tr>
        `;
    });

    html += `
            </tbody>
        </table>
    `;

    container.innerHTML = html;

    generateInsights(comparison, insightsContainer);
}

function generateInsights(comparison, container) {
    const insights = [];

    const bestByF1 = comparison.reduce((best, current) => {
        return (current['F1-Score'] > best['F1-Score']) ? current : best;
    });

    insights.push({
        icon: '🏆',
        title: 'Meilleur Modèle',
        text: `${bestByF1.Model} (${bestByF1.Framework}) avec un F1-Score de ${(bestByF1['F1-Score'] * 100).toFixed(2)}%`
    });

    // Compare frameworks
    const sparkModels = comparison.filter(m => m.Framework === 'Spark MLlib');
    const sklearnModels = comparison.filter(m => m.Framework === 'scikit-learn');

    if (sparkModels.length > 0 && sklearnModels.length > 0) {
        const avgSparkTime = sparkModels.reduce((sum, m) => sum + m['Training Time'], 0) / sparkModels.length;
        const avgSklearnTime = sklearnModels.reduce((sum, m) => sum + m['Training Time'], 0) / sklearnModels.length;

        const faster = avgSparkTime < avgSklearnTime ? 'Spark MLlib' : 'scikit-learn';
        const speedup = Math.abs(avgSparkTime - avgSklearnTime).toFixed(2);

        insights.push({
            icon: '⚡',
            title: 'Performance',
            text: `${faster} est plus rapide de ${speedup} secondes en moyenne`
        });
    }

    // Best accuracy
    const bestAccuracy = comparison.reduce((best, current) => {
        return (current.Accuracy > best.Accuracy) ? current : best;
    });

    insights.push({
        icon: '🎯',
        title: 'Meilleure Précision',
        text: `${bestAccuracy.Model} atteint ${(bestAccuracy.Accuracy * 100).toFixed(2)}% d'accuracy`
    });

    let html = '';
    insights.forEach(insight => {
        html += `
            <div class="insight-card">
                <div class="insight-icon">${insight.icon}</div>
                <div class="insight-title">${insight.title}</div>
                <div class="insight-text">${insight.text}</div>
            </div>
        `;
    });

    container.innerHTML = html;
}
