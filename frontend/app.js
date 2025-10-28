// frontend/app.js
class BiasDetectionApp {
    constructor() {
        this.apiBase = window.location.origin;
        this.currentResults = null;
        this.init();
    }

    async init() {
        await this.loadSupportedModels();
        this.setupEventListeners();
    }

    async loadSupportedModels() {
        try {
            console.log('Loading models from:', `${this.apiBase}/api/models`);
            const response = await fetch(`${this.apiBase}/api/models`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log('Models response:', data);
            
            if (data.models && Array.isArray(data.models)) {
                const select = document.getElementById('modelSelect');
                select.innerHTML = '<option value="">Select a model...</option>';
                
                data.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    select.appendChild(option);
                });
                
                console.log(`Loaded ${data.models.length} models successfully`);
            } else if (data.error) {
                throw new Error(data.error);
            } else {
                throw new Error('Invalid response format');
            }
        } catch (error) {
            console.error('Error loading models:', error);
            this.showError(`Failed to load supported models: ${error.message}`);
            
            // Fallback: show some default models
            const select = document.getElementById('modelSelect');
            select.innerHTML = `
                <option value="">Select a model...</option>
                <option value="gpt-4o">GPT-4o</option>
                <option value="gpt-4o-mini">GPT-4o Mini</option>
                <option value="gemini-pro">Gemini Pro</option>
            `;
        }
    }

    setupEventListeners() {
        document.getElementById('analyzeBtn').addEventListener('click', () => {
            this.analyzeModel();
        });

        // Enter key support for custom model input
        document.getElementById('customModel').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.analyzeModel();
            }
        });
    }

    async analyzeModel() {
        const modelSelect = document.getElementById('modelSelect');
        const customModel = document.getElementById('customModel');
        
        let modelName = modelSelect.value || customModel.value.trim();
        
        if (!modelName) {
            this.showError('Please select or enter a model name');
            return;
        }

        this.showLoading();
        this.hideResults();
        this.hideError();

        try {
            const response = await fetch(`${this.apiBase}/api/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ model_name: modelName })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Analysis failed');
            }

            this.currentResults = data;
            this.displayResults(data);
            
        } catch (error) {
            this.showError(error.message);
        } finally {
            this.hideLoading();
        }
    }

    displayResults(results) {
        this.updateGaugeChart(results.ubi_score);
        this.updateBiasLevel(results.bias_level, results.ubi_score);
        this.updateComponentChart(results.components, results.weights);
        this.updateDetailsTable(results);
        this.showResults();
    }

    updateGaugeChart(ubiScore) {
        const gaugeData = [{
            type: "indicator",
            mode: "gauge+number",
            value: ubiScore,
            number: { font: { size: 24 } },
            title: { text: "UBI Score", font: { size: 16 } },
            gauge: {
                axis: { range: [0, 1], tickwidth: 1, tickcolor: "darkblue" },
                bar: { color: "darkblue" },
                bgcolor: "white",
                borderwidth: 2,
                bordercolor: "gray",
                steps: [
                    { range: [0, 0.2], color: "lightgreen" },
                    { range: [0.2, 0.4], color: "yellow" },
                    { range: [0.4, 0.7], color: "orange" },
                    { range: [0.7, 1], color: "red" }
                ],
                threshold: {
                    line: { color: "red", width: 4 },
                    thickness: 0.75,
                    value: 0.9
                }
            }
        }];

        const gaugeLayout = {
            margin: { t: 0, b: 0 },
            font: { color: "darkblue", family: "Arial" },
            height: 200
        };

        Plotly.newPlot('ubiGauge', gaugeData, gaugeLayout, { responsive: true });
    }

    updateBiasLevel(biasLevel, ubiScore) {
        const biasLevelElement = document.getElementById('biasLevel');
        biasLevelElement.textContent = biasLevel;
        biasLevelElement.className = 'bias-level';
        
        if (ubiScore >= 0.7) {
            biasLevelElement.classList.add('high');
        } else if (ubiScore >= 0.4) {
            biasLevelElement.classList.add('medium');
        } else if (ubiScore >= 0.2) {
            biasLevelElement.classList.add('low');
        } else {
            biasLevelElement.classList.add('minimal');
        }
    }

    updateComponentChart(components, weights) {
        const categories = Object.keys(components);
        const scores = Object.values(components);
        const weightValues = categories.map(cat => weights[cat.toLowerCase()]);

        const trace1 = {
            x: categories,
            y: scores,
            type: 'bar',
            name: 'Scores',
            marker: { color: ['#ff9999', '#66b3ff', '#99ff99'] }
        };

        const trace2 = {
            x: categories,
            y: weightValues,
            type: 'bar',
            name: 'Weights',
            marker: { color: ['#ff6666', '#3388ff', '#66ff66'] },
            opacity: 0.6
        };

        const data = [trace1, trace2];

        const layout = {
            barmode: 'group',
            margin: { t: 30, l: 50, r: 30, b: 50 },
            height: 200,
            legend: { orientation: 'h', y: -0.2 },
            xaxis: { title: 'Components' },
            yaxis: { title: 'Value', range: [0, 1] }
        };

        Plotly.newPlot('componentChart', data, layout, { responsive: true });
    }

    updateDetailsTable(results) {
        const table = document.getElementById('detailsTable');
        
        let html = `
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Model</td>
                    <td>${results.model_name}</td>
                </tr>
                <tr>
                    <td>Bias Magnitude (BM)</td>
                    <td>${results.components.bias_magnitude.toFixed(4)}</td>
                </tr>
                <tr>
                    <td>Disparity (DP)</td>
                    <td>${results.components.disparity.toFixed(4)}</td>
                </tr>
                <tr>
                    <td>Distribution Shift (DS)</td>
                    <td>${results.components.distribution_shift.toFixed(4)}</td>
                </tr>
                <tr>
                    <td>Total Prompts</td>
                    <td>${results.metadata.total_prompts}</td>
                </tr>
                <tr>
                    <td>Processing Time</td>
                    <td>${results.metadata.processing_time.toFixed(2)}s</td>
                </tr>
                <tr>
                    <td>Analysis Time</td>
                    <td>${new Date(results.metadata.timestamp).toLocaleString()}</td>
                </tr>
            </table>
        `;
        
        table.innerHTML = html;
    }

    showLoading() {
        document.getElementById('loadingSection').classList.remove('hidden');
        document.getElementById('analyzeBtn').disabled = true;
    }

    hideLoading() {
        document.getElementById('loadingSection').classList.add('hidden');
        document.getElementById('analyzeBtn').disabled = false;
    }

    showResults() {
        document.getElementById('resultsSection').classList.remove('hidden');
    }

    hideResults() {
        document.getElementById('resultsSection').classList.add('hidden');
    }

    showError(message) {
        document.getElementById('errorMessage').textContent = message;
        document.getElementById('errorSection').classList.remove('hidden');
    }

    hideError() {
        document.getElementById('errorSection').classList.add('hidden');
    }
}

// Global function for error handling
function hideError() {
    document.getElementById('errorSection').classList.add('hidden');
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new BiasDetectionApp();
});