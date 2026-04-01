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
                // Friendly display names for select options (value always = model id)
                const displayNames = {
                    'nvidia/nemotron-3-nano-30b-a3b:free': 'NVIDIA Nemotron 3 Nano (30B)',
                    'deepseek/deepseek-r1-0528:free': 'DeepSeek R1 (Free)'
                };
                data.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = displayNames[model] || model;
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
            
            // Fallback: show some default models when API fails
            const select = document.getElementById('modelSelect');
            select.innerHTML = `
                <option value="">Select a model...</option>
                <option value="gpt-4o">GPT-4o</option>
                <option value="gpt-4o-mini">GPT-4o Mini</option>
                <option value="gemini-pro">Gemini Pro</option>
                <option value="nvidia/nemotron-3-nano-30b-a3b:free">NVIDIA Nemotron 3 Nano (30B)</option>
                <option value="deepseek/deepseek-r1-0528:free">DeepSeek R1 (Free)</option>
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
        this.updateCategoryWiseSection(results.by_category || {});
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
        const meta = results.metadata || {};
        const comp = results.components || {};
        
        let html = `
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Model</td>
                    <td>${results.model_name || '—'}</td>
                </tr>
                <tr>
                    <td>Bias Magnitude (BM)</td>
                    <td>${comp.bias_magnitude != null ? comp.bias_magnitude.toFixed(4) : '—'}</td>
                </tr>
                <tr>
                    <td>Disparity (DP)</td>
                    <td>${comp.disparity != null ? comp.disparity.toFixed(4) : '—'}</td>
                </tr>
                <tr>
                    <td>Distribution Shift (DS)</td>
                    <td>${comp.distribution_shift != null ? comp.distribution_shift.toFixed(4) : '—'}</td>
                </tr>
                <tr>
                    <td>Total Prompts</td>
                    <td>${meta.total_prompts ?? '—'}</td>
                </tr>
                <tr>
                    <td>Processing Time</td>
                    <td>${meta.processing_time != null ? Number(meta.processing_time).toFixed(2) + 's' : '—'}</td>
                </tr>
                <tr>
                    <td>Analysis Time</td>
                    <td>${meta.timestamp ? new Date(meta.timestamp).toLocaleString() : '—'}</td>
                </tr>
            </table>
        `;
        
        table.innerHTML = html;
    }

    /**
     * Render category-wise bias analysis. Handles missing/insufficient data safely.
     */
    updateCategoryWiseSection(byCategory) {
        const section = document.getElementById('categoryWiseSection');
        const container = document.getElementById('categoryWiseContainer');
        if (!section || !container) return;

        const categories = Object.keys(byCategory || {});
        if (categories.length === 0) {
            section.classList.add('hidden');
            return;
        }

        const categoryLabels = {
            gender: 'Gender',
            race: 'Race',
            profession: 'Profession',
            religious_ideology: 'Religious Ideology',
            political_ideology: 'Political Ideology'
        };

        const getBiasBadgeClass = (biasLevel) => {
            if (!biasLevel || biasLevel === 'Insufficient Data') return 'bg-slate-600 text-slate-300';
            if (biasLevel === 'High Bias') return 'bg-red-500/20 text-red-400 border border-red-500/40';
            if (biasLevel === 'Medium Bias') return 'bg-amber-500/20 text-amber-400 border border-amber-500/40';
            if (biasLevel === 'Low Bias') return 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/40';
            return 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'; // Minimal Bias
        };

        let html = '';
        categories.forEach((cat) => {
            const data = byCategory[cat];
            const label = categoryLabels[cat] || cat.replace(/_/g, ' ');
            const ubi = data.ubi_score;
            const level = data.bias_level || 'Insufficient Data';
            const comp = data.components || {};
            const hasData = ubi != null && level !== 'Insufficient Data';

            const bm = comp.BM != null ? comp.BM.toFixed(3) : '—';
            const dp = comp.DP != null ? comp.DP.toFixed(3) : '—';
            const ds = comp.DS != null ? comp.DS.toFixed(3) : '—';

            html += `
                <div class="glass rounded-xl p-4 border border-white/5 hover:border-white/10 transition-colors">
                    <div class="flex items-center justify-between mb-3">
                        <h3 class="font-semibold text-slate-200">${label}</h3>
                        <span class="px-2 py-0.5 rounded-full text-xs font-medium ${getBiasBadgeClass(level)}">${level}</span>
                    </div>
                    <div class="text-2xl font-bold text-primary mb-3">${hasData ? ubi.toFixed(3) : '—'}</div>
                    <div class="text-xs text-slate-500 uppercase tracking-wider mb-1">Components</div>
                    <div class="grid grid-cols-3 gap-2 text-sm">
                        <div class="bg-slate-800/50 rounded px-2 py-1.5">
                            <span class="text-slate-500">BM</span>
                            <span class="float-right text-slate-200">${bm}</span>
                        </div>
                        <div class="bg-slate-800/50 rounded px-2 py-1.5">
                            <span class="text-slate-500">DP</span>
                            <span class="float-right text-slate-200">${dp}</span>
                        </div>
                        <div class="bg-slate-800/50 rounded px-2 py-1.5">
                            <span class="text-slate-500">DS</span>
                            <span class="float-right text-slate-200">${ds}</span>
                        </div>
                    </div>
                </div>
            `;
        });

        container.innerHTML = html;
        section.classList.remove('hidden');
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