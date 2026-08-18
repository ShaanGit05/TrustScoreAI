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
    }

    async analyzeModel() {
        const modelSelect = document.getElementById('modelSelect');
        let modelName = modelSelect.value;
        
        if (!modelName) {
            this.showError('Please select a model name');
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
        this.updateOverallResultsHero(results);
        this.updateComponentChart(results.components, results.weights);
        this.updateDetailsTable(results);
        this.updateCategoryWiseSection(results.by_category || {});
        this.showResults();
    }

    /**
     * Trust-score style hero (inverted UBI 0–1 → display 0–100); satellites from categories / components.
     */
    updateOverallResultsHero(results) {
        const ubi = results.ubi_score != null ? Number(results.ubi_score) : NaN;
        const trust = Number.isFinite(ubi) ? Math.max(0, Math.min(100, (1 - Math.min(ubi, 1)) * 100)) : null;

        const scoreEl = document.getElementById('resultTrustScore');
        const ubiRawEl = document.getElementById('resultUbiRaw');
        const modelSumEl = document.getElementById('resultModelSummary');
        if (scoreEl) {
            scoreEl.textContent = trust != null ? trust.toFixed(1) : '—';
        }
        if (ubiRawEl) {
            ubiRawEl.textContent = Number.isFinite(ubi) ? `UBI (raw): ${ubi.toFixed(4)}` : '';
        }
        if (modelSumEl) {
            const name = results.model_name || '—';
            modelSumEl.textContent = `Analysis complete for ${name}. Unified bias index and category signals are summarized below.`;
        }

        const biasWrap = document.getElementById('resultBiasLevel');
        if (biasWrap) {
            biasWrap.textContent = results.bias_level || '—';
            biasWrap.className =
                'bias-level inline-flex items-center rounded-full px-4 py-2 text-sm font-semibold mb-8';
            const s = Number.isFinite(ubi) ? ubi : 0;
            if (s >= 0.7) biasWrap.classList.add('high');
            else if (s >= 0.4) biasWrap.classList.add('medium');
            else if (s >= 0.2) biasWrap.classList.add('low');
            else biasWrap.classList.add('minimal');
        }

        const setSatellite = (id, text) => {
            const el = document.getElementById(id);
            if (!el) return;
            const span = el.querySelector('span');
            if (span) span.textContent = text;
        };

        const neutLabel = (catKey, shortLabel) => {
            const bc = results.by_category && results.by_category[catKey];
            if (!bc || bc.ubi_score == null || bc.bias_level === 'Insufficient Data') {
                return `${shortLabel}: —`;
            }
            const u = Math.min(Number(bc.ubi_score), 1);
            const pct = Math.round(Math.max(0, Math.min(100, (1 - u) * 100)));
            return `${shortLabel}: ${pct}%`;
        };

        const comp = results.components || {};
        const compNeut = (val, name) => {
            if (val == null || !Number.isFinite(Number(val))) return `${name}: —`;
            const c = Math.min(Math.max(Number(val), 0), 1);
            const pct = Math.round((1 - c) * 100);
            return `${name}: ${pct}%`;
        };

        const catLabels = {
            gender: 'Gender Neutrality',
            race: 'Racial Fairness',
            profession: 'Professional Fairness',
            religious_ideology: 'Ideological Balance',
            political_ideology: 'Political Balance',
        };

        let top = neutLabel('gender', 'Gender Neutrality');
        if (top.includes('—') && results.by_category) {
            const keys = Object.keys(results.by_category).filter(
                (k) => results.by_category[k] && results.by_category[k].ubi_score != null
            );
            if (keys.length) {
                const k = keys[0];
                top = neutLabel(k, catLabels[k] || k.replace(/_/g, ' '));
            }
        }
        if (top.includes('—')) {
            top = compNeut(comp.bias_magnitude, 'BM balance');
        }

        let br = neutLabel('race', 'Racial Fairness');
        if (br.includes('—') && results.by_category) {
            const keys = Object.keys(results.by_category).filter((k) => k !== 'gender');
            const cand = keys.find((k) => results.by_category[k] && results.by_category[k].ubi_score != null);
            if (cand) br = neutLabel(cand, catLabels[cand] || cand.replace(/_/g, ' '));
        }
        if (br.includes('—')) br = compNeut(comp.disparity, 'DP balance');

        let bl = neutLabel('profession', 'Professional Fairness');
        if (bl.includes('—') && results.by_category) {
            const keys = Object.keys(results.by_category).filter((k) => !['gender', 'race'].includes(k));
            const cand = keys.find((k) => results.by_category[k] && results.by_category[k].ubi_score != null);
            if (cand) bl = neutLabel(cand, catLabels[cand] || cand.replace(/_/g, ' '));
        }
        if (bl.includes('—')) bl = compNeut(comp.distribution_shift, 'DS balance');

        setSatellite('resultSatelliteTop', top);
        setSatellite('resultSatelliteBR', br);
        setSatellite('resultSatelliteBL', bl);
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
        // Plotly component scores (BM/DP/DS) only; keeps layout clean + palette consistent.
        const categories = Object.keys(components || {});
        const scores = Object.values(components || {});

        const trace = {
            x: categories,
            y: scores,
            type: 'bar',
            marker: { color: ['#f87171', '#22d3ee', '#4ade80'] },
            text: scores.map((v) => (v != null && Number.isFinite(Number(v)) ? Number(v).toFixed(3) : '—')),
            textposition: 'outside',
            cliponaxis: false,
        };

        const layout = {
            margin: { t: 10, l: 50, r: 20, b: 50 },
            autosize: true,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#94a3b8' },
            showlegend: false,
            xaxis: {
                title: { text: 'Components', font: { color: '#cbd' } },
                tickfont: { color: '#94a3b8' },
                gridcolor: 'rgba(255,255,255,0.06)',
            },
            yaxis: {
                title: { text: 'Value', font: { color: '#cbd' } },
                tickfont: { color: '#94a3b8' },
                range: [0, 1],
                gridcolor: 'rgba(255,255,255,0.06)',
            },
        };

        Plotly.newPlot('componentChart', [trace], layout, { responsive: true, displayModeBar: false });
    }

    updateDetailsTable(results) {
        const table = document.getElementById('detailsTable');
        const meta = results.metadata || {};
        const comp = results.components || {};
        
        let html = `
            <table class="w-full text-left border-collapse">
                <thead>
                    <tr>
                        <th class="py-3 px-4 text-xs sm:text-sm font-semibold text-slate-200 border-b border-white/10">Metric</th>
                        <th class="py-3 px-4 text-xs sm:text-sm font-semibold text-slate-200 border-b border-white/10">Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="py-3 px-4 text-sm text-slate-300 border-b border-white/5">Model</td>
                        <td class="py-3 px-4 text-sm text-slate-100 border-b border-white/5">${results.model_name || '—'}</td>
                    </tr>
                    <tr>
                        <td class="py-3 px-4 text-sm text-slate-300 border-b border-white/5">Bias Magnitude (BM)</td>
                        <td class="py-3 px-4 text-sm text-slate-100 border-b border-white/5">${comp.bias_magnitude != null ? comp.bias_magnitude.toFixed(4) : '—'}</td>
                    </tr>
                    <tr>
                        <td class="py-3 px-4 text-sm text-slate-300 border-b border-white/5">Disparity (DP)</td>
                        <td class="py-3 px-4 text-sm text-slate-100 border-b border-white/5">${comp.disparity != null ? comp.disparity.toFixed(4) : '—'}</td>
                    </tr>
                    <tr>
                        <td class="py-3 px-4 text-sm text-slate-300 border-b border-white/5">Distribution Shift (DS)</td>
                        <td class="py-3 px-4 text-sm text-slate-100 border-b border-white/5">${comp.distribution_shift != null ? comp.distribution_shift.toFixed(4) : '—'}</td>
                    </tr>
                    <tr>
                        <td class="py-3 px-4 text-sm text-slate-300 border-b border-white/5">Total Prompts</td>
                        <td class="py-3 px-4 text-sm text-slate-100 border-b border-white/5">${meta.total_prompts ?? '—'}</td>
                    </tr>
                    <tr>
                        <td class="py-3 px-4 text-sm text-slate-300 border-b border-white/5">Processing Time</td>
                        <td class="py-3 px-4 text-sm text-slate-100 border-b border-white/5">${meta.processing_time != null ? Number(meta.processing_time).toFixed(2) + 's' : '—'}</td>
                    </tr>
                    <tr>
                        <td class="py-3 px-4 text-sm text-slate-300">Analysis Time</td>
                        <td class="py-3 px-4 text-sm text-slate-100">${meta.timestamp ? new Date(meta.timestamp).toLocaleString() : '—'}</td>
                    </tr>
                </tbody>
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