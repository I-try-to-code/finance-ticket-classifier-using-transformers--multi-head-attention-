/**
 * Banking77 Intent Classification - Frontend Application Logic
 * Handles user input, client-side validation, API communication,
 * asynchronous state transitions, and dynamic visualization.
 */

(function () {
    'use strict';

    // DOM Elements
    const elements = {
        form: document.getElementById('prediction-form'),
        queryInput: document.getElementById('query-input'),
        charCounter: document.getElementById('char-counter'),
        btnClear: document.getElementById('btn-clear'),
        btnSubmit: document.getElementById('btn-submit'),
        healthBadge: document.getElementById('health-badge'),
        healthStatusText: document.getElementById('health-status-text'),
        deviceMeta: document.getElementById('device-meta'),
        apiDocsLink: document.getElementById('api-docs-link'),
        systemAlert: document.getElementById('system-alert'),
        alertMessage: document.getElementById('alert-message'),
        emptyState: document.getElementById('empty-state'),
        predictionContainer: document.getElementById('prediction-container'),
        predictedIntentName: document.getElementById('predicted-intent-name'),
        predictedConfidence: document.getElementById('predicted-confidence'),
        candidateList: document.getElementById('candidate-list'),
        latencyBadge: document.getElementById('latency-badge'),
        latencyValue: document.getElementById('latency-value'),
        queryEchoText: document.getElementById('query-echo-text'),
        exampleChips: document.querySelectorAll('.chip-btn'),
        // Settings Modal
        btnSettings: document.getElementById('btn-settings'),
        settingsModal: document.getElementById('settings-modal'),
        btnCloseModal: document.getElementById('btn-close-modal'),
        inputApiUrl: document.getElementById('input-api-url'),
        btnSaveApi: document.getElementById('btn-save-api'),
        btnResetApi: document.getElementById('btn-reset-api'),
        // Theme Toggle
        btnThemeToggle: document.getElementById('btn-theme-toggle'),
        themeIconSun: document.getElementById('theme-icon-sun'),
        themeIconMoon: document.getElementById('theme-icon-moon'),
    };

    // Theme Management (Default: Light Mode)
    function getStoredTheme() {
        return localStorage.getItem('banking77_theme') || 'light';
    }

    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        if (elements.themeIconSun && elements.themeIconMoon) {
            if (theme === 'dark') {
                elements.themeIconSun.style.display = 'inline-block';
                elements.themeIconMoon.style.display = 'none';
            } else {
                elements.themeIconSun.style.display = 'none';
                elements.themeIconMoon.style.display = 'inline-block';
            }
        }
    }

    function toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        localStorage.setItem('banking77_theme', newTheme);
        applyTheme(newTheme);
    }

    // Helper: Get active API Base URL
    function getBaseUrl() {
        if (window.BANKING77_CONFIG && typeof window.BANKING77_CONFIG.getApiBaseUrl === 'function') {
            return window.BANKING77_CONFIG.getApiBaseUrl();
        }
        return window.API_BASE_URL || 'http://localhost:8000';
    }

    // Show / Hide Alert
    function showAlert(message, type = 'error') {
        if (!elements.systemAlert || !elements.alertMessage) return;
        elements.alertMessage.textContent = message;
        elements.systemAlert.className = `alert-box alert-${type}`;
        elements.systemAlert.style.display = 'block';
    }

    function hideAlert() {
        if (!elements.systemAlert) return;
        elements.systemAlert.style.display = 'none';
    }

    // Health Check
    async function checkBackendHealth() {
        const baseUrl = getBaseUrl();
        elements.healthBadge.className = 'health-badge connecting';
        elements.healthStatusText.textContent = 'Connecting...';

        if (elements.apiDocsLink) {
            elements.apiDocsLink.href = `${baseUrl}/docs`;
        }

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 4000);

            const res = await fetch(`${baseUrl}/health`, {
                signal: controller.signal,
                headers: { 'Accept': 'application/json' }
            });
            clearTimeout(timeoutId);

            if (res.ok) {
                const data = await res.json();
                if (data.status === 'healthy') {
                    elements.healthBadge.className = 'health-badge healthy';
                    elements.healthStatusText.textContent = `Online (${data.device.toUpperCase()})`;
                    if (elements.deviceMeta) {
                        elements.deviceMeta.textContent = `Device: ${data.device.toUpperCase()}`;
                    }
                    return;
                }
            }

            elements.healthBadge.className = 'health-badge unhealthy';
            elements.healthStatusText.textContent = 'Model Offline';
        } catch (err) {
            elements.healthBadge.className = 'health-badge unhealthy';
            elements.healthStatusText.textContent = 'Backend Offline';
        }
    }

    // Update Character Counter
    function updateCharCounter() {
        const length = elements.queryInput.value.length;
        elements.charCounter.textContent = `${length} / 1000 characters`;

        elements.charCounter.classList.remove('limit-near', 'limit-reached');
        if (length >= 1000) {
            elements.charCounter.classList.add('limit-reached');
        } else if (length >= 900) {
            elements.charCounter.classList.add('limit-near');
        }
    }

    // Clear Input
    function clearInput() {
        elements.queryInput.value = '';
        updateCharCounter();
        elements.queryInput.focus();
        hideAlert();
    }

    // Set Loading State
    function setLoading(isLoading) {
        if (isLoading) {
            elements.btnSubmit.disabled = true;
            elements.btnSubmit.classList.add('loading');
            elements.btnSubmit.querySelector('.btn-text').textContent = 'Predicting...';
        } else {
            elements.btnSubmit.disabled = false;
            elements.btnSubmit.classList.remove('loading');
            elements.btnSubmit.querySelector('.btn-text').textContent = 'Classify Intent';
        }
    }

    // Render Prediction Results
    function renderPredictions(data) {
        hideAlert();
        
        // Hide empty state, show container
        elements.emptyState.style.display = 'none';
        elements.predictionContainer.style.display = 'flex';

        // 1. Primary Prediction
        elements.predictedIntentName.textContent = data.predicted_intent;
        const confidencePct = (data.confidence * 100).toFixed(1);
        elements.predictedConfidence.textContent = `${confidencePct}%`;

        // 2. Candidate Breakdown
        elements.candidateList.innerHTML = '';
        if (Array.isArray(data.top_predictions)) {
            data.top_predictions.forEach((item, index) => {
                const probPct = (item.probability * 100).toFixed(1);
                const candidateEl = document.createElement('div');
                candidateEl.className = 'candidate-item';
                candidateEl.innerHTML = `
                    <div class="candidate-info">
                        <span class="candidate-name" title="${item.intent}">#${index + 1} ${item.intent}</span>
                        <span class="candidate-pct">${probPct}%</span>
                    </div>
                    <div class="progress-track">
                        <div class="progress-fill" style="width: 0%;"></div>
                    </div>
                `;
                elements.candidateList.appendChild(candidateEl);

                // Trigger animation after DOM insertion
                requestAnimationFrame(() => {
                    const fill = candidateEl.querySelector('.progress-fill');
                    if (fill) fill.style.width = `${Math.max(1, probPct)}%`;
                });
            });
        }

        // 3. Latency Badge
        if (data.latency_ms !== undefined) {
            elements.latencyValue.textContent = `${data.latency_ms.toFixed(2)} ms`;
            elements.latencyBadge.style.display = 'inline-flex';
        }

        // 4. Query Echo
        elements.queryEchoText.textContent = `"${data.text}"`;
    }

    // Handle Prediction Submission
    async function submitPrediction() {
        const rawText = elements.queryInput.value;
        const trimmedText = rawText.trim();

        // Client-side validation
        if (!trimmedText) {
            showAlert('Please enter a customer banking query before submitting.', 'warning');
            elements.queryInput.focus();
            return;
        }

        if (trimmedText.length > 1000) {
            showAlert('Query exceeds maximum allowed limit of 1000 characters.', 'warning');
            return;
        }

        const baseUrl = getBaseUrl();
        setLoading(true);
        hideAlert();

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 10000); // 10s timeout

        try {
            const response = await fetch(`${baseUrl}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                },
                body: JSON.stringify({ text: trimmedText }),
                signal: controller.signal,
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                let errorMessage = `Server error (HTTP ${response.status})`;
                try {
                    const errData = await response.json();
                    if (errData && errData.detail) {
                        if (Array.isArray(errData.detail)) {
                            errorMessage = errData.detail.map(d => d.msg || JSON.stringify(d)).join(', ');
                        } else {
                            errorMessage = errData.detail;
                        }
                    }
                } catch (_) {}
                throw new Error(errorMessage);
            }

            const data = await response.json();
            renderPredictions(data);
        } catch (err) {
            clearTimeout(timeoutId);
            if (err.name === 'AbortError') {
                showAlert('Request timed out after 10 seconds. Is the backend server busy or warming up?', 'error');
            } else {
                const isConnError = err.message.includes('Failed to fetch') || err.message.includes('NetworkError');
                if (isConnError) {
                    showAlert(`Failed to reach backend at ${baseUrl}. Ensure FastAPI is running via 'python -m src.api.main'.`, 'error');
                } else {
                    showAlert(err.message || 'An error occurred during intent prediction.', 'error');
                }
            }
        } finally {
            setLoading(false);
        }
    }

    // Modal Control Functions
    function openSettingsModal() {
        elements.inputApiUrl.value = getBaseUrl();
        elements.settingsModal.classList.add('active');
        elements.inputApiUrl.focus();
    }

    function closeSettingsModal() {
        elements.settingsModal.classList.remove('active');
    }

    function saveSettings() {
        const url = elements.inputApiUrl.value.trim();
        if (window.BANKING77_CONFIG) {
            window.BANKING77_CONFIG.setApiBaseUrl(url);
            window.API_BASE_URL = window.BANKING77_CONFIG.getApiBaseUrl();
        }
        closeSettingsModal();
        checkBackendHealth();
    }

    function resetSettings() {
        if (window.BANKING77_CONFIG) {
            const defaultUrl = window.BANKING77_CONFIG.resetApiBaseUrl();
            elements.inputApiUrl.value = defaultUrl;
            window.API_BASE_URL = defaultUrl;
        }
        closeSettingsModal();
        checkBackendHealth();
    }

    // Attach Event Listeners
    function initEvents() {
        // Character counter
        elements.queryInput.addEventListener('input', updateCharCounter);

        // Clear button
        elements.btnClear.addEventListener('click', clearInput);

        // Form submit
        elements.form.addEventListener('submit', (e) => {
            e.preventDefault();
            submitPrediction();
        });

        // Ctrl + Enter shortcut
        elements.queryInput.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                submitPrediction();
            }
        });

        // Example Query Chips
        elements.exampleChips.forEach((chip) => {
            chip.addEventListener('click', () => {
                const query = chip.getAttribute('data-query');
                if (query) {
                    elements.queryInput.value = query;
                    updateCharCounter();
                    submitPrediction();
                }
            });
        });

        // Theme Toggle Event
        if (elements.btnThemeToggle) {
            elements.btnThemeToggle.addEventListener('click', toggleTheme);
        }

        // Settings Modal Events
        elements.btnSettings.addEventListener('click', openSettingsModal);
        elements.btnCloseModal.addEventListener('click', closeSettingsModal);
        elements.btnSaveApi.addEventListener('click', saveSettings);
        elements.btnResetApi.addEventListener('click', resetSettings);
        elements.settingsModal.addEventListener('click', (e) => {
            if (e.target === elements.settingsModal) closeSettingsModal();
        });

        // Close modal on Escape
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && elements.settingsModal.classList.contains('active')) {
                closeSettingsModal();
            }
        });
    }

    // Initialize Application
    function init() {
        applyTheme(getStoredTheme());
        initEvents();
        updateCharCounter();
        checkBackendHealth();

        // Periodic health check every 30 seconds
        setInterval(checkBackendHealth, 30000);
    }

    // Run on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
