/**
 * Configuration module for Banking77 Intent Classifier Frontend.
 * Provides a single source of truth for backend API base URL resolution.
 */

(function () {
    const STORAGE_KEY = 'banking77_api_base_url';
    const DEFAULT_PORT = '8000';

    function detectDefaultUrl() {
        if (typeof window === 'undefined') return 'http://localhost:8000';
        
        // If loaded via file:// protocol or origin is 'null'
        if (!window.location.origin || window.location.origin === 'null' || window.location.protocol === 'file:') {
            return 'http://localhost:8000';
        }
        
        // If served from FastAPI directly (e.g., http://localhost:8000 or staging host)
        return window.location.origin;
    }

    const config = {
        STORAGE_KEY: STORAGE_KEY,
        
        getApiBaseUrl: function () {
            const saved = localStorage.getItem(STORAGE_KEY);
            if (saved && saved.trim()) {
                return saved.trim().replace(/\/+$/, '');
            }
            return detectDefaultUrl();
        },

        setApiBaseUrl: function (url) {
            if (!url || !url.trim()) {
                localStorage.removeItem(STORAGE_KEY);
            } else {
                localStorage.setItem(STORAGE_KEY, url.trim().replace(/\/+$/, ''));
            }
        },

        resetApiBaseUrl: function () {
            localStorage.removeItem(STORAGE_KEY);
            return detectDefaultUrl();
        }
    };

    // Attach globally
    window.BANKING77_CONFIG = config;
    // Backward compatibility alias
    window.API_BASE_URL = config.getApiBaseUrl();
})();
