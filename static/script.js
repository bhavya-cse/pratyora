/**
 * PRATYORA — Frontend Application Controller
 * Handles tabs, multi-modal uploads, API communication,
 * progress tracking, DOM updates, theme persistence,
 * and accessible interactions.
 */

(function () {
  'use strict';

  // Flask backend URL.
  // The frontend may run on port 5500 while Flask runs on port 5001.
  // If this page is served by Flask itself (port 5001), use same-origin
  // relative URLs (no CORS involved). Otherwise (e.g. VS Code Live Server
  // on :5500) call Flask explicitly. Using the same hostname the page was
  // loaded from avoids localhost vs 127.0.0.1 origin mismatches.
  const API_BASE_URL = window.location.protocol === 'https:'
    ? ''
    : window.location.port === '5001'
      ? ''
      : `http://${window.location.hostname || '127.0.0.1'}:5001`;

  // State Management
  const state = {
    activeTab: 'text',
    selectedImageFile: null,
    isProcessing: false,
  };

  // Initialize on DOM Ready
  document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initNavigation();
    initTextForm();
    initImageUpload();
    initLinkForm();
    initHeaderScroll();
    initProcessReveal();
    initReveal();
    initCounters();
    initActiveNav();
    checkSystemHealth();
  });

  /**
   * Browsers word network-level fetch failures differently:
   * Chrome/Edge: "Failed to fetch", Safari: "Load failed", Firefox: "NetworkError...".
   */
  function isNetworkError(error) {
    const m = (error && error.message || '').toLowerCase();
    return error instanceof TypeError ||
      m.includes('failed to fetch') || m.includes('load failed') || m.includes('networkerror');
  }

  const BACKEND_DOWN_MSG =
    `Could not reach the Pratyora backend at ${API_BASE_URL || window.location.origin}. ` +
    'Make sure Flask is running (python app.py in the project folder, after pip install -r requirements.txt), ' +
    'then open http://127.0.0.1:5001/api/health to confirm it responds.';

  /* ==========================================================================
     API Helpers
     ========================================================================== */

  /**
   * Parse a server response safely.
   * This prevents HTML error pages from being mistaken for network failures.
   */
  async function readJsonResponse(response) {
    const contentType = response.headers.get('content-type') || '';

    if (!contentType.includes('application/json')) {
      const responseText = await response.text();
      throw new Error(
        `The server returned a non-JSON response (HTTP ${response.status}). ` +
        (responseText.slice(0, 150) || 'Check that the Flask API is running.')
      );
    }

    return response.json();
  }

  /**
   * Send JSON requests to the Flask backend.
   */
  async function apiJsonRequest(path, options = {}) {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers: {
        ...(options.headers || {}),
      },
    });

    const data = await readJsonResponse(response);

    if (!response.ok) {
      throw new Error(
        data.error || `Server returned HTTP ${response.status}.`
      );
    }

    return data;
  }

  /* ==========================================================================
     Visual Polish Helpers
     ========================================================================== */

  function initHeaderScroll() {
    const header = document.getElementById('siteHeader');
    if (!header) return;

    const onScroll = () => {
      header.classList.toggle('is-scrolled', window.scrollY > 8);
    };

    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
  }

  function initProcessReveal() {
    const grid = document.getElementById('processGrid');
    if (!grid) return;

    if (!('IntersectionObserver' in window)) {
      grid.classList.add('is-visible');
      return;
    }

    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          grid.classList.add('is-visible');
          io.disconnect();
        }
      });
    }, { threshold: 0.35 });

    io.observe(grid);
  }

  function animateNumber(el, target, duration) {
    if (!el) return;

    const reduce = window.matchMedia(
      '(prefers-reduced-motion: reduce)'
    ).matches;

    if (reduce) {
      el.textContent = `${target.toFixed(2)}%`;
      return;
    }

    const start = performance.now();

    function frame(now) {
      const t = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - t, 3);

      el.textContent = `${(target * eased).toFixed(2)}%`;

      if (t < 1) {
        requestAnimationFrame(frame);
      }
    }

    requestAnimationFrame(frame);
  }

  function showToast(message) {
    const toast = document.getElementById('toast');
    if (!toast) return;

    toast.textContent = message;
    toast.classList.add('is-visible');

    clearTimeout(showToast._t);

    showToast._t = setTimeout(() => {
      toast.classList.remove('is-visible');
    }, 2200);
  }


  /* ==========================================================================
     Scroll reveal, stat counters, active nav link
     ========================================================================== */

  function initReveal() {
    document.documentElement.classList.add('js');
    const items = document.querySelectorAll('.reveal');
    if (!('IntersectionObserver' in window)) {
      items.forEach((el) => el.classList.add('in'));
      return;
    }
    // stagger siblings inside the same grid
    document.querySelectorAll('.features-grid, .process-grid, .limits-grid, .stats-grid')
      .forEach((grid) => {
        grid.querySelectorAll(':scope > .reveal').forEach((el, i) => {
          el.style.transitionDelay = `${i * 90}ms`;
        });
      });
    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('in');
          io.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });
    items.forEach((el) => io.observe(el));
  }

  function initCounters() {
    const counters = document.querySelectorAll('[data-count]');
    if (!counters.length) return;
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const run = (el) => {
      const target = parseInt(el.dataset.count, 10) || 0;
      if (reduce || target === 0) { el.textContent = target.toLocaleString(); return; }
      const start = performance.now();
      const duration = 1400;
      (function frame(now) {
        const t = Math.min((now - start) / duration, 1);
        el.textContent = Math.round(target * (1 - Math.pow(1 - t, 3))).toLocaleString();
        if (t < 1) requestAnimationFrame(frame);
      })(start);
    };
    if (!('IntersectionObserver' in window)) { counters.forEach(run); return; }
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) { run(e.target); io.unobserve(e.target); } });
    }, { threshold: 0.6 });
    counters.forEach((el) => { if (!reduce) el.textContent = '0'; io.observe(el); });
  }

  function initActiveNav() {
    const links = document.querySelectorAll('.nav-link');
    if (!links.length || !('IntersectionObserver' in window)) return;
    const map = {};
    links.forEach((a) => { map[a.getAttribute('href').slice(1)] = a; });
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting && map[e.target.id]) {
          links.forEach((l) => l.classList.remove('is-current'));
          map[e.target.id].classList.add('is-current');
        }
      });
    }, { rootMargin: '-45% 0px -50% 0px' });
    Object.keys(map).forEach((id) => { const el = document.getElementById(id); if (el) io.observe(el); });
  }

  /* ==========================================================================
     Theme Management
     ========================================================================== */

  function initTheme() {
    const savedTheme = localStorage.getItem('pratyora_theme') || 'light';

    document.documentElement.setAttribute('data-theme', savedTheme);

    const themeToggleBtn = document.getElementById('themeToggle');

    if (themeToggleBtn) {
      themeToggleBtn.addEventListener('click', () => {
        const currentTheme =
          document.documentElement.getAttribute('data-theme') || 'light';

        const newTheme = currentTheme === 'light' ? 'dark' : 'light';

        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('pratyora_theme', newTheme);
      });
    }
  }

  /* ==========================================================================
     Mobile Navigation
     ========================================================================== */

  function initNavigation() {
    const mobileBtn = document.getElementById('mobileMenuBtn');
    const mainNav = document.getElementById('mainNav');

    if (mobileBtn && mainNav) {
      mobileBtn.addEventListener('click', () => {
        const isOpen = mainNav.classList.toggle('is-open');

        mobileBtn.setAttribute('aria-expanded', isOpen.toString());
      });

      mainNav.querySelectorAll('.nav-link').forEach((link) => {
        link.addEventListener('click', () => {
          mainNav.classList.remove('is-open');
          mobileBtn.setAttribute('aria-expanded', 'false');
        });
      });
    }
  }

  /* ==========================================================================
     System Health Check
     ========================================================================== */

  async function checkSystemHealth() {
    const modelDot = document.getElementById('modelStatusDot');
    const modelText = document.getElementById('modelStatusText');
    const ocrDot = document.getElementById('ocrStatusDot');
    const ocrText = document.getElementById('ocrStatusText');

    try {
      // FIX: Use a template literal with the actual Flask backend URL.
      const data = await apiJsonRequest('/api/health');

      if (data.model && data.model.available) {
        if (modelDot) modelDot.className = 'status-indicator online';
        if (modelText) modelText.textContent = 'ML Model Active';
      } else {
        if (modelDot) modelDot.className = 'status-indicator offline';
        if (modelText) modelText.textContent = 'ML Model Missing';
      }

      if (
        data.system_dependencies &&
        data.system_dependencies.tesseract_ocr &&
        data.system_dependencies.tesseract_ocr.available
      ) {
        if (ocrDot) ocrDot.className = 'status-indicator online';
        if (ocrText) ocrText.textContent = 'Tesseract OCR Ready';
      } else {
        if (ocrDot) ocrDot.className = 'status-indicator offline';
        if (ocrText) ocrText.textContent = 'Tesseract Missing';
      }
    } catch (error) {
      console.warn('Could not contact the Pratyora API:', error);

      if (modelDot) modelDot.className = 'status-indicator offline';
      if (modelText) modelText.textContent = 'Backend Offline';

      if (ocrDot) ocrDot.className = 'status-indicator offline';
      if (ocrText) ocrText.textContent = 'Status Unavailable';

    }
  }

  /* ==========================================================================
     Workspace Tab Switching
     ========================================================================== */

  window.switchTab = function (tabName) {
    if (state.isProcessing) return;

    const tabs = ['text', 'image', 'link'];

    if (!tabs.includes(tabName)) return;

    state.activeTab = tabName;
    hideAlert();

    tabs.forEach((tab) => {
      const cap = tab.charAt(0).toUpperCase() + tab.slice(1);
      const btn = document.getElementById(`tabBtn${cap}`);
      const panel = document.getElementById(`tabPanel${cap}`);

      if (!btn || !panel) return;

      const isActive = tab === tabName;

      btn.classList.toggle('active', isActive);
      btn.setAttribute('aria-selected', isActive.toString());

      panel.classList.toggle('active', isActive);
      panel.hidden = !isActive;
    });
  };

  /* ==========================================================================
     Tab 1: Text Analysis
     ========================================================================== */

  function initTextForm() {
    const textarea = document.getElementById('newsTextInput');
    const counter = document.getElementById('charCounter');

    if (textarea && counter) {
      textarea.addEventListener('input', () => {
        const len = textarea.value.length;

        counter.textContent =
          `${len.toLocaleString()} / 20,000 characters`;
      });
    }
  }

  window.clearTextInput = function () {
    const textarea = document.getElementById('newsTextInput');
    const counter = document.getElementById('charCounter');

    if (textarea) textarea.value = '';
    if (counter) counter.textContent = '0 / 20,000 characters';

    hideAlert();
  };

  window.handleTextSubmit = async function (e) {
    if (e) e.preventDefault();

    if (state.isProcessing) return;

    const textarea = document.getElementById('newsTextInput');
    const submitBtn = document.getElementById('analyzeTextBtn');

    if (!textarea) {
      showAlert('The news text input could not be found.');
      return;
    }

    const rawText = (textarea.value || '').trim();

    if (rawText.length < 15) {
      showAlert(
        'Please enter at least 15 characters of news text or claim to analyze.'
      );
      return;
    }

    if (rawText.length > 20000) {
      showAlert('Text exceeds maximum limit of 20,000 characters.');
      return;
    }

    setProcessingState(true, submitBtn);
    hideAlert();

    try {
      // FIX: Request goes to Flask on port 5001, not the Live Server port.
      const data = await apiJsonRequest('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: rawText }),
      });

      if (!data.success) {
        showAlert(data.error || 'Failed to analyze text.');
      } else {
        renderResults(data, 'Text Analysis');
      }
    } catch (error) {
      console.error('Text analysis request failed:', error);

      showAlert(
        isNetworkError(error)
          ? BACKEND_DOWN_MSG
          : `Text analysis failed: ${error.message}`
      );
    } finally {
      setProcessingState(false, submitBtn);
    }
  };

  /* ==========================================================================
     Tab 2: Image & Screenshot Analysis
     ========================================================================== */

  function initImageUpload() {
    const dropZone = document.getElementById('imageDropZone');
    const fileInput = document.getElementById('imageFileInput');

    if (!dropZone || !fileInput) return;

    ['dragenter', 'dragover'].forEach((eventName) => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.add('is-dragover');
      });
    });

    ['dragleave', 'drop'].forEach((eventName) => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.remove('is-dragover');
      });
    });

    dropZone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;

      if (dt && dt.files && dt.files.length > 0) {
        processSelectedImage(dt.files[0]);
      }
    });
  }

  window.handleImageSelect = function (e) {
    if (e.target.files && e.target.files.length > 0) {
      processSelectedImage(e.target.files[0]);
    }
  };

  function processSelectedImage(file) {
    hideAlert();

    const allowed = ['image/png', 'image/jpeg', 'image/webp'];

    if (!allowed.includes(file.type)) {
      showAlert(
        `Unsupported file format '${file.type || file.name}'. ` +
        'Please choose a PNG, JPG, or WEBP image.'
      );
      return;
    }

    const maxSize = 5 * 1024 * 1024;

    if (file.size > maxSize) {
      showAlert(
        `Image size (${(file.size / (1024 * 1024)).toFixed(2)} MB) ` +
        'exceeds the 5 MB limit.'
      );
      return;
    }

    state.selectedImageFile = file;

    const prompt = document.getElementById('imageUploadPrompt');
    const previewContainer =
      document.getElementById('imagePreviewContainer');
    const previewImg = document.getElementById('imagePreviewImg');
    const filenameEl = document.getElementById('imageFilename');
    const filesizeEl = document.getElementById('imageFilesize');
    const submitBtn = document.getElementById('analyzeImageBtn');

    if (filenameEl) filenameEl.textContent = file.name;

    if (filesizeEl) {
      filesizeEl.textContent =
        `${(file.size / (1024 * 1024)).toFixed(2)} MB • ${file.type}`;
    }

    const reader = new FileReader();

    reader.onload = (e) => {
      if (previewImg) previewImg.src = e.target.result;
      if (prompt) prompt.style.display = 'none';
      if (previewContainer) previewContainer.style.display = 'flex';
      if (submitBtn) submitBtn.disabled = false;
    };

    reader.onerror = () => {
      state.selectedImageFile = null;
      showAlert('Could not read the selected image. Please try another file.');
    };

    reader.readAsDataURL(file);
  }

  window.clearImageInput = function () {
    state.selectedImageFile = null;

    const fileInput = document.getElementById('imageFileInput');
    const prompt = document.getElementById('imageUploadPrompt');
    const previewContainer =
      document.getElementById('imagePreviewContainer');
    const previewImg = document.getElementById('imagePreviewImg');
    const submitBtn = document.getElementById('analyzeImageBtn');

    if (fileInput) fileInput.value = '';
    if (prompt) prompt.style.display = 'block';
    if (previewContainer) previewContainer.style.display = 'none';
    if (previewImg) previewImg.src = '';
    if (submitBtn) submitBtn.disabled = true;

    hideAlert();
  };

  window.handleImageSubmit = async function (e) {
    if (e) e.preventDefault();

    if (state.isProcessing || !state.selectedImageFile) return;

    const submitBtn = document.getElementById('analyzeImageBtn');

    setProcessingState(true, submitBtn);
    hideAlert();

    const formData = new FormData();
    formData.append('image', state.selectedImageFile);

    try {
      // FIX: images must go to /api/analyze-image (not /api/analyze).
      const response = await fetch(
        `${API_BASE_URL}/api/analyze-image`,
        {
          method: 'POST',
          body: formData,
        }
      );

      const data = await readJsonResponse(response);

      if (!response.ok || !data.success) {
        showAlert(data.error || 'Failed to process image OCR or analyze content.');
      } else {
        renderResults(data, 'Image OCR Analysis');
      }
    } catch (error) {
      console.error('Image analysis request failed:', error);

      showAlert(
        isNetworkError(error)
          ? BACKEND_DOWN_MSG
          : `Image analysis failed: ${error.message}`
      );
    } finally {
      setProcessingState(false, submitBtn);
    }
  };

  /* ==========================================================================
     Tab 3: News Link Analysis
     ========================================================================== */

  function initLinkForm() {
    const input = document.getElementById('newsLinkInput');

    if (!input) return;

    // When a user pastes a link, trim stray spaces/newlines around it right away.
    input.addEventListener('paste', () => {
      setTimeout(() => {
        input.value = input.value.trim();
      }, 0);
    });
  }

  window.clearLinkInput = function () {
    const input = document.getElementById('newsLinkInput');

    if (input) {
      input.value = '';
      input.focus();
    }

    hideAlert();
  };

  /**
   * Accepts "https://site.com/story", "site.com/story" or "www.site.com/story"
   * and returns a clean http(s) URL string, or null when it is not a usable link.
   */
  function normalizeNewsUrl(raw) {
    let value = String(raw || '').trim();

    if (!value || /\s/.test(value)) return null;

    if (!/^[a-z][a-z0-9+.-]*:\/\//i.test(value)) {
      value = `https://${value}`;
    }

    let parsed;

    try {
      parsed = new URL(value);
    } catch (error) {
      return null;
    }

    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      return null;
    }

    // Needs a real host such as news.example.com (not "localhost" or a bare word).
    if (!parsed.hostname || !parsed.hostname.includes('.')) return null;

    return parsed.href;
  }

  window.handleLinkSubmit = async function (e) {
    if (e) e.preventDefault();

    if (state.isProcessing) return;

    const input = document.getElementById('newsLinkInput');
    const submitBtn = document.getElementById('analyzeLinkBtn');

    if (!input) {
      showAlert('The link input could not be found.');
      return;
    }

    const url = normalizeNewsUrl(input.value);

    if (!url) {
      showAlert(
        'Please paste a full link to a news page, for example https://www.example.com/news/story.'
      );
      return;
    }

    input.value = url;

    setProcessingState(true, submitBtn);
    hideAlert();

    try {
      const data = await apiJsonRequest('/api/analyze-link', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url }),
      });

      if (!data.success) {
        showAlert(
          data.error ||
            'Could not read that link. Paste the article text into the Text tab instead.'
        );
      } else {
        renderResults(data, 'Link Analysis');
      }
    } catch (error) {
      console.error('Link analysis request failed:', error);

      showAlert(
        isNetworkError(error)
          ? BACKEND_DOWN_MSG
          : `Link analysis failed: ${error.message}`
      );
    } finally {
      setProcessingState(false, submitBtn);
    }
  };

  /* ==========================================================================
     Unified Results Rendering
     ========================================================================== */

  /* ==========================================================================
     Detected-language badge
     ========================================================================== */

  const LANGUAGE_NATIVE_NAMES = {
    en: 'English', hi: 'हिन्दी', bn: 'বাংলা', te: 'తెలుగు', kn: 'ಕನ್ನಡ',
    mr: 'मराठी', ta: 'தமிழ்', gu: 'ગુજરાતી', or: 'ଓଡ଼ିଆ', bho: 'भोजपुरी',
    pa: 'ਪੰਜਾਬੀ', ml: 'മലയാളം',
  };

  // Languages whose detection/translation is weaker (matches the hero "Beta" chips).
  const BETA_LANGUAGES = new Set(['or', 'bho']);

  const GLOBE_ICON =
    '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" ' +
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<circle cx="12" cy="12" r="10"/><path d="M2 12h20"/>' +
    '<path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>';

  // Results carry language info at the top level.
  function pickLanguageInfo(data) {
    return data && data.detected_language_name ? data : null;
  }

  function makeBadgePart(tag, className, text) {
    const el = document.createElement(tag);
    el.className = className;
    if (text != null) el.textContent = text;
    return el;
  }

  function renderLanguageBadge(data) {
    const badge = document.getElementById('detectedLanguageBadge');
    if (!badge) return;

    const info = pickLanguageInfo(data);

    badge.replaceChildren();

    if (!info) {
      badge.style.display = 'none';
      badge.removeAttribute('data-level');
      return;
    }

    const code = String(info.detected_language || '').toLowerCase();
    const name = info.detected_language_name;
    const native = LANGUAGE_NATIVE_NAMES[code];
    const confidence = Number.isFinite(info.language_confidence)
      ? Math.round(info.language_confidence * 100)
      : null;

    const icon = makeBadgePart('span', 'language-badge__icon');
    icon.innerHTML = GLOBE_ICON; // static, trusted markup only
    badge.appendChild(icon);

    badge.appendChild(makeBadgePart('span', 'language-badge__name', name));

    if (native && native !== name) {
      const nativeEl = makeBadgePart('span', 'language-badge__native', native);
      nativeEl.setAttribute('lang', code);
      badge.appendChild(nativeEl);
    }

    if (confidence !== null) {
      const level = confidence >= 90 ? 'high' : confidence >= 80 ? 'medium' : 'low';
      badge.dataset.level = level;

      const conf = makeBadgePart('span', 'language-badge__conf');
      const meter = makeBadgePart('span', 'language-badge__meter');
      const fill = document.createElement('i');
      fill.style.width = `${confidence}%`;
      meter.appendChild(fill);
      conf.appendChild(meter);
      conf.appendChild(document.createTextNode(`${confidence}%`));
      badge.appendChild(conf);
    } else {
      badge.removeAttribute('data-level');
    }

    if (info.translated) {
      const tag = makeBadgePart(
        'span',
        'language-badge__tag language-badge__tag--translated',
        'Translated'
      );
      tag.title = info.translation_note || 'Translated to English before analysis';
      badge.appendChild(tag);
    }

    if (BETA_LANGUAGES.has(code)) {
      const tag = makeBadgePart('span', 'language-badge__tag language-badge__tag--beta', 'Beta');
      tag.title = 'Detection and translation are weaker for this language';
      badge.appendChild(tag);
    }

    badge.setAttribute(
      'aria-label',
      `Detected language: ${name}` +
        (confidence !== null ? `, ${confidence}% confidence` : '') +
        (info.translated ? ', translated to English' : '') +
        (BETA_LANGUAGES.has(code) ? ', beta support' : '')
    );
    badge.title = confidence !== null
      ? `Detection confidence: ${confidence}%`
      : 'Detected language';
    badge.style.display = 'inline-flex';
  }
    /* ==========================================================================
     Published Fact-Check Results
     ========================================================================== */

  function renderFactChecks(data) {
    const dashboard = document.getElementById('resultsDashboard');
    if (!dashboard) return;

    // Remove previous results so old fact-checks are never left on screen.
    const oldSection = document.getElementById('publishedFactChecks');

    if (oldSection) oldSection.remove();

    // Image/link results may not yet include fact-check data.
    if (!data || !data.fact_check) return;

    const factCheck = data.fact_check;
    const section = document.createElement('section');

    section.id = 'publishedFactChecks';
    section.setAttribute('aria-labelledby', 'publishedFactChecksTitle');

    // Use the existing site's theme and typography where possible.
    Object.assign(section.style, {
      marginTop: '24px',
      padding: '22px',
      border: '1px solid var(--border-color, #dfe5e7)',
      borderRadius: '16px',
      background: 'var(--card-bg, transparent)',
      color: 'var(--text-primary, inherit)'
    });

    const heading = document.createElement('h3');
    heading.id = 'publishedFactChecksTitle';
    heading.textContent = 'Published fact-checks';

    Object.assign(heading.style, {
      margin: '0 0 8px',
      fontSize: '1.15rem'
    });

    section.appendChild(heading);

  // Show the evidence-adjusted estimate and preserve the original ML score.
  if (
    assessment &&
    Number.isFinite(Number(assessment.adjusted_real_percentage)) &&
    Number.isFinite(Number(assessment.adjusted_fake_percentage))
  ) {
    const scorePanel = document.createElement('div');

    Object.assign(scorePanel.style, {
      margin: '12px 0 18px',
      padding: '16px',
      borderRadius: '12px',
      background: 'var(--surface-secondary, rgba(100, 150, 150, 0.10))',
      border: '1px solid var(--border-color, #dfe5e7)'
    });

    const scoreTitle = document.createElement('h4');
    scoreTitle.textContent = 'Evidence-adjusted estimate';
    scoreTitle.style.margin = '0 0 12px';
    scorePanel.appendChild(scoreTitle);

    const realScore = document.createElement('p');
    realScore.textContent =
      `Estimated real: ${Number(assessment.adjusted_real_percentage).toFixed(2)}%`;

    Object.assign(realScore.style, {
      fontWeight: '700',
      fontSize: '1.1rem',
      margin: '0 0 6px'
    });

    scorePanel.appendChild(realScore);

    const fakeScore = document.createElement('p');
    fakeScore.textContent =
      `Estimated fake: ${Number(assessment.adjusted_fake_percentage).toFixed(2)}%`;
    fakeScore.style.margin = '0 0 10px';
    scorePanel.appendChild(fakeScore);

    const originalReal = assessment.model_real_percentage;

    if (Number.isFinite(Number(originalReal))) {
      const original = document.createElement('p');
      original.textContent =
        `Original ML real score: ${Number(originalReal).toFixed(2)}%`;
      original.style.margin = '0 0 8px';
      scorePanel.appendChild(original);
    }

    const limitation = document.createElement('p');
    limitation.textContent =
      'Experimental estimate. Relevant source keywords do not prove a claim is true.';
    Object.assign(limitation.style, {
      margin: '0',
      fontSize: '0.9rem',
      lineHeight: '1.5'
    });
    scorePanel.appendChild(limitation);

    section.appendChild(scorePanel);
  }


    const intro = document.createElement('p');
    intro.textContent =
      'Independent publisher ratings for matching claims. These are separate from Pratyora’s machine-learning estimate.';

    Object.assign(intro.style, {
      margin: '0 0 16px',
      lineHeight: '1.6',
      fontSize: '0.92rem',
      opacity: '0.8'
    });

    section.appendChild(intro);

    const results = Array.isArray(factCheck.results)
      ? factCheck.results
      : [];

    // Multiple API entries can point to the same article.
    const seenUrls = new Set();
    const uniqueResults = results.filter((item) => {
      const url = String(item.url || '').trim();

      if (url && seenUrls.has(url)) return false;
      if (url) seenUrls.add(url);

      return true;
    });

    if (factCheck.status === 'matched' && uniqueResults.length > 0) {
      uniqueResults.forEach((item) => {
        const card = document.createElement('article');

        Object.assign(card.style, {
          padding: '16px',
          marginTop: '12px',
          border: '1px solid var(--border-color, #dfe5e7)',
          borderRadius: '12px'
        });

        const publisher = document.createElement('p');
        publisher.textContent =
          `Publisher: ${item.publisher || 'Unknown publisher'}`;

        Object.assign(publisher.style, {
          margin: '0 0 8px',
          fontSize: '0.88rem',
          fontWeight: '600'
        });

        const rating = document.createElement('p');
        rating.textContent = `Publisher rating: ${item.rating || 'Not provided'}`;

        Object.assign(rating.style, {
          margin: '0 0 8px',
          fontWeight: '700'
        });

        const title = document.createElement('h4');
        title.textContent = item.title || 'Fact-check article';

        Object.assign(title.style, {
          margin: '0 0 8px',
          fontSize: '1rem',
          lineHeight: '1.5'
        });

        card.appendChild(publisher);
        card.appendChild(rating);
        card.appendChild(title);

        if (item.claim) {
          const claim = document.createElement('p');
          claim.textContent = `Reviewed claim: ${item.claim}`;

          Object.assign(claim.style, {
            margin: '0 0 10px',
            fontSize: '0.9rem',
            lineHeight: '1.6'
          });

          card.appendChild(claim);
        }

        if (item.review_date) {
          const date = document.createElement('p');
          const parsedDate = new Date(item.review_date);

          date.textContent = Number.isNaN(parsedDate.getTime())
            ? `Review date: ${item.review_date}`
            : `Review date: ${parsedDate.toLocaleDateString()}`;

          Object.assign(date.style, {
            margin: '0 0 10px',
            fontSize: '0.85rem',
            opacity: '0.75'
          });

          card.appendChild(date);
        }

        // Only create clickable links for valid HTTP(S) URLs.
        try {
          const parsedUrl = new URL(item.url);

          if (parsedUrl.protocol === 'https:' ||
              parsedUrl.protocol === 'http:') {
            const link = document.createElement('a');
            link.href = parsedUrl.href;
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            link.textContent = 'Read original fact-check ↗';

            Object.assign(link.style, {
              display: 'inline-block',
              fontWeight: '600',
              textDecoration: 'underline',
              overflowWrap: 'anywhere'
            });

            card.appendChild(link);
          }
        } catch (_) {
          // Omit malformed or missing URLs.
        }

        section.appendChild(card);
      });
    } else {
      const message = document.createElement('p');

      if (factCheck.status === 'no_match') {
        message.textContent =
          'No matching published fact-check was found. This does not mean the claim is true or false.';
      } else {
        message.textContent =
          factCheck.message ||
          'Published fact-check results are temporarily unavailable.';
      }

      Object.assign(message.style, {
        margin: '0',
        lineHeight: '1.6'
      });

      section.appendChild(message);
    }

    dashboard.appendChild(section);
  }
    /* ==========================================================================
     Published Fact-Check Results
     ========================================================================== */

  function renderFactChecks(data) {
    const dashboard = document.getElementById('resultsDashboard');
    if (!dashboard) return;

    // Remove previous results so old fact-checks are never left on screen.
    const oldSection = document.getElementById('publishedFactChecks');

    if (oldSection) oldSection.remove();

    // Image/link results may not yet include fact-check data.
    if (!data || !data.fact_check) return;

    const factCheck = data.fact_check;
    const section = document.createElement('section');

    section.id = 'publishedFactChecks';
    section.setAttribute('aria-labelledby', 'publishedFactChecksTitle');

    // Use the existing site's theme and typography where possible.
    Object.assign(section.style, {
      marginTop: '24px',
      padding: '22px',
      border: '1px solid var(--border-color, #dfe5e7)',
      borderRadius: '16px',
      background: 'var(--card-bg, transparent)',
      color: 'var(--text-primary, inherit)'
    });

    const heading = document.createElement('h3');
    heading.id = 'publishedFactChecksTitle';
    heading.textContent = 'Published fact-checks';

    Object.assign(heading.style, {
      margin: '0 0 8px',
      fontSize: '1.15rem'
    });

    section.appendChild(heading);

    const intro = document.createElement('p');
    intro.textContent =
      'Independent publisher ratings for matching claims. These are separate from Pratyora’s machine-learning estimate.';

    Object.assign(intro.style, {
      margin: '0 0 16px',
      lineHeight: '1.6',
      fontSize: '0.92rem',
      opacity: '0.8'
    });

    section.appendChild(intro);

    const results = Array.isArray(factCheck.results)
      ? factCheck.results
      : [];

    // Multiple API entries can point to the same article.
    const seenUrls = new Set();
    const uniqueResults = results.filter((item) => {
      const url = String(item.url || '').trim();

      if (url && seenUrls.has(url)) return false;
      if (url) seenUrls.add(url);

      return true;
    });

    if (factCheck.status === 'matched' && uniqueResults.length > 0) {
      uniqueResults.forEach((item) => {
        const card = document.createElement('article');

        Object.assign(card.style, {
          padding: '16px',
          marginTop: '12px',
          border: '1px solid var(--border-color, #dfe5e7)',
          borderRadius: '12px'
        });

        const publisher = document.createElement('p');
        publisher.textContent =
          `Publisher: ${item.publisher || 'Unknown publisher'}`;

        Object.assign(publisher.style, {
          margin: '0 0 8px',
          fontSize: '0.88rem',
          fontWeight: '600'
        });

        const rating = document.createElement('p');
        rating.textContent = `Publisher rating: ${item.rating || 'Not provided'}`;

        Object.assign(rating.style, {
          margin: '0 0 8px',
          fontWeight: '700'
        });

        const title = document.createElement('h4');
        title.textContent = item.title || 'Fact-check article';

        Object.assign(title.style, {
          margin: '0 0 8px',
          fontSize: '1rem',
          lineHeight: '1.5'
        });

        card.appendChild(publisher);
        card.appendChild(rating);
        card.appendChild(title);

        if (item.claim) {
          const claim = document.createElement('p');
          claim.textContent = `Reviewed claim: ${item.claim}`;

          Object.assign(claim.style, {
            margin: '0 0 10px',
            fontSize: '0.9rem',
            lineHeight: '1.6'
          });

          card.appendChild(claim);
        }

        if (item.review_date) {
          const date = document.createElement('p');
          const parsedDate = new Date(item.review_date);

          date.textContent = Number.isNaN(parsedDate.getTime())
            ? `Review date: ${item.review_date}`
            : `Review date: ${parsedDate.toLocaleDateString()}`;

          Object.assign(date.style, {
            margin: '0 0 10px',
            fontSize: '0.85rem',
            opacity: '0.75'
          });

          card.appendChild(date);
        }

        // Only create clickable links for valid HTTP(S) URLs.
        try {
          const parsedUrl = new URL(item.url);

          if (parsedUrl.protocol === 'https:' ||
              parsedUrl.protocol === 'http:') {
            const link = document.createElement('a');
            link.href = parsedUrl.href;
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            link.textContent = 'Read original fact-check ↗';

            Object.assign(link.style, {
              display: 'inline-block',
              fontWeight: '600',
              textDecoration: 'underline',
              overflowWrap: 'anywhere'
            });

            card.appendChild(link);
          }
        } catch (_) {
          // Omit malformed or missing URLs.
        }

        section.appendChild(card);
      });
    } else {
      const message = document.createElement('p');

      if (factCheck.status === 'no_match') {
        message.textContent =
          'No matching published fact-check was found. This does not mean the claim is true or false.';
      } else {
        message.textContent =
          factCheck.message ||
          'Published fact-check results are temporarily unavailable.';
      }

      Object.assign(message.style, {
        margin: '0',
        lineHeight: '1.6'
      });

      section.appendChild(message);
    }

    dashboard.appendChild(section);
  }
/* ==========================================================================
   Live Web Evidence
   ========================================================================== */

function renderWebSearch(data) {
  const dashboard = document.getElementById('resultsDashboard');
  if (!dashboard) return;

  const oldSection = document.getElementById('liveWebEvidence');
  if (oldSection) oldSection.remove();

  const search = data && data.web_search;
  const assessment = data && data.evidence_assessment;
  if (!search) return;

  const section = document.createElement('section');
  section.id = 'liveWebEvidence';
  section.setAttribute('aria-labelledby', 'liveWebEvidenceTitle');

  Object.assign(section.style, {
    marginTop: '24px',
    padding: '22px',
    border: '1px solid var(--border-color, #dfe5e7)',
    borderRadius: '16px',
    background: 'var(--card-bg, transparent)',
    color: 'var(--text-primary, inherit)'
  });

  const heading = document.createElement('h3');
  heading.id = 'liveWebEvidenceTitle';
  heading.textContent = 'Live Web Evidence';

  Object.assign(heading.style, {
    margin: '0 0 10px',
    fontSize: '1.15rem'
  });

  section.appendChild(heading);
  if (assessment) {
  const evidenceStatus = document.createElement('p');
  const evidenceMessage = document.createElement('p');


const statusLabels = {
  published_fact_check_found: 'Published fact-check found',
  multiple_trusted_sources_match: 'Multiple potentially authoritative sources match',
  sources_found_unverified: 'Sources found — claim unverified',
  no_relevant_sources_returned: 'No usable sources found',
  search_unavailable: 'Evidence search unavailable',
  insufficient_input: 'Insufficient input',
  verified_true: 'Trusted sources confirm this claim',
  likely_false: 'Trusted sources contradict this claim',
  partially_verified: 'Trusted sources partly confirm this claim',
  questionable: 'A trusted source contradicts this claim',
  disputed: 'Trusted sources disagree',
  unverified: 'Sources found, claim not confirmed',
  unavailable: 'Evidence search unavailable',
  error: 'Trusted-source check could not run'
};


  evidenceStatus.textContent =
    statusLabels[assessment.status] || 'Evidence assessment unavailable';

  evidenceMessage.textContent = assessment.message || '';

  Object.assign(evidenceStatus.style, {
    fontWeight: '600',
    margin: '0 0 8px',
    lineHeight: '1.5'
  });

  Object.assign(evidenceMessage.style, {
    lineHeight: '1.6',
    margin: '0 0 14px'
  });

  section.appendChild(evidenceStatus);
  section.appendChild(evidenceMessage);
  if (assessment.verification_limit) {
  const limitation = document.createElement('p');

  limitation.textContent = assessment.verification_limit;

  Object.assign(limitation.style, {
    lineHeight: '1.6',
    margin: '0 0 14px',
    fontSize: '0.9rem',
    opacity: '0.88'
  });

  section.appendChild(limitation);
}
}

  const status = document.createElement('p');
  status.textContent =
    search.status === 'success'
      ? 'Web search completed. Review the sources below before deciding whether the claim is accurate.'
      : (search.message || 'Web evidence is currently unavailable.');

  Object.assign(status.style, {
    lineHeight: '1.6',
    margin: '0 0 14px'
  });

  section.appendChild(status);

  if (search.answer) {
    const summaryTitle = document.createElement('h4');
    summaryTitle.textContent = 'Search summary';
    summaryTitle.style.margin = '12px 0 6px';
    section.appendChild(summaryTitle);

    const summary = document.createElement('p');
    summary.textContent = search.answer;

    Object.assign(summary.style, {
      lineHeight: '1.7',
      whiteSpace: 'pre-wrap',
      overflowWrap: 'anywhere',
      margin: '0 0 16px'
    });

    section.appendChild(summary);
  }

  const results = Array.isArray(search.results)
    ? search.results
    : [];

  if (results.length) {
    const sourcesTitle = document.createElement('h4');
    sourcesTitle.textContent = `Sources found (${results.length})`;
    sourcesTitle.style.margin = '14px 0 10px';
    section.appendChild(sourcesTitle);

    results.forEach((item) => {
      if (!item || typeof item.url !== 'string') return;

      let parsedUrl;

      try {
        parsedUrl = new URL(item.url);
      } catch {
        return;
      }

      if (!['https:', 'http:'].includes(parsedUrl.protocol)) return;

      const article = document.createElement('article');

      Object.assign(article.style, {
        padding: '12px 0',
        borderTop: '1px solid var(--border-color, #dfe5e7)'
      });

      const link = document.createElement('a');
      link.href = parsedUrl.href;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = item.title || 'Open source article';

      Object.assign(link.style, {
        fontWeight: '600',
        lineHeight: '1.5',
        overflowWrap: 'anywhere'
      });

      article.appendChild(link);

      if (item.content) {
        const description = document.createElement('p');
        description.textContent = item.content;

        Object.assign(description.style, {
          lineHeight: '1.6',
          margin: '7px 0',
          overflowWrap: 'anywhere'
        });

        article.appendChild(description);
      }

      if (item.published_date) {
        const date = document.createElement('p');
        date.textContent = `Published: ${item.published_date}`;
        date.style.fontSize = '0.85rem';
        article.appendChild(date);
      }

      section.appendChild(article);
    });
  } else {
    const noResults = document.createElement('p');
    noResults.textContent =
      'No usable sources were returned. This does not mean the claim is false.';
    section.appendChild(noResults);
  }

  const disclaimer = document.createElement('p');
  disclaimer.textContent =
    'Search results are supporting evidence, not proof. Check the original source, date, location, and whether its details match the submitted claim.';

  Object.assign(disclaimer.style, {
    margin: '16px 0 0',
    fontSize: '0.9rem',
    lineHeight: '1.6'
  });

  section.appendChild(disclaimer);
  dashboard.appendChild(section);
}

  /* ==========================================================================
     Trusted-source verification (verdict card + warning action)
     ========================================================================== */

  function safeHttpUrl(url) {
    try {
      const parsed = new URL(url);
      return (parsed.protocol === 'http:' || parsed.protocol === 'https:') ? parsed.href : '';
    } catch (e) {
      return '';
    }
  }

  function vEl(tag, className, text) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text !== undefined && text !== null) el.textContent = text;
    return el;
  }

  const SOURCE_CATEGORY_LABELS = {
    official: 'Official',
    fact_check: 'Fact-check',
    fact_checker: 'Fact-checker',
    trusted_news: 'Trusted news',
    general: 'Other web'
  };

  function buildSourceList(title, sources, stanceClass) {
    const wrap = vEl('div', 'vf-sources');
    wrap.appendChild(vEl('h4', 'vf-sources-title', title));
    const list = vEl('ul', 'vf-source-list');

    sources.forEach((src) => {
      const item = vEl('li', `vf-source ${stanceClass}`);
      const top = vEl('div', 'vf-source-top');
      top.appendChild(vEl('span', 'vf-tag', SOURCE_CATEGORY_LABELS[src.category] || 'Source'));
      top.appendChild(vEl('span', 'vf-domain', src.domain || ''));
      if (src.rating) top.appendChild(vEl('span', 'vf-rating', `Rated: ${src.rating}`));
      item.appendChild(top);

      const href = safeHttpUrl(src.url);
      if (href) {
        const link = vEl('a', 'vf-source-link', src.title || href);
        link.href = href;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        item.appendChild(link);
      } else {
        item.appendChild(vEl('span', 'vf-source-link', src.title || 'Source'));
      }
      if (src.reason) item.appendChild(vEl('p', 'vf-source-reason', src.reason));
      if (src.snippet) item.appendChild(vEl('p', 'vf-source-snippet', 'Snippet judged: “' + src.snippet + '”'));
      list.appendChild(item);
    });

    wrap.appendChild(list);
    return wrap;
  }

  function renderVerification(data) {
    const old = document.getElementById('verificationCard');
    if (old) old.remove();

    const panel = document.getElementById('verdictPanel');
    const caption = document.getElementById('verdictCaption');
    const realLabel = document.getElementById('realScorePct')?.parentElement?.querySelector('.score-label');
    const fakeLabel = document.getElementById('fakeScorePct')?.parentElement?.querySelector('.score-label');

    const v = data && data.verification;

    // Restore default labels (the wording model) unless trusted sources decided the score.
    if (realLabel) realLabel.textContent = 'Real language patterns';
    if (fakeLabel) fakeLabel.textContent = 'Fake language patterns';

    if (!v || !panel) return;

    const decided = data.score_basis === 'trusted_sources';

    if (decided) {
      if (realLabel) realLabel.textContent = 'Real (evidence score)';
      if (fakeLabel) fakeLabel.textContent = 'Fake (evidence score)';
      panel.dataset.verdict =
        v.status === 'verified_true' ? 'real' : v.status === 'likely_false' ? 'fake' : 'uncertain';
      if (caption) caption.textContent = v.summary || '';
    } else if (caption) {
      caption.textContent =
        (v.summary || 'No trusted source confirmed this claim.') +
        ' The percentages below are the wording model’s estimate only.';
    }

    const action = v.action || {};
    const kind = action.type || 'caution';
    const card = vEl('section', `verification-card vf-${kind}`);
    card.id = 'verificationCard';
    card.setAttribute('aria-live', 'polite');

    const head = vEl('div', 'vf-head');
    head.appendChild(vEl('span', 'vf-icon', kind === 'warn' ? '!' : kind === 'confirm' ? '✓' : '?'));
    head.appendChild(vEl('h3', 'vf-headline', action.headline || 'Verification result'));
    card.appendChild(head);

    if (decided && Number.isFinite(Number(v.real_percentage))) {
      card.appendChild(vEl(
        'p', 'vf-score',
        `Evidence score: ${Number(v.real_percentage).toFixed(0)}% real · ${Number(v.fake_percentage).toFixed(0)}% fake, based on trusted sources`
      ));
    }

    if (v.claim_checked) {
      const claim = vEl('p', 'vf-claim');
      claim.appendChild(vEl('strong', '', 'Claim checked: '));
      claim.appendChild(document.createTextNode(v.claim_checked));
      card.appendChild(claim);
    }

    if (action.correct_info) {
      const info = vEl('p', 'vf-info');
      info.appendChild(vEl('strong', '', kind === 'warn' ? 'What trusted sources actually report: ' : 'What trusted sources report: '));
      info.appendChild(document.createTextNode(action.correct_info));
      card.appendChild(info);
    }

    if (Array.isArray(action.advice) && action.advice.length) {
      const ul = vEl('ul', 'vf-advice');
      action.advice.forEach((line) => ul.appendChild(vEl('li', '', line)));
      card.appendChild(ul);
    }

    if (kind === 'warn' && action.share_text) {
      const actions = vEl('div', 'vf-actions');
      const copyBtn = vEl('button', 'btn btn-outline btn-sm', 'Copy fact-check message');
      copyBtn.type = 'button';
      copyBtn.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(action.share_text);
          showToast('Fact-check message copied');
        } catch (e) {
          showToast('Could not copy the message.');
        }
      });
      actions.appendChild(copyBtn);
      card.appendChild(actions);
    }

    if (v.contradicting_sources && v.contradicting_sources.length) {
      card.appendChild(buildSourceList('Sources that contradict the claim', v.contradicting_sources, 'is-contradict'));
    }
    if (v.supporting_sources && v.supporting_sources.length) {
      card.appendChild(buildSourceList('Sources that support the claim', v.supporting_sources, 'is-support'));
    }
    if (!decided && v.related_sources && v.related_sources.length) {
      card.appendChild(buildSourceList('Related coverage (not confirming or denying)', v.related_sources, 'is-related'));
    }

    if (v.error) {
      card.appendChild(vEl('p', 'vf-limits', 'Technical detail: ' + v.error));
    }

    if (Array.isArray(v.limitations) && v.limitations.length) {
      card.appendChild(vEl('p', 'vf-limits', v.limitations.join(' ')));
    }

    panel.insertAdjacentElement('afterend', card);
  }

  function renderResults(data, sourceTag) {
    const dashboard = document.getElementById('resultsDashboard');
    const tagEl = document.getElementById('resultsSourceTag');
    const badgeEl = document.getElementById('predictionBadge');
    const realPctEl = document.getElementById('realScorePct');
    const fakePctEl = document.getElementById('fakeScorePct');
    const realBarEl = document.getElementById('realScoreBar');
    const fakeBarEl = document.getElementById('fakeScoreBar');

    const metaConfidenceEl = document.getElementById('metaConfidence');
    const metaWordsEl = document.getElementById('metaWords');
    const metaCharsEl = document.getElementById('metaChars');
    const metaTimestampEl = document.getElementById('metaTimestamp');
    const contentDisplay = document.getElementById('extractedContentDisplay');

    renderLanguageBadge(data);
renderWebSearch(data);
    if (!dashboard || !badgeEl) {
      console.error('Required results dashboard elements were not found.');
      showAlert('Could not display results because a required page element is missing.');
      return;
    }

    if (tagEl) tagEl.textContent = sourceTag || 'Analysis Result';
const realScore = Math.min(
  100,
  Math.max(0, parseFloat(data.real_percentage) || 0)
);

const fakeScore = Math.min(
  100,
  Math.max(0, parseFloat(data.fake_percentage) || 0)
);

// Be conservative when the model's label conflicts with its score.
// The backend's normal fake-news cutoff is 65%.
// The backend uses an even higher cutoff for translated text.
let displayLabel = data.label || 'Inconclusive / Mixed Evidence';

if (
  displayLabel === 'Likely Fake' &&
  fakeScore < 65
) {
  displayLabel = 'Inconclusive / Mixed Evidence';
}

if (
  displayLabel === 'Likely Real' &&
  fakeScore > 40
) {
  displayLabel = 'Inconclusive / Mixed Evidence';
}

let badgeClass = 'inconclusive';

if (displayLabel === 'Likely Real') {
  badgeClass = 'likely-real';
} else if (displayLabel === 'Likely Fake') {
  badgeClass = 'likely-fake';
} else if (displayLabel === 'Insufficient Evidence') {
  badgeClass = 'insufficient-evidence';
}

badgeEl.textContent = displayLabel;
badgeEl.className = `prediction-badge ${badgeClass}`;

    animateNumber(realPctEl, realScore, 1100);
    animateNumber(fakePctEl, fakeScore, 1100);

    // Verdict panel and credibility spectrum.
    const verdictPanel = document.getElementById('verdictPanel');
    const verdictMarker = document.getElementById('verdictMarker');
    const verdictCaption = document.getElementById('verdictCaption');

    if (verdictPanel) {
      const kind =
        badgeClass === 'likely-real'
          ? 'real'
          : badgeClass === 'likely-fake'
            ? 'fake'
            : 'uncertain';

      verdictPanel.dataset.verdict = kind;

      if (verdictCaption) {
        verdictCaption.textContent =
          kind === 'real'
            ? 'The wording resembles reliable reporting.'
            : kind === 'fake'
              ? 'The wording resembles known misinformation.'
              : (data.insufficient_evidence && data.warning
                  ? data.warning
                  : `Mixed signals: the wording scored ${fakeScore}% fake / ${realScore}% real, which is too close to call.`);
      }
    }

    // Place the marker according to the model's fake percentage.
    // Clamp it to the visible spectrum so it cannot disappear off-screen.
    if (verdictMarker) {
      verdictMarker.style.left = `${Math.min(96, Math.max(4, fakeScore))}%`;
    }

    // Animate score bars. These widths remain set after the animation.
    if (realBarEl) {
      realBarEl.style.width = '0%';

      requestAnimationFrame(() => {
        realBarEl.style.width = `${realScore}%`;
      });
    }

    if (fakeBarEl) {
      fakeBarEl.style.width = '0%';

      requestAnimationFrame(() => {
        fakeBarEl.style.width = `${fakeScore}%`;
      });
    }

    if (metaConfidenceEl) {
      metaConfidenceEl.textContent = data.confidence || 'Moderate';
    }

    const extractedText = data.extracted_text || '';
    const wordCount =
      data.analyzed_words ||
      (extractedText ? extractedText.trim().split(/\s+/).length : 0);

    const characterCount =
      data.analyzed_characters || extractedText.length;

    if (metaWordsEl) metaWordsEl.textContent = `${wordCount} words`;
    if (metaCharsEl) metaCharsEl.textContent = `${characterCount} characters`;

    const d = data.timestamp ? new Date(data.timestamp) : new Date();

    if (metaTimestampEl) {
      metaTimestampEl.textContent = d.toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
    }

    // Display analyzed content safely using textContent.
    let textToDisplay = '';

    if (data.source_type === 'image_ocr') {
      textToDisplay =
        `[EXTRACTED OCR TEXT FROM ${data.filename || 'IMAGE'}]\n\n` +
        `${data.extracted_text || ''}`;
    } else if (data.source_type === 'link') {
      const where = data.source_url || data.url || 'THE LINK';
      const title = data.article_title ? `${data.article_title}\n\n` : '';

      textToDisplay =
        `[ARTICLE TEXT READ FROM ${where}]\n\n` +
        `${title}${data.extracted_text || data.original_text || ''}`;
    } else {
      textToDisplay =
        data.extracted_text ||
        data.original_text ||
        data.text_snippet ||
        '';
    }

    if (contentDisplay) {
      contentDisplay.textContent =
        textToDisplay || 'No extracted content was returned by the server.';
    }

    renderVerification(data);
    renderFactChecks(data);

    dashboard.style.display = 'block';
    dashboard.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  window.copyExtractedContent = async function () {
    const element = document.getElementById('extractedContentDisplay');
    const content = element ? element.textContent : '';

    if (!content) {
      showToast('There is no extracted content to copy.');
      return;
    }

    try {
      await navigator.clipboard.writeText(content);
      showToast('Copied to clipboard');
    } catch (error) {
      // Fallback for browsers that do not allow Clipboard API access.
      const textarea = document.createElement('textarea');

      textarea.value = content;
      textarea.setAttribute('readonly', '');
      textarea.style.position = 'fixed';
      textarea.style.opacity = '0';

      document.body.appendChild(textarea);
      textarea.select();

      const copied = document.execCommand('copy');
      document.body.removeChild(textarea);

      showToast(copied ? 'Copied to clipboard' : 'Could not copy content.');
    }
  };

  window.resetResults = function () {
    const dashboard = document.getElementById('resultsDashboard');
    const workspace = document.getElementById('workspace');

    if (dashboard) dashboard.style.display = 'none';

    if (workspace) {
      workspace.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  /* ==========================================================================
     Helper Utilities
     ========================================================================== */

  function setProcessingState(isProc, btnEl) {
    state.isProcessing = isProc;

    if (btnEl) {
      if (isProc) {
        btnEl.classList.add('is-loading');
        btnEl.disabled = true;
      } else {
        btnEl.classList.remove('is-loading');

        // Keep upload buttons disabled if no file has been selected.
        if (btnEl.id === 'analyzeImageBtn') {
          btnEl.disabled = !state.selectedImageFile;
        } else {
          btnEl.disabled = false;
        }
      }
    }
  }

  function showAlert(msg) {
    const alertBox = document.getElementById('workspaceAlert');
    const content = document.getElementById('alertContent');

    if (alertBox && content) {
      content.textContent = msg;
      alertBox.style.display = 'flex';

      alertBox.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest',
      });
    } else {
      console.error(msg);
    }
  }

  function hideAlert() {
    const alertBox = document.getElementById('workspaceAlert');

    if (alertBox) {
      alertBox.style.display = 'none';
    }
  }
})();