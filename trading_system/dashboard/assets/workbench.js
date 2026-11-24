/* Trading 212 Quantitative Control Plane — Zero-Build SPA Workbench
 * Self-hosted vanilla ES module/script. Zero external CDNs, zero raw HTML string sinks,
 * zero 3.5s polling loops, and non-destructive SSE + focus-safe DOM patching.
 */
(function (globalScope) {
  "use strict";

  const TOKEN_STORAGE_KEY = "t212_dashboard_token";
  const HEARTBEAT_STALE_MS = 25000;
  const SSE_MIN_BACKOFF_MS = 1000;
  const SSE_MAX_BACKOFF_MS = 15000;
  const MAX_DYNAMIC_PARAMETERS = 64;
  const MAX_DYNAMIC_STRING_LENGTH = 256;
  const DYNAMIC_KEY_RE = /^[A-Za-z_][A-Za-z0-9_]{1,63}$/;
  const HHMM_RE = /^([01]\d|2[0-3]):([0-5]\d)$/;

  const GROUP_ORDER = [
    "strategy",
    "screener",
    "risk",
    "exit_tsl",
    "session",
    "leverage",
    "providers",
    "runtime",
  ];

  const GROUP_LABELS = {
    strategy: "Strategy & Trend Gates",
    screener: "Screener & Regime Filters",
    risk: "Risk & Position Sizing",
    exit_tsl: "Exit, Stops & TSL Decay",
    session: "Session & Rate Governance",
    leverage: "Leverage & Universe Mode",
    providers: "External Data & AI Providers",
    runtime: "Runtime & Restart-Required Settings",
  };

  const PARAM_TO_METRIC_HINTS = {
    MIN_ADX_TREND: ["adx"],
    STRATEGY_EXCEPTIONAL_ADX: ["adx"],
    REGIME_BREAKOUT_ADX_MIN: ["adx"],
    FALLBACK_SCREENER_MIN_ADX: ["adx"],
    MIN_EMA20_SLOPE_PCT: ["ema20_slope_5d"],
    ENTRY_EARNINGS_BLACKOUT_DAYS: ["earnings_in_days"],
    HIGH_VOL_MAX_REALIZED_VOL: ["realized_vol"],
    HIGH_VOL_MAX_ATR_SPIKE_RATIO: ["atr_spike_ratio"],
    ATR_SL_MULTIPLIER: ["atr", "atr_ratio"],
  };

  const GATE_TO_METRIC_HINTS = {
    adx_trend_gate: ["adx"],
    dmi_trend_gate: ["adx", "plus_di", "minus_di"],
    setup_quality_tier: ["adx"],
    setup_score_gate: ["adx"],
    ema_slope_gate: ["ema20_slope_5d"],
    earnings_blackout_gate: ["earnings_in_days"],
    catalyst_risk_gate: ["earnings_in_days"],
    high_vol_erratic_gate: ["realized_vol", "atr_spike_ratio"],
    regime_classifier: ["adx", "realized_vol", "atr_spike_ratio"],
    fallback_screener_adx_gate: ["adx"],
  };

  const STRUCTURAL_GATES = new Set([
    "stale_epoch_superseded",
    "market_closed",
    "system_halted",
    "insufficient_data",
    "no_etp_mapping",
    "circuit_breaker",
  ]);

  const PLAIN_ENGLISH_GATE_TRANSLATIONS = {
    adx_trend_gate: "Trend strength ADX below momentum floor (< 18.0)",
    min_adx_trend: "Trend strength ADX below minimum threshold (< 18.0)",
    dmi_trend_gate: "DMI directional indicator below breakout threshold",
    setup_quality_tier: "Setup quality score below favorable threshold",
    setup_score_gate: "Technical setup score below required threshold",
    ema_slope_gate: "20-day EMA trend slope insufficient (flat or downtrend)",
    min_ema20_slope_pct: "20-day EMA slope below required angle",
    earnings_blackout_gate: "Earnings release scheduled within 48h blackout window",
    earnings_blackout: "Earnings release scheduled within 48h blackout window",
    entry_earnings_blackout_days: "Earnings report too close to entry window",
    catalyst_risk_gate: "High catalyst risk from upcoming corporate action",
    high_vol_erratic_gate: "Realized volatility too erratic for entry",
    high_vol_max_realized_vol: "Realized volatility exceeds ceiling (> 65% annualized)",
    regime_classifier: "Market regime in low-volatility chop",
    macro_regime_chop: "Market regime in low-volatility chop",
    fallback_screener_adx_gate: "Fallback screener trend strength insufficient",
    spread_gate: "Broker bid-ask spread wider than limit (> 1.5%)",
    max_bid_ask_spread_pct: "Bid-ask spread too wide (> 1.5%)",
    correlation_limit: "Sector or portfolio correlation heat limit reached",
    max_sector_positions: "Sector position limit reached",
    circuit_breaker: "Daily circuit breaker active (drawdown limit reached)",
    stale_epoch_superseded: "Configuration updated mid-scan (cycle superseded)",
    market_closed: "Exchange session currently closed",
    system_halted: "Trading halted by emergency panic switch",
    insufficient_data: "Insufficient historical price candles for calculation",
    no_etp_mapping: "No eligible leveraged ETP pair found",
    vwap_gate: "Price trading below session VWAP benchmark",
    vwap_overextended: "Price overextended too far above session VWAP (> 3.5%)",
    vwap_dead_zone: "Price within VWAP dead-zone buffer (insufficient breakout distance)",
    counter_trend_filter: "Macro trend alignment check failed (below 200 SMA)",
    single_equity_cash_defense: "Single equity below VWAP: 100% Cash Defense enforced",
  };

  function translateGateRejection(gateId, label) {
    if (gateId && PLAIN_ENGLISH_GATE_TRANSLATIONS[gateId]) {
      return PLAIN_ENGLISH_GATE_TRANSLATIONS[gateId];
    }
    const cleanId = String(gateId || "").toLowerCase();
    for (const [key, text] of Object.entries(PLAIN_ENGLISH_GATE_TRANSLATIONS)) {
      if (cleanId.includes(key)) return text;
    }
    if (label && typeof label === "string" && label.length > 0) {
      return label;
    }
    return `Asset filtered out by ${gateId || "risk rule"}`;
  }

  function getDoc() {
    return typeof document !== "undefined" ? document : null;
  }

  function getWin() {
    return typeof window !== "undefined" ? window : globalScope;
  }

  function readInitialData() {
    try {
      if (typeof INITIAL_DATA !== "undefined" && INITIAL_DATA && typeof INITIAL_DATA === "object") {
        return INITIAL_DATA;
      }
    } catch (_) {
      // ignore lexical TDZ
    }
    const win = getWin();
    if (win && win.INITIAL_DATA && typeof win.INITIAL_DATA === "object") {
      return win.INITIAL_DATA;
    }
    return {};
  }

  const state = {
    initialized: false,
    root: null,
    data: {},
    route: { tab: "monitor", param: null, gate: null },
    viewMode: (function () {
      try {
        const win = getWin();
        const stored = win && win.sessionStorage && win.sessionStorage.getItem("quant_workbench_view_mode");
        return stored === "advanced" ? "advanced" : "simple";
      } catch (_) {
        return "simple";
      }
    })(),
    trackerExpanded: false,
    pipelineProgress: null,
    directionFilter: "ALL",
    configSearch: "",
    configGroup: "all",
    stagedChanges: {},
    draftBaseEpochId: null,
    previewData: null,
    previewLoading: false,
    previewError: null,
    previewTimer: null,
    rebaseState: null, // { activeEpochId, remoteConfig, choices: { [key]: 'mine'|'remote' } }
    isCommitting: false,
    reEpochModalLoading: false,
    drawerCollapsedMobile: true,
    sse: {
      source: null,
      status: "idle", // idle | connecting | connected | reconnecting | offline
      backoffMs: SSE_MIN_BACKOFF_MS,
      reconnectTimer: null,
      hasDisconnected: false,
      lastSnapshotAt: Date.now(),
      lastHeartbeatAt: Date.now(),
    },
    uiTimer: null,
    hashChangeHandler: null,
    activeModal: null, // { close, onEscape }
    pendingRetryAction: null,
    paramDomCache: {}, // key -> { card, input, baseBadge, stagedBadge, restartBadge, resetBtn }
    lastSchemaSignature: "",
  };

  /* ── Safe Storage & Protocol Helpers ───────────────────────────────────── */

  function isFileProtocol() {
    const win = getWin();
    try {
      return Boolean(win && win.location && win.location.protocol === "file:");
    } catch (_) {
      return false;
    }
  }

  function getStoredToken() {
    const win = getWin();
    try {
      if (win && win.sessionStorage) {
        return win.sessionStorage.getItem(TOKEN_STORAGE_KEY) || "";
      }
    } catch (_) {
      return "";
    }
    return "";
  }

  function setStoredToken(token) {
    const win = getWin();
    try {
      if (win && win.sessionStorage) {
        if (token) {
          win.sessionStorage.setItem(TOKEN_STORAGE_KEY, token);
        } else {
          win.sessionStorage.removeItem(TOKEN_STORAGE_KEY);
        }
      }
    } catch (_) {
      // ignore storage restrictions
    }
  }

  function clearStoredToken() {
    setStoredToken("");
  }

  function isUnlocked() {
    return getStoredToken().length > 0;
  }

  function isControllerAttached() {
    return state.data.controller_attached !== false;
  }

  function isReadOnlyMode() {
    return isFileProtocol() || !isControllerAttached();
  }

  function isSystemHalted() {
    const d = state.data || {};
    const broker = d.broker_status || {};
    return Boolean(d.is_halted || d.system_halted || d.halted || broker.is_halted);
  }

  function getActiveEpochId() {
    const d = state.data || {};
    if (d.config_epoch_id) return String(d.config_epoch_id);
    if (d.config && d.config.config_epoch_id) return String(d.config.config_epoch_id);
    if (d.active_epoch_id) return String(d.active_epoch_id);
    return "epoch-0";
  }

  function getRawConfigObject() {
    const d = state.data || {};
    return d.config && typeof d.config === "object" ? d.config : {};
  }

  function extractConfigValues(cfgObj) {
    if (!cfgObj || typeof cfgObj !== "object") return {};
    if (cfgObj.values && typeof cfgObj.values === "object" && !Array.isArray(cfgObj.values)) {
      return cfgObj.values;
    }
    return cfgObj;
  }

  function getActiveConfigValues() {
    return extractConfigValues(getRawConfigObject());
  }

  function getDynamicParametersMap() {
    const raw = getRawConfigObject();
    if (raw.dynamic_parameters && typeof raw.dynamic_parameters === "object") {
      return raw.dynamic_parameters;
    }
    if (state.data && state.data.dynamic_parameters && typeof state.data.dynamic_parameters === "object") {
      return state.data.dynamic_parameters;
    }
    return {};
  }

  function getSchemaObject() {
    const d = state.data || {};
    if (d.schema && typeof d.schema === "object") return d.schema;
    return {};
  }

  function getSchemaParameters() {
    const schema = getSchemaObject();
    const raw = getRawConfigObject();
    const params = Array.isArray(schema.parameters) ? schema.parameters.slice() : [];
    const dynSpecs = raw.dynamic_parameter_specs || state.data.dynamic_parameter_specs || [];
    const seen = new Set(params.map((p) => p && p.key));
    dynSpecs.forEach((spec) => {
      if (spec && spec.key && !seen.has(spec.key)) {
        params.push(Object.assign({ is_dynamic: true }, spec));
        seen.add(spec.key);
      }
    });
    return params;
  }

  function getBuiltinProfiles() {
    const d = state.data || {};
    const schema = getSchemaObject();
    return d.presets || d.builtin_profiles || schema.builtin_profiles || {};
  }

  function getPendingRestartKeys() {
    const d = state.data || {};
    const rawCfg = getRawConfigObject();
    const raw = Array.isArray(d.pending_restart_keys)
      ? d.pending_restart_keys
      : (Array.isArray(rawCfg.pending_restart_keys) ? rawCfg.pending_restart_keys : []);
    return Array.isArray(raw) ? raw.map(String) : [];
  }

  function getQuarantineWarnings() {
    const d = state.data || {};
    const rawCfg = getRawConfigObject();
    const raw = d.config_quarantine_warnings || rawCfg.config_quarantine_warnings || [];
    return Array.isArray(raw) ? raw.map(String).filter(Boolean) : [];
  }

  function getEffectiveConfigValue(key) {
    const cfg = getActiveConfigValues();
    if (Object.prototype.hasOwnProperty.call(cfg, key)) {
      return cfg[key];
    }
    const dyn = getDynamicParametersMap();
    if (Object.prototype.hasOwnProperty.call(dyn, key)) {
      return dyn[key];
    }
    const param = getSchemaParameters().find((p) => p.key === key);
    return param ? param.default : undefined;
  }

  /* ── Safe DOM Construction Helpers (Element/Text Nodes Only) ───────────── */

  function clearNode(node) {
    if (!node) return;
    while (node.firstChild) {
      node.removeChild(node.firstChild);
    }
  }

  function applyAttrs(node, attrs) {
    if (!attrs) return;
    Object.keys(attrs).forEach((k) => {
      const val = attrs[k];
      if (val === undefined || val === null || val === false) return;
      if (k === "text") {
        node.textContent = String(val);
      } else if (k === "className") {
        node.className = String(val);
      } else if (k === "htmlFor") {
        node.setAttribute("for", String(val));
      } else if (k === "hidden") {
        node.hidden = true;
      } else if (k === "disabled") {
        node.disabled = true;
        node.setAttribute("aria-disabled", "true");
      } else if (k === "checked") {
        node.checked = true;
      } else if (k === "value") {
        node.value = String(val);
      } else if (k === "dataset" && typeof val === "object") {
        Object.keys(val).forEach((dk) => {
          if (val[dk] !== undefined && val[dk] !== null) {
            node.dataset[dk] = String(val[dk]);
          }
        });
      } else if (k.startsWith("on") && typeof val === "function") {
        node.addEventListener(k.slice(2).toLowerCase(), val);
      } else {
        node.setAttribute(k, val === true ? "" : String(val));
      }
    });
  }

  function el(tag, attrs, ...children) {
    const doc = getDoc();
    const node = doc.createElement(tag);
    applyAttrs(node, attrs);
    appendChildren(node, children);
    return node;
  }

  function svgEl(tag, attrs, ...children) {
    const doc = getDoc();
    const node = doc.createElementNS("http://www.w3.org/2000/svg", tag);
    if (attrs) {
      Object.keys(attrs).forEach((k) => {
        const val = attrs[k];
        if (val === undefined || val === null) return;
        if (k === "text") {
          node.textContent = String(val);
        } else {
          node.setAttribute(k, String(val));
        }
      });
    }
    appendChildren(node, children);
    return node;
  }

  function appendChildren(parent, children) {
    const doc = getDoc();
    children.forEach((child) => {
      if (child === null || child === undefined || child === false) return;
      if (Array.isArray(child)) {
        appendChildren(parent, child);
      } else if (typeof child === "string" || typeof child === "number") {
        parent.appendChild(doc.createTextNode(String(child)));
      } else {
        parent.appendChild(child);
      }
    });
  }

  function isElementInside(node, container) {
    let cur = node;
    while (cur) {
      if (cur === container) return true;
      cur = cur.parentNode;
    }
    return false;
  }

  function getFocusedElement() {
    const doc = getDoc();
    return doc ? doc.activeElement : null;
  }

  function formatNumber(n, decimals) {
    const num = Number(n);
    if (!Number.isFinite(num)) return "0.00";
    return num.toLocaleString("en-US", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  }

  function formatCurrencyAmount(amount, prefix, suffix) {
    const num = Number(amount);
    if (!Number.isFinite(num)) return "—";
    const absStr = Math.abs(num).toFixed(2);
    const sign = num < 0 ? "-" : "";
    return `${sign}${prefix || ""}${absStr}${suffix || ""}`;
  }

  function formatPriceWithCurrency(px, curCode, prefix, suffix, opts = {}) {
    const num = Number(px);
    if (!Number.isFinite(num) || num <= 0) return "—";
    const code = (curCode || (suffix === "p" ? "GBX" : (prefix === "$" ? "USD" : "EUR"))).toUpperCase();
    const compact = Boolean(opts.compact);

    if (code === "GBX" || suffix === "p") {
      const gbp = num / 100.0;
      const eur = gbp * 1.204;
      const penceStr = formatNumber(num, 2) + "p";
      const gbpStr = "£" + formatNumber(gbp, 2);
      if (compact) {
        return `${penceStr} (${gbpStr})`;
      }
      return `${penceStr} (${gbpStr} • ≈ €${formatNumber(eur, 2)})`;
    }

    if (code === "USD" || prefix === "$") {
      const eur = num / 1.085;
      const usdStr = "$" + formatNumber(num, 2);
      if (compact) {
        return `${usdStr} (≈ €${formatNumber(eur, 2)})`;
      }
      return `${usdStr} (≈ €${formatNumber(eur, 2)})`;
    }

    if (code === "GBP" || prefix === "£") {
      const eur = num * 1.204;
      const gbpStr = "£" + formatNumber(num, 2);
      return `${gbpStr} (≈ €${formatNumber(eur, 2)})`;
    }

    return `€${formatNumber(num, 2)}`;
  }

  function valuesEqual(a, b) {
    if (Array.isArray(a) && Array.isArray(b)) {
      if (a.length !== b.length) return false;
      for (let i = 0; i < a.length; i += 1) {
        if (a[i] !== b[i]) return false;
      }
      return true;
    }
    if (typeof a === "number" && typeof b === "number") {
      return Math.abs(a - b) < 1e-12;
    }
    return a === b;
  }

  function formatShortValue(val) {
    if (val === null || val === undefined) return "null";
    if (Array.isArray(val)) return val.join(", ") || "(empty)";
    if (typeof val === "boolean") return val ? "true" : "false";
    return String(val);
  }

  /* ── Toasts ────────────────────────────────────────────────────────────── */

  function showToast(message, variant) {
    const doc = getDoc();
    const root = doc && doc.getElementById("toast-root");
    if (!root) return;
    const cls = variant === "error"
      ? "wb-toast wb-toast--error"
      : (variant === "success" ? "wb-toast wb-toast--success" : "wb-toast");
    const toast = el("div", { className: cls, role: "status", text: message });
    root.appendChild(toast);
    setTimeout(() => {
      if (toast.parentNode) toast.parentNode.removeChild(toast);
    }, 4800);
  }

  /* ── Modal Manager (Focus Trap + Escape + Non-Destructive Cancel) ──────── */

  function closeModal() {
    const doc = getDoc();
    const root = doc && doc.getElementById("modal-root");
    if (root) clearNode(root);
    const prevFocus = state.activeModal && state.activeModal.prevFocus;
    state.activeModal = null;
    if (prevFocus && typeof prevFocus.focus === "function") {
      try {
        prevFocus.focus();
      } catch (_) {
        // ignore detached elements
      }
    }
  }

  function openModal({ id, title, wide, drawer, buildBody, buildFooter, onCancel }) {
    const doc = getDoc();
    const root = doc && doc.getElementById("modal-root");
    if (!root) return null;

    const prevFocus = getFocusedElement();
    clearNode(root);

    const titleId = `${id || "wb-modal"}-title`;
    const bodyEl = el("div", { className: "wb-modal-body" });
    const footEl = el("div", { className: "wb-modal-foot" });
    const dialog = el(
      "div",
      {
        id: id || "wb-modal",
        className: drawer ? "wb-modal wb-modal--drawer" : (wide ? "wb-modal wb-modal--wide" : "wb-modal"),
        role: "dialog",
        "aria-modal": "true",
        "aria-labelledby": titleId,
      },
      el(
        "div",
        { className: "wb-modal-head" },
        el("h2", { id: titleId, className: "wb-modal-title", text: title }),
        el("button", {
          type: "button",
          className: "wb-btn wb-btn--sm wb-btn--ghost",
          "aria-label": "Close dialog",
          text: "✕",
          onClick: () => {
            closeModal();
            if (onCancel) onCancel();
          },
        })
      ),
      bodyEl,
      footEl
    );

    const backdrop = el(
      "div",
      {
        className: "wb-modal-backdrop",
        onClick: (ev) => {
          if (ev.target === backdrop) {
            closeModal();
            if (onCancel) onCancel();
          }
        },
      },
      dialog
    );

    if (buildBody) buildBody(bodyEl, dialog);
    if (buildFooter) buildFooter(footEl, dialog);

    dialog.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape") {
        ev.preventDefault();
        closeModal();
        if (onCancel) onCancel();
        return;
      }
      if (ev.key === "Tab") {
        const focusables = dialog.querySelectorAll(
          'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), a[href], [tabindex]:not([tabindex="-1"])'
        );
        if (!focusables || focusables.length === 0) return;
        const first = focusables[0];
        const last = focusables[focusables.length - 1];
        const active = getFocusedElement();
        if (ev.shiftKey && active === first) {
          ev.preventDefault();
          last.focus();
        } else if (!ev.shiftKey && active === last) {
          ev.preventDefault();
          first.focus();
        }
      }
    });

    root.appendChild(backdrop);
    state.activeModal = { prevFocus, dialog, onCancel };

    const firstFocus = dialog.querySelector(
      'input:not([disabled]), select:not([disabled]), button:not([disabled])'
    );
    if (firstFocus && typeof firstFocus.focus === "function") {
      try {
        firstFocus.focus();
      } catch (_) {
        // ignore
      }
    }
    return dialog;
  }

  /* ── HTTP API Gateway Client & Auth Unlock Flow ────────────────────────── */

  async function apiRequest(path, options) {
    const opts = options || {};
    const method = opts.method || "GET";

    if (isFileProtocol()) {
      return {
        ok: false,
        status: 0,
        data: { error: "Offline file:// viewer is strictly read-only." },
      };
    }

    if (method !== "GET" && !isControllerAttached()) {
      return {
        ok: false,
        status: 503,
        data: { error: state.data.read_only_reason || "Detached read-only server (no Orchestrator attached)." },
      };
    }

    const headers = Object.assign({}, opts.headers || {});
    if (method !== "GET") {
      headers["Content-Type"] = "application/json";
    }
    const token = opts.tokenOverride !== undefined ? opts.tokenOverride : getStoredToken();
    if (token) {
      headers["X-Dashboard-Token"] = token;
    }

    const fetchOpts = {
      method,
      headers,
      cache: "no-store",
    };
    if (opts.body !== undefined) {
      fetchOpts.body = typeof opts.body === "string" ? opts.body : JSON.stringify(opts.body);
    }

    try {
      const resp = await fetch(path, fetchOpts);
      let payload = {};
      try {
        payload = await resp.json();
      } catch (_) {
        payload = {};
      }
      if ((resp.status === 401 || resp.status === 403) && !opts.skipAuthModal) {
        if (token) {
          clearStoredToken();
          renderHeader();
          renderDiffDrawer();
        }
        if (typeof opts.retryFactory === "function") {
          state.pendingRetryAction = opts.retryFactory;
        }
        openAuthModal(payload.error || "Authentication required for mutating control actions.");
      }
      return { ok: resp.ok, status: resp.status, data: payload };
    } catch (err) {
      return {
        ok: false,
        status: 0,
        data: { error: err && err.message ? err.message : "Network error contacting dashboard server" },
      };
    }
  }

  function openAuthModal(initialNotice) {
    if (isFileProtocol()) {
      showToast("Offline file:// viewer cannot authenticate.", "error");
      return;
    }
    if (!isControllerAttached()) {
      showToast("Server is running in detached read-only mode.", "error");
      return;
    }

    let tokenInput = null;
    let errorBox = null;
    let submitBtn = null;

    openModal({
      id: "auth-modal",
      title: "Unlock Dashboard Control Plane",
      onCancel: () => {
        // Escape/Cancel MUST NEVER clear stagedChanges or draftBaseEpochId
        state.pendingRetryAction = null;
      },
      buildBody: (bodyEl) => {
        errorBox = el("div", {
          className: "wb-banner wb-banner--danger",
          hidden: !initialNotice,
          text: initialNotice || "",
        });
        tokenInput = el("input", {
          id: "auth-token-input",
          type: "password",
          className: "wb-input wb-input--mono",
          autocomplete: "off",
          placeholder: "Paste dashboard token...",
        });
        const form = el(
          "form",
          {
            id: "auth-modal-form",
            onSubmit: async (ev) => {
              ev.preventDefault();
              await submitAuth();
            },
          },
          el(
            "p",
            { className: "wb-param-hint", style: "margin-bottom:0.65rem" },
            "Token is stored only in browser sessionStorage. Pending staged config edits remain preserved."
          ),
          el(
            "div",
            { className: "wb-callout", style: "margin-bottom:0.75rem" },
            el("strong", { text: "Raspberry Pi CLI hint: " }),
            el("code", { className: "wb-mono", text: "python3 -m trading_system.dashboard.server --print-token" }),
            " (or inspect ",
            el("code", { className: "wb-mono", text: "data/.dashboard_token" }),
            ")."
          ),
          errorBox,
          el(
            "div",
            { className: "wb-field" },
            el("label", { className: "wb-label", htmlFor: "auth-token-input", text: "Dashboard Token" }),
            tokenInput
          )
        );
        bodyEl.appendChild(form);
      },
      buildFooter: (footEl) => {
        const cancelBtn = el("button", {
          type: "button",
          className: "wb-btn wb-btn--ghost",
          text: "Cancel",
          onClick: () => {
            closeModal();
            state.pendingRetryAction = null;
          },
        });
        submitBtn = el("button", {
          type: "button",
          id: "auth-submit-btn",
          className: "wb-btn wb-btn--primary",
          text: "Verify & Unlock",
          onClick: submitAuth,
        });
        footEl.appendChild(cancelBtn);
        footEl.appendChild(submitBtn);
      },
    });

    async function submitAuth() {
      const candidate = tokenInput ? tokenInput.value.trim() : "";
      if (!candidate) {
        if (errorBox) {
          errorBox.hidden = false;
          errorBox.textContent = "Please enter the dashboard token.";
        }
        return;
      }
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.textContent = "Verifying...";
      }
      const res = await apiRequest("/api/auth/verify", {
        method: "POST",
        body: {},
        tokenOverride: candidate,
        skipAuthModal: true,
      });
      if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.textContent = "Verify & Unlock";
      }
      if (res.ok) {
        setStoredToken(candidate);
        const retry = state.pendingRetryAction;
        state.pendingRetryAction = null;
        closeModal();
        renderHeader();
        renderDiffDrawer();
        showToast("Control plane unlocked for this browser session.", "success");
        if (typeof retry === "function") {
          await retry();
        } else if (Object.keys(state.stagedChanges).length > 0 && state.route.tab === "config") {
          scheduleServerPreview();
        }
      } else {
        if (errorBox) {
          errorBox.hidden = false;
          const retryHint = res.data && res.data.retry_after ? ` (Retry in ${res.data.retry_after}s)` : "";
          errorBox.textContent = (res.data && res.data.error ? res.data.error : `Verification failed (${res.status})`) + retryHint;
        }
      }
    }
  }

  function openQuickSettingsDrawer() {
    if (isFileProtocol()) {
      showToast("Offline file:// viewer cannot modify settings.", "warn");
      return;
    }
    const currentValues = getActiveConfigValues();
    const staged = state.stagedChanges || {};
    const riskButtons = [];

    openModal({
      id: "quick-settings-drawer",
      title: "⚙️ Quick Strategy Settings",
      drawer: true,
      buildBody: (bodyEl) => {
        bodyEl.classList.add("wb-drawer-body");

        bodyEl.appendChild(
          el(
            "p",
            { className: "wb-param-hint", style: "margin-bottom:1.25rem; font-size:0.82rem;" },
            "Configure core quantitative parameters below. Changes hot-reload dynamically via ConfigBus without restarting the trading engine."
          )
        );

        const coreGrid = el("div", { className: "wb-quick-ctrl-grid" });

        // 1. Active Strategy
        const stratVal = staged.ACTIVE_STRATEGY !== undefined ? staged.ACTIVE_STRATEGY : (currentValues.ACTIVE_STRATEGY || "vwap");
        const stratSelect = el(
          "select",
          {
            id: "qs-strategy-select",
            className: "wb-select wb-mono",
            onChange: (e) => stageParameterValue("ACTIVE_STRATEGY", e.target.value),
          },
          el("option", { value: "vwap", selected: stratVal === "vwap", text: "VWAP Session Trend Engine (Intraday Breakout)" }),
          el("option", { value: "ml_alpha", selected: stratVal === "ml_alpha", text: "ML Alpha Engine (Multi-Factor ONNX)" })
        );
        coreGrid.appendChild(
          el(
            "div",
            { className: "wb-quick-ctrl-card" },
            el("label", { htmlFor: "qs-strategy-select", className: "wb-quick-ctrl-label", text: "1. Active Strategy" }),
            stratSelect,
            el("span", { className: "wb-param-hint", text: "Governs universe candidate screening math and entry/exit gates." })
          )
        );

        // 2. Risk Profile
        const curRisk = Number(staged.MAX_RISK_PER_TRADE !== undefined ? staged.MAX_RISK_PER_TRADE : (currentValues.MAX_RISK_PER_TRADE || 0.015));
        const riskBtnRow = el("div", { className: "wb-pill-row", style: "margin-top:0.4rem; gap:0.4rem;" });
        const profiles = [
          [0.01, "Conservative 1.0%"],
          [0.015, "Balanced 1.5%"],
          [0.02, "Aggressive 2.0%"],
        ];
        profiles.forEach(([pct, label]) => {
          const btn = el("button", {
            type: "button",
            className: Math.abs(curRisk - pct) < 0.001 ? "wb-btn wb-btn--sm wb-btn--primary" : "wb-btn wb-btn--sm",
            text: label,
            onClick: () => {
              stageParameterValue("MAX_RISK_PER_TRADE", pct);
              profiles.forEach((p, idx) => {
                if (riskButtons[idx]) {
                  riskButtons[idx].className = (Math.abs(p[0] - pct) < 0.001) ? "wb-btn wb-btn--sm wb-btn--primary" : "wb-btn wb-btn--sm";
                }
              });
            },
          });
          riskButtons.push(btn);
          riskBtnRow.appendChild(btn);
        });
        coreGrid.appendChild(
          el(
            "div",
            { className: "wb-quick-ctrl-card" },
            el("div", { className: "wb-quick-ctrl-label", text: "2. Risk Profile (Per Trade)" }),
            riskBtnRow,
            el("span", { className: "wb-param-hint", text: "Controls capital allocation per trade based on account equity." })
          )
        );

        // 3. Stop Loss ATR Buffer
        const curSl = Number(staged.ATR_SL_MULTIPLIER !== undefined ? staged.ATR_SL_MULTIPLIER : (currentValues.ATR_SL_MULTIPLIER || 2.2));
        const slInput = el("input", {
          id: "qs-sl-input",
          type: "number",
          step: "0.1",
          min: "1.0",
          max: "5.0",
          className: "wb-input wb-mono",
          value: String(curSl),
          onChange: (e) => stageParameterValue("ATR_SL_MULTIPLIER", parseFloat(e.target.value)),
        });
        coreGrid.appendChild(
          el(
            "div",
            { className: "wb-quick-ctrl-card" },
            el("label", { htmlFor: "qs-sl-input", className: "wb-quick-ctrl-label", text: "3. Trailing Stop Loss Buffer (x ATR)" }),
            slInput,
            el("span", { className: "wb-param-hint", text: "ATR distance below entry for initial stop and dynamic trailing ratchets." })
          )
        );

        // 4. Session Close Cutoff Time
        const curCutoff = String(staged.SESSION_CLOSE_CUTOFF_UTC !== undefined ? staged.SESSION_CLOSE_CUTOFF_UTC : (currentValues.SESSION_CLOSE_CUTOFF_UTC || "16:20"));
        const cutoffInput = el("input", {
          id: "qs-cutoff-input",
          type: "text",
          className: "wb-input wb-mono",
          value: curCutoff,
          placeholder: "16:20",
          onChange: (e) => stageParameterValue("SESSION_CLOSE_CUTOFF_UTC", e.target.value.trim()),
        });
        coreGrid.appendChild(
          el(
            "div",
            { className: "wb-quick-ctrl-card" },
            el("label", { htmlFor: "qs-cutoff-input", className: "wb-quick-ctrl-label", text: "4. Session Close Evacuation (UTC)" }),
            cutoffInput,
            el("span", { className: "wb-param-hint", text: "Cutoff time to flatten unhedged leverage and prevent overnight gap risk." })
          )
        );

        bodyEl.appendChild(coreGrid);

        // Full parameters navigation link
        bodyEl.appendChild(
          el(
            "div",
            { style: "margin-top:1.5rem; padding-top:1rem; border-top:1px solid var(--border-subtle);" },
            el(
              "p",
              { className: "wb-param-hint", style: "font-size:0.8rem;" },
              "Need deeper configuration? Access all quantitative parameters in the ",
              el("a", {
                href: "#config",
                onClick: (e) => {
                  e.preventDefault();
                  closeModal();
                  navigateTo("config");
                },
                text: "Full Config & Diff Workspace →",
                style: "color:var(--sky); font-weight:600; text-decoration:none;",
              })
            )
          )
        );
      },
      buildFooter: (footEl) => {
        footEl.appendChild(
          el("button", {
            type: "button",
            className: "wb-btn wb-btn--ghost",
            text: "Cancel",
            onClick: () => closeModal(),
          })
        );
        footEl.appendChild(
          el("button", {
            type: "button",
            id: "qs-apply-btn",
            className: "wb-btn wb-btn--primary",
            text: "Apply & Hot-Reload",
            onClick: async () => {
              if (Object.keys(state.stagedChanges).length === 0) {
                showToast("No settings changes were made.", "info");
                closeModal();
                return;
              }
              closeModal();
              await submitConfigCommit("Quick Settings Update");
            },
          })
        );
      },
    });
  }

  function toggleQuickSettingsDrawer(open) {
    if (open === false) {
      closeModal();
    } else {
      openQuickSettingsDrawer();
    }
  }

  /* ── Route & Deep-Link Parsing ─────────────────────────────────────────── */

  function parseHashRoute() {
    const win = getWin();
    const rawHash = (win && win.location && win.location.hash) ? win.location.hash : "#monitor";
    const stripped = rawHash.replace(/^#\/?/, "");
    const qIdx = stripped.indexOf("?");
    const tabPart = (qIdx >= 0 ? stripped.slice(0, qIdx) : stripped).replace(/^\/+/, "").toLowerCase();
    const queryPart = qIdx >= 0 ? stripped.slice(qIdx + 1) : "";

    const validTabs = ["monitor", "config"];
    const tab = validTabs.includes(tabPart) ? tabPart : "monitor";
    let param = null;
    let gate = null;
    if (queryPart) {
      try {
        const sp = new URLSearchParams(queryPart);
        param = sp.get("param") || sp.get("key");
        gate = sp.get("gate");
      } catch (_) {
        // ignore malformed query
      }
    }
    return { tab, param, gate };
  }

  function navigateTo(tab, queryParams) {
    const win = getWin();
    let hash = `#${tab}`;
    if (queryParams && (queryParams.param || queryParams.gate)) {
      const parts = [];
      if (queryParams.param) parts.push(`param=${encodeURIComponent(queryParams.param)}`);
      if (queryParams.gate) parts.push(`gate=${encodeURIComponent(queryParams.gate)}`);
      if (parts.length > 0) hash += `?${parts.join("&")}`;
    }
    state.route = {
      tab,
      param: queryParams && queryParams.param ? queryParams.param : null,
      gate: queryParams && queryParams.gate ? queryParams.gate : null,
    };
    if (win && win.location && win.location.hash !== hash) {
      win.location.hash = hash;
    }
    applyRouteVisibility();
  }

  function applyRouteVisibility() {
    const doc = getDoc();
    if (!doc) return;
    const activeTab = state.route.tab || "monitor";
    ["monitor", "config"].forEach((tabId) => {
      const tabBtn = doc.getElementById(`tab-${tabId}`);
      const panel = doc.getElementById(`workspace-${tabId}`);
      const isSelected = tabId === activeTab;
      if (tabBtn) {
        tabBtn.setAttribute("aria-selected", isSelected ? "true" : "false");
        tabBtn.setAttribute("tabindex", isSelected ? "0" : "-1");
      }
      if (panel) {
        panel.hidden = !isSelected;
      }
    });

    if (activeTab === "config") {
      renderConfigWorkspace();
      renderDiffDrawer();
      focusDeepLinkedParam();
    } else {
      renderMonitorWorkspace();
    }
  }

  function focusDeepLinkedParam() {
    const doc = getDoc();
    if (!doc || !state.route.param) return;
    const card = doc.querySelector(`.wb-param-card[data-param-key="${cssEscapeSimple(state.route.param)}"]`);
    if (card) {
      if (typeof card.scrollIntoView === "function") {
        try {
          card.scrollIntoView({ behavior: "smooth", block: "center" });
        } catch (_) {
          // ignore non-browser test mocks
        }
      }
      const ctrl = card.querySelector("input, select");
      if (ctrl && typeof ctrl.focus === "function" && (!state.activeModal)) {
        try {
          ctrl.focus();
        } catch (_) {
          // ignore
        }
      }
    }
  }

  function cssEscapeSimple(str) {
    return String(str).replace(/[^A-Za-z0-9_-]/g, "\\$&");
  }

  /* ── Local Validation & Diff Engine ────────────────────────────────────── */

  function validateSingleValue(spec, val) {
    if (!spec) return null;
    const key = spec.key;
    if (val === null || val === undefined) {
      if (spec.nullable) return null;
      return { key, code: "invalid_type", message: `${key} cannot be null` };
    }
    if (spec.type === "bool") {
      if (typeof val !== "boolean") return { key, code: "invalid_type", message: `${key} must be boolean` };
    } else if (spec.type === "int") {
      if (typeof val !== "number" || !Number.isInteger(val)) {
        return { key, code: "invalid_type", message: `${key} must be an integer` };
      }
    } else if (spec.type === "float") {
      if (typeof val !== "number" || !Number.isFinite(val)) {
        return { key, code: "non_finite_float", message: `${key} must be a finite number` };
      }
    } else if (spec.type === "str") {
      if (typeof val !== "string") return { key, code: "invalid_type", message: `${key} must be a string` };
      if (val.length > MAX_DYNAMIC_STRING_LENGTH) {
        return { key, code: "string_too_long", message: `${key} exceeds ${MAX_DYNAMIC_STRING_LENGTH} characters` };
      }
      if (!val.trim()) return { key, code: "invalid_value", message: `${key} must be non-empty` };
      if (key === "SESSION_CLOSE_CUTOFF_UTC" && !HHMM_RE.test(val.trim())) {
        return { key, code: "invalid_format", message: `${key} must match HH:MM (UTC 24h)` };
      }
    } else if (spec.type === "string_list") {
      if (!Array.isArray(val)) return { key, code: "invalid_type", message: `${key} must be a list of strings` };
    }

    if ((spec.type === "int" || spec.type === "float") && typeof val === "number") {
      if (spec.min !== null && spec.min !== undefined && val < Number(spec.min)) {
        return { key, code: "out_of_bounds", message: `${key} (${val}) < min ${spec.min}` };
      }
      if (spec.max !== null && spec.max !== undefined && val > Number(spec.max)) {
        return { key, code: "out_of_bounds", message: `${key} (${val}) > max ${spec.max}` };
      }
    }
    if (Array.isArray(spec.allowed_values) && spec.allowed_values.length > 0) {
      if (!spec.allowed_values.includes(val)) {
        return {
          key,
          code: "invalid_value",
          message: `${key} must be one of ${spec.allowed_values.join(", ")}`,
        };
      }
    }
    return null;
  }

  function computeLocalValidationAndDiff(baseConfig, stagedChanges, schemaParams) {
    const schemaByKey = {};
    schemaParams.forEach((p) => {
      if (p && p.key) schemaByKey[p.key] = p;
    });

    const diff = [];
    const errors = [];
    const nonSimulatable = [];
    const merged = Object.assign({}, extractConfigValues(baseConfig));
    const dynMap = getDynamicParametersMap();
    Object.keys(dynMap).forEach((k) => {
      if (!Object.prototype.hasOwnProperty.call(merged, k)) merged[k] = dynMap[k];
    });

    Object.keys(stagedChanges || {}).forEach((key) => {
      const spec = schemaByKey[key];
      const newVal = stagedChanges[key];
      const oldVal = getEffectiveConfigValue(key);
      merged[key] = newVal;

      if (!spec) {
        errors.push({ key, code: "unknown_key", message: `Parameter ${key} is no longer in schema` });
        return;
      }
      const err = validateSingleValue(spec, newVal);
      if (err) errors.push(err);

      const simulatable = spec.group === "strategy" || spec.group === "screener";
      if (!simulatable) {
        nonSimulatable.push({
          key,
          group: spec.group,
          reload_mode: spec.reload_mode || "hot_reload",
        });
      }

      diff.push({
        key,
        label: spec.label || key,
        group: spec.group || "screener",
        unit: spec.unit || "",
        old_value: oldVal,
        new_value: newVal,
        reload_mode: spec.reload_mode || "hot_reload",
        revalidation_scope: spec.revalidation_scope || "resize_entry",
        simulatable,
      });
    });

    // Cross-field invariants matching trading_system/config.py
    if (
      typeof merged.MAX_RISK_PER_TRADE === "number" &&
      typeof merged.MAX_POSITION_PCT === "number" &&
      merged.MAX_RISK_PER_TRADE > merged.MAX_POSITION_PCT
    ) {
      errors.push({
        key: "MAX_RISK_PER_TRADE",
        code: "cross_field_error",
        message: "MAX_RISK_PER_TRADE cannot exceed MAX_POSITION_PCT",
      });
    }
    if (
      typeof merged.MIN_SL_PCT === "number" &&
      typeof merged.MAX_SL_PCT === "number" &&
      merged.MAX_SL_PCT <= merged.MIN_SL_PCT
    ) {
      errors.push({
        key: "MAX_SL_PCT",
        code: "cross_field_error",
        message: "MAX_SL_PCT must be greater than MIN_SL_PCT",
      });
    }
    if (
      typeof merged.ETP_MIN_SL_PCT === "number" &&
      typeof merged.ETP_MAX_SL_PCT === "number" &&
      merged.ETP_MAX_SL_PCT <= merged.ETP_MIN_SL_PCT
    ) {
      errors.push({
        key: "ETP_MAX_SL_PCT",
        code: "cross_field_error",
        message: "ETP_MAX_SL_PCT must be greater than ETP_MIN_SL_PCT",
      });
    }
    if (
      typeof merged.TP1_R_MULTIPLE === "number" &&
      typeof merged.TP2_R_MULTIPLE === "number" &&
      typeof merged.TP3_R_MULTIPLE === "number" &&
      !(merged.TP1_R_MULTIPLE < merged.TP2_R_MULTIPLE && merged.TP2_R_MULTIPLE < merged.TP3_R_MULTIPLE)
    ) {
      errors.push({
        key: "TP2_R_MULTIPLE",
        code: "cross_field_error",
        message: "Take-profit multiples must satisfy TP1 < TP2 < TP3",
      });
    }
    if (
      typeof merged.TP1_TRIM_PCT === "number" &&
      typeof merged.TP2_TRIM_PCT === "number" &&
      typeof merged.TP3_TRIM_PCT === "number"
    ) {
      const sum = merged.TP1_TRIM_PCT + merged.TP2_TRIM_PCT + merged.TP3_TRIM_PCT;
      if (Math.abs(sum - 1.0) > 1e-4) {
        errors.push({
          key: "TP1_TRIM_PCT",
          code: "cross_field_error",
          message: `TP1+TP2+TP3 trim percentages must sum to 1.0 (currently ${sum.toFixed(2)})`,
        });
      }
    }
    if (
      typeof merged.TSL_DECAY_DAY1_MULT === "number" &&
      typeof merged.TSL_DECAY_FLOOR_MULT === "number" &&
      merged.TSL_DECAY_DAY1_MULT < merged.TSL_DECAY_FLOOR_MULT
    ) {
      errors.push({
        key: "TSL_DECAY_DAY1_MULT",
        code: "cross_field_error",
        message: "TSL_DECAY_DAY1_MULT cannot be below TSL_DECAY_FLOOR_MULT",
      });
    }
    if (
      typeof merged.TSL_DECAY_RUNNER_MULT === "number" &&
      typeof merged.TSL_DECAY_FLOOR_MULT === "number" &&
      merged.TSL_DECAY_RUNNER_MULT < merged.TSL_DECAY_FLOOR_MULT
    ) {
      errors.push({
        key: "TSL_DECAY_RUNNER_MULT",
        code: "cross_field_error",
        message: "TSL_DECAY_RUNNER_MULT cannot be below TSL_DECAY_FLOOR_MULT",
      });
    }
    if (merged.ENABLE_5X_LEVERAGE === true && Number(merged.MAX_LEVERAGE) < 5) {
      errors.push({
        key: "MAX_LEVERAGE",
        code: "leverage_conflict",
        message: "ENABLE_5X_LEVERAGE=true conflicts with MAX_LEVERAGE < 5",
      });
    }
    if (merged.ENABLE_5X_LEVERAGE === false && Number(merged.MAX_LEVERAGE) > 3) {
      errors.push({
        key: "MAX_LEVERAGE",
        code: "leverage_conflict",
        message: "ENABLE_5X_LEVERAGE=false conflicts with MAX_LEVERAGE > 3",
      });
    }
    if (merged.PREFER_5X_LEVERAGE === true && merged.ENABLE_5X_LEVERAGE !== true) {
      errors.push({
        key: "PREFER_5X_LEVERAGE",
        code: "leverage_conflict",
        message: "PREFER_5X_LEVERAGE=true requires ENABLE_5X_LEVERAGE=true",
      });
    }

    return { diff, errors, nonSimulatable };
  }

  /* ── Staging Mutations ─────────────────────────────────────────────────── */

  function stageParameterValue(key, rawValue) {
    if (Object.keys(state.stagedChanges).length === 0 && !state.draftBaseEpochId) {
      state.draftBaseEpochId = getActiveEpochId();
    }
    const baseVal = getEffectiveConfigValue(key);
    if (valuesEqual(rawValue, baseVal)) {
      delete state.stagedChanges[key];
    } else {
      state.stagedChanges[key] = rawValue;
    }
    if (Object.keys(state.stagedChanges).length === 0) {
      state.draftBaseEpochId = null;
      state.previewData = null;
      state.previewError = null;
      state.rebaseState = null;
    } else if (state.drawerCollapsedMobile) {
      state.drawerCollapsedMobile = false;
    }
    updateConfigCardStagingUI(key);
    renderDiffDrawer();
    scheduleServerPreview();
  }

  function updateConfigCardStagingUI(key) {
    const entry = state.paramDomCache[key];
    if (!entry) return;
    const isStaged = Object.prototype.hasOwnProperty.call(state.stagedChanges, key);
    if (isStaged) {
      entry.card.classList.add("is-staged");
    } else {
      entry.card.classList.remove("is-staged");
    }
    if (entry.stagedBadge) entry.stagedBadge.hidden = !isStaged;
    if (entry.resetBtn) entry.resetBtn.hidden = !isStaged;
  }

  function discardStagedDraft() {
    state.stagedChanges = {};
    state.draftBaseEpochId = null;
    state.previewData = null;
    state.previewError = null;
    state.rebaseState = null;
    if (state.previewTimer) {
      clearTimeout(state.previewTimer);
      state.previewTimer = null;
    }
    renderConfigWorkspace(true);
    renderDiffDrawer();
  }

  function scheduleServerPreview() {
    if (state.previewTimer) {
      clearTimeout(state.previewTimer);
      state.previewTimer = null;
    }
    const keys = Object.keys(state.stagedChanges);
    if (keys.length === 0) {
      state.previewData = null;
      state.previewError = null;
      return;
    }
    // R16 / F1: While Locked or Read-Only, NEVER call /api/config/preview
    if (!isUnlocked() || isReadOnlyMode()) {
      state.previewData = null;
      state.previewError = null;
      return;
    }
    state.previewTimer = setTimeout(() => {
      state.previewTimer = null;
      runServerPreview();
    }, 220);
  }

  async function runServerPreview() {
    if (!isUnlocked() || isReadOnlyMode() || Object.keys(state.stagedChanges).length === 0) {
      return;
    }
    state.previewLoading = true;
    state.previewError = null;
    renderDiffDrawer();
    const res = await apiRequest("/api/config/preview", {
      method: "POST",
      body: {
        patch: state.stagedChanges,
        expected_epoch_id: state.draftBaseEpochId || getActiveEpochId(),
      },
      skipAuthModal: true,
    });
    state.previewLoading = false;
    if (res.ok) {
      state.previewData = res.data;
      state.previewError = null;
    } else if (res.status === 401) {
      clearStoredToken();
      state.previewData = null;
      state.previewError = "Session token rejected; unlock to run server preview.";
      renderHeader();
    } else {
      state.previewData = null;
      state.previewError = (res.data && res.data.error) || `Preview failed (${res.status})`;
    }
    renderDiffDrawer();
  }

  async function submitConfigCommit(reasonText) {
    if (state.isCommitting || Object.keys(state.stagedChanges).length === 0) return;
    state.isCommitting = true;
    renderDiffDrawer();
    try {
      const expectedEpoch = state.draftBaseEpochId || getActiveEpochId();
      const body = {
        patch: Object.assign({}, state.stagedChanges),
        expected_epoch_id: expectedEpoch,
        reason: reasonText || "",
      };

      const res = await apiRequest("/api/config", {
        method: "POST",
        body,
        retryFactory: () => submitConfigCommit(reasonText),
      });

      if (res.ok) {
        applyMutationSnapshotResponse(res.data);
        const newEpoch = getActiveEpochId();
        discardStagedDraft();
        showToast(`Committed configuration epoch ${newEpoch}.`, "success");
        return;
      }

      if (res.status === 409) {
        const remoteEpoch = (res.data && (res.data.active_epoch_id || res.data.actual_epoch_id)) || getActiveEpochId();
        const remoteConfig = (res.data && res.data.config) || getActiveConfigValues();
        if (res.data && res.data.config) {
          state.data.config = res.data.config;
          state.data.config_epoch_id = remoteEpoch;
        }
        if (res.data && Array.isArray(res.data.pending_restart_keys)) {
          state.data.pending_restart_keys = res.data.pending_restart_keys;
        }
        const choices = {};
        Object.keys(state.stagedChanges).forEach((k) => {
          choices[k] = "mine";
        });
        state.rebaseState = {
          activeEpochId: remoteEpoch,
          remoteConfig,
          errorMessage: (res.data && res.data.error) || `Epoch conflict: remote is ${remoteEpoch}`,
          choices,
        };
        renderHeader();
        renderDiffDrawer();
        showToast(`Epoch conflict (${remoteEpoch}). Rebase your staged edits before committing.`, "error");
        return;
      }

      if (res.status !== 401 && res.status !== 403) {
        const msg = (res.data && res.data.error) || `Commit failed (${res.status})`;
        state.previewError = msg;
        renderDiffDrawer();
        showToast(msg, "error");
      }
    } finally {
      state.isCommitting = false;
      renderDiffDrawer();
    }
  }

  function applyRebaseChoices(overrideChoices) {
    if (!state.rebaseState) return;
    const { activeEpochId, remoteConfig, choices } = state.rebaseState;
    if (overrideChoices && typeof overrideChoices === "object" && typeof overrideChoices.preventDefault !== "function") {
      Object.assign(choices, overrideChoices);
    }
    const currentLiveEpoch = getActiveEpochId();
    const sseAdvanced = Boolean(currentLiveEpoch && activeEpochId && currentLiveEpoch !== activeEpochId);
    const targetEpochId = sseAdvanced ? currentLiveEpoch : activeEpochId;
    const targetRemoteConfig = sseAdvanced ? getRawConfigObject() : remoteConfig;
    const remoteValues = extractConfigValues(targetRemoteConfig);
    const nextStaged = {};
    Object.keys(state.stagedChanges).forEach((key) => {
      if (choices[key] === "mine") {
        const mineVal = state.stagedChanges[key];
        const remoteVal = Object.prototype.hasOwnProperty.call(remoteValues, key)
          ? remoteValues[key]
          : getEffectiveConfigValue(key);
        if (!valuesEqual(mineVal, remoteVal)) {
          nextStaged[key] = mineVal;
        }
      }
    });
    if (!sseAdvanced && remoteConfig && typeof remoteConfig === "object") {
      state.data.config = remoteConfig;
      state.data.config_epoch_id = activeEpochId;
    }
    state.stagedChanges = nextStaged;
    state.draftBaseEpochId = Object.keys(nextStaged).length > 0 ? targetEpochId : null;
    state.rebaseState = null;
    state.previewData = null;
    renderConfigWorkspace(true);
    renderDiffDrawer();
    if (Object.keys(nextStaged).length > 0) {
      scheduleServerPreview();
      showToast(
        sseAdvanced
          ? `Rebased draft onto newer ${targetEpochId} (superseded ${activeEpochId}). Review diff and click Commit.`
          : `Rebased draft onto ${targetEpochId}. Review diff and click Commit.`,
        "success"
      );
    } else {
      showToast(`Accepted remote ${targetEpochId} values.`, "success");
    }
  }

  function loadBuiltinProfileIntoDraft(profileId) {
    const profiles = getBuiltinProfiles();
    const profile = profiles && profiles[profileId];
    if (!profile || !profile.values) return;

    const applyValues = (replaceExisting) => {
      const baseEpoch = state.draftBaseEpochId || getActiveEpochId();
      const nextStaged = replaceExisting ? {} : Object.assign({}, state.stagedChanges);
      Object.keys(profile.values).forEach((k) => {
        const val = profile.values[k];
        const curVal = getEffectiveConfigValue(k);
        if (!valuesEqual(val, curVal)) {
          nextStaged[k] = val;
        } else if (replaceExisting) {
          delete nextStaged[k];
        }
      });
      state.stagedChanges = nextStaged;
      state.draftBaseEpochId = Object.keys(nextStaged).length > 0 ? baseEpoch : null;
      state.drawerCollapsedMobile = false;
      renderConfigWorkspace(true);
      renderDiffDrawer();
      scheduleServerPreview();
      showToast(`Loaded profile '${profile.label || profileId}' into staged draft.`, "success");
    };

    if (Object.keys(state.stagedChanges).length > 0) {
      openModal({
        id: "profile-confirm-modal",
        title: "Unsaved Staged Changes Exist",
        buildBody: (bodyEl) => {
          bodyEl.appendChild(
            el("p", {
              text: `Loading '${profile.label || profileId}' affects ${Object.keys(profile.values).length} parameters. Choose whether to replace or merge with your ${Object.keys(state.stagedChanges).length} staged change(s).`,
            })
          );
        },
        buildFooter: (footEl) => {
          footEl.appendChild(
            el("button", {
              type: "button",
              className: "wb-btn wb-btn--ghost",
              text: "Cancel",
              onClick: closeModal,
            })
          );
          footEl.appendChild(
            el("button", {
              type: "button",
              className: "wb-btn",
              text: "Merge with Draft",
              onClick: () => {
                closeModal();
                applyValues(false);
              },
            })
          );
          footEl.appendChild(
            el("button", {
              type: "button",
              className: "wb-btn wb-btn--primary",
              text: "Replace Draft",
              onClick: () => {
                closeModal();
                applyValues(true);
              },
            })
          );
        },
      });
    } else {
      applyValues(true);
    }
  }

  /* ── Custom Dynamic Parameter Modals (+ Add / Delete) ──────────────────── */

  function openAddParameterModal() {
    if (isReadOnlyMode()) {
      showToast("Cannot add parameters in read-only mode.", "error");
      return;
    }
    const schema = getSchemaObject();
    const metrics = Array.isArray(schema.allowlisted_screener_metrics) && schema.allowlisted_screener_metrics.length > 0
      ? schema.allowlisted_screener_metrics
      : ["adx", "rsi14", "realized_vol", "atr_spike_ratio", "ema20_slope_5d"];
    const ops = Array.isArray(schema.allowed_gate_operators) && schema.allowed_gate_operators.length > 0
      ? schema.allowed_gate_operators
      : ["<", "<=", ">", ">=", "==", "!="];
    const onMissingModes = Array.isArray(schema.allowed_on_missing_modes) && schema.allowed_on_missing_modes.length > 0
      ? schema.allowed_on_missing_modes
      : ["reject", "skip_gate"];
    const currentDynCount = getSchemaParameters().filter((p) => p && p.is_dynamic).length;

    let keyIn, labelIn, typeIn, defaultIn, groupIn, unitIn, minIn, maxIn, stepIn, hintIn;
    let metricIn, opIn, thresholdIn, onMissingIn, errorBox, submitBtn;

    openModal({
      id: "add-param-drawer",
      title: "+ Add Declarative Screener Parameter",
      wide: true,
      buildBody: (bodyEl) => {
        errorBox = el("div", { className: "wb-banner wb-banner--danger", hidden: true });
        if (currentDynCount >= MAX_DYNAMIC_PARAMETERS) {
          errorBox.hidden = false;
          errorBox.textContent = `Maximum limit of ${MAX_DYNAMIC_PARAMETERS} custom parameters reached.`;
        }

        keyIn = el("input", { id: "dyn-key", className: "wb-input wb-input--mono", placeholder: "e.g. MIN_RSI14_FLOOR" });
        labelIn = el("input", { id: "dyn-label", className: "wb-input", placeholder: "e.g. Min RSI(14) Floor" });
        typeIn = el(
          "select",
          { id: "dyn-type", className: "wb-select" },
          el("option", { value: "float", text: "float" }),
          el("option", { value: "int", text: "int" }),
          el("option", { value: "bool", text: "bool" })
        );
        defaultIn = el("input", { id: "dyn-default", className: "wb-input wb-input--mono", value: "35" });
        groupIn = el(
          "select",
          { id: "dyn-group", className: "wb-select" },
          el("option", { value: "screener", text: "screener" }),
          el("option", { value: "strategy", text: "strategy" })
        );
        unitIn = el("input", { id: "dyn-unit", className: "wb-input", placeholder: "e.g. RSI" });
        minIn = el("input", { id: "dyn-min", type: "number", step: "any", className: "wb-input wb-input--mono", placeholder: "optional" });
        maxIn = el("input", { id: "dyn-max", type: "number", step: "any", className: "wb-input wb-input--mono", placeholder: "optional" });
        stepIn = el("input", { id: "dyn-step", type: "number", step: "any", className: "wb-input wb-input--mono", value: "0.5" });
        hintIn = el("input", { id: "dyn-hint", className: "wb-input", placeholder: "Explains gate impact..." });

        metricIn = el(
          "select",
          { id: "dyn-gate-metric", className: "wb-select wb-input--mono" },
          metrics.map((m) => el("option", { value: m, text: m }))
        );
        opIn = el(
          "select",
          { id: "dyn-gate-op", className: "wb-select wb-input--mono" },
          ops.map((op) => el("option", { value: op, text: op }))
        );
        thresholdIn = el("input", { id: "dyn-gate-threshold", type: "number", step: "any", className: "wb-input wb-input--mono", value: "35" });
        onMissingIn = el(
          "select",
          { id: "dyn-gate-missing", className: "wb-select" },
          onMissingModes.map((m) => el("option", { value: m, text: m }))
        );

        bodyEl.appendChild(errorBox);
        bodyEl.appendChild(
          el("p", {
            className: "wb-param-hint",
            style: "margin-bottom:0.75rem",
            text: `Custom parameters (${currentDynCount}/${MAX_DYNAMIC_PARAMETERS}) bind a deterministic screener_metric_gate rule evaluated during screening and strategy checks.`,
          })
        );
        bodyEl.appendChild(
          el(
            "div",
            { className: "wb-group-grid" },
            el("div", { className: "wb-field" }, el("label", { className: "wb-label", htmlFor: "dyn-key", text: "Identifier Key" }), keyIn),
            el("div", { className: "wb-field" }, el("label", { className: "wb-label", htmlFor: "dyn-label", text: "Label" }), labelIn),
            el("div", { className: "wb-field" }, el("label", { className: "wb-label", htmlFor: "dyn-type", text: "Type" }), typeIn),
            el("div", { className: "wb-field" }, el("label", { className: "wb-label", htmlFor: "dyn-default", text: "Default Value" }), defaultIn),
            el("div", { className: "wb-field" }, el("label", { className: "wb-label", htmlFor: "dyn-group", text: "Group" }), groupIn),
            el("div", { className: "wb-field" }, el("label", { className: "wb-label", htmlFor: "dyn-unit", text: "Unit" }), unitIn),
            el("div", { className: "wb-field" }, el("label", { className: "wb-label", htmlFor: "dyn-min", text: "Min" }), minIn),
            el("div", { className: "wb-field" }, el("label", { className: "wb-label", htmlFor: "dyn-max", text: "Max" }), maxIn),
            el("div", { className: "wb-field" }, el("label", { className: "wb-label", htmlFor: "dyn-step", text: "Step" }), stepIn),
            el("div", { className: "wb-field" }, el("label", { className: "wb-label", htmlFor: "dyn-hint", text: "Impact Hint" }), hintIn)
          )
        );
        bodyEl.appendChild(
          el(
            "div",
            { className: "wb-card", style: "margin-top:0.85rem;padding:0.85rem" },
            el("div", { className: "wb-label", style: "margin-bottom:0.5rem", text: "Screener Metric Gate Binding (Required)" }),
            el(
              "div",
              { className: "wb-group-grid" },
              el("div", { className: "wb-field" }, el("label", { className: "wb-label", htmlFor: "dyn-gate-metric", text: "Allowlisted Metric" }), metricIn),
              el("div", { className: "wb-field" }, el("label", { className: "wb-label", htmlFor: "dyn-gate-op", text: "Operator" }), opIn),
              el("div", { className: "wb-field" }, el("label", { className: "wb-label", htmlFor: "dyn-gate-threshold", text: "Numeric Threshold" }), thresholdIn),
              el("div", { className: "wb-field" }, el("label", { className: "wb-label", htmlFor: "dyn-gate-missing", text: "On Missing Metric" }), onMissingIn)
            )
          )
        );
      },
      buildFooter: (footEl) => {
        footEl.appendChild(
          el("button", { type: "button", className: "wb-btn wb-btn--ghost", text: "Cancel", onClick: closeModal })
        );
        submitBtn = el("button", {
          type: "button",
          id: "dyn-submit-btn",
          className: "wb-btn wb-btn--primary",
          text: "Create Parameter",
          onClick: submitDynamicParam,
        });
        footEl.appendChild(submitBtn);
      },
    });

    async function submitDynamicParam() {
      const key = keyIn.value.trim();
      const label = labelIn.value.trim();
      const ptype = typeIn.value;
      const rawDef = defaultIn.value.trim();
      const threshNum = Number(thresholdIn.value);

      if (!DYNAMIC_KEY_RE.test(key)) {
        errorBox.hidden = false;
        errorBox.textContent = "Key must match ^[A-Za-z_][A-Za-z0-9_]{1,63}$.";
        return;
      }
      if (!label || label.length > MAX_DYNAMIC_STRING_LENGTH) {
        errorBox.hidden = false;
        errorBox.textContent = `Label is required (max ${MAX_DYNAMIC_STRING_LENGTH} chars).`;
        return;
      }
      if (!Number.isFinite(threshNum)) {
        errorBox.hidden = false;
        errorBox.textContent = "Gate threshold must be a finite number.";
        return;
      }

      let parsedDefault;
      if (ptype === "bool") {
        parsedDefault = rawDef.toLowerCase() === "true" || rawDef === "1";
      } else if (ptype === "int") {
        parsedDefault = parseInt(rawDef, 10);
        if (!Number.isInteger(parsedDefault)) {
          errorBox.hidden = false;
          errorBox.textContent = "Default value must be an integer.";
          return;
        }
      } else {
        parsedDefault = Number(rawDef);
        if (!Number.isFinite(parsedDefault)) {
          errorBox.hidden = false;
          errorBox.textContent = "Default value must be a finite float.";
          return;
        }
      }

      const spec = {
        key,
        label,
        type: ptype,
        default: parsedDefault,
        group: groupIn.value || "screener",
        unit: unitIn.value.trim(),
        impact_hint: hintIn.value.trim(),
        screener_metric_gate: {
          metric: metricIn.value,
          op: opIn.value,
          threshold: threshNum,
          on_missing: onMissingIn.value || "reject",
        },
      };
      if (minIn.value.trim() !== "" && Number.isFinite(Number(minIn.value))) spec.min = Number(minIn.value);
      if (maxIn.value.trim() !== "" && Number.isFinite(Number(maxIn.value))) spec.max = Number(maxIn.value);
      if (stepIn.value.trim() !== "" && Number(stepIn.value) > 0) spec.step = Number(stepIn.value);

      await executeCreate(spec);
    }

    async function executeCreate(spec) {
      if (submitBtn) submitBtn.disabled = true;
      const res = await apiRequest("/api/params", {
        method: "POST",
        body: { spec, expected_epoch_id: getActiveEpochId() },
        retryFactory: () => executeCreate(spec),
      });
      if (submitBtn) submitBtn.disabled = false;
      if (res.ok) {
        applyMutationSnapshotResponse(res.data);
        closeModal();
        renderConfigWorkspace(true);
        showToast(`Created dynamic parameter '${spec.key}'.`, "success");
      } else if (res.status !== 401) {
        errorBox.hidden = false;
        errorBox.textContent = (res.data && res.data.error) || `Create failed (${res.status})`;
      }
    }
  }

  function openDeleteParamModal(key) {
    if (isReadOnlyMode()) {
      showToast("Cannot delete parameters in read-only mode.", "error");
      return;
    }
    let confirmBtn = null;
    openModal({
      id: "delete-param-modal",
      title: `Delete Custom Parameter: ${key}`,
      buildBody: (bodyEl) => {
        bodyEl.appendChild(
          el("p", {
            text: `Deleting '${key}' commits immediately, prunes any stored override for '${key}', and preserves any unrelated stagedChanges in your draft.`,
          })
        );
      },
      buildFooter: (footEl) => {
        footEl.appendChild(
          el("button", { type: "button", className: "wb-btn wb-btn--ghost", text: "Cancel", onClick: closeModal })
        );
        confirmBtn = el("button", {
          type: "button",
          id: "delete-param-confirm-btn",
          className: "wb-btn wb-btn--danger",
          text: "Delete Parameter",
          onClick: executeDelete,
        });
        footEl.appendChild(confirmBtn);
      },
    });

    async function executeDelete() {
      if (confirmBtn) confirmBtn.disabled = true;
      const res = await apiRequest(`/api/params/${encodeURIComponent(key)}/delete`, {
        method: "POST",
        body: { expected_epoch_id: getActiveEpochId() },
        retryFactory: () => executeDelete(),
      });
      if (confirmBtn) confirmBtn.disabled = false;
      if (res.ok) {
        delete state.stagedChanges[key];
        if (Object.keys(state.stagedChanges).length === 0) {
          state.draftBaseEpochId = null;
        }
        applyMutationSnapshotResponse(res.data);
        closeModal();
        renderConfigWorkspace(true);
        renderDiffDrawer();
        showToast(`Deleted custom parameter '${key}'.`, "success");
      } else if (res.status !== 401) {
        closeModal();
        showToast((res.data && res.data.error) || `Delete failed (${res.status})`, "error");
      }
    }
  }

  /* ── Panic Flatten, Resume & Re-Epoch Modals ───────────────────────────── */

  function openPanicModal(titleOverride) {
    if (isReadOnlyMode()) {
      showToast("Panic Flatten is unavailable in read-only mode.", "error");
      return;
    }
    let confirmBtn = null;
    openModal({
      id: "panic-modal",
      title: titleOverride || "Confirm Emergency Panic Flatten",
      buildBody: (bodyEl) => {
        const count = (state.data.active_signals || []).length;
        bodyEl.appendChild(
          el("div", {
            className: "wb-banner wb-banner--danger",
            text: `This immediately halts automated trading (.halted), cancels active broker stop orders, and submits market liquidations for ${count} active position(s).`,
          })
        );
        bodyEl.appendChild(
          el("p", {
            className: "wb-param-hint",
            text: "Any failed liquidation remains visible as UNPROTECTED / PANIC FAILED until liquidated or re-armed.",
          })
        );
      },
      buildFooter: (footEl) => {
        footEl.appendChild(
          el("button", { type: "button", className: "wb-btn wb-btn--ghost", text: "Cancel", onClick: closeModal })
        );
        confirmBtn = el("button", {
          type: "button",
          id: "panic-confirm-btn",
          className: "wb-btn wb-btn--danger",
          text: "Execute Panic Flatten",
          onClick: executePanic,
        });
        footEl.appendChild(confirmBtn);
      },
    });

    async function executePanic() {
      if (confirmBtn) confirmBtn.disabled = true;
      const res = await apiRequest("/api/panic", {
        method: "POST",
        body: { confirm: true },
        retryFactory: () => executePanic(),
      });
      if (confirmBtn) confirmBtn.disabled = false;
      if (res.ok) {
        closeModal();
        applyMutationSnapshotResponse(res.data);
        await fetchCatchUpData();
        showToast("Panic Flatten executed.", "success");
      } else if (res.status !== 401) {
        closeModal();
        showToast((res.data && res.data.error) || `Panic Flatten failed (${res.status})`, "error");
      }
    }
  }

  function getUnprotectedPositions() {
    const signals = state.data.active_signals || [];
    return signals.filter((s) =>
      Boolean(
        s &&
          (s.panic_failed ||
            s.unprotected_reason ||
            s.status === "PANIC_FAILED" ||
            s.status === "UNPROTECTED" ||
            !s.broker_stop_order_id)
      )
    );
  }

  function openResumeModal() {
    if (isReadOnlyMode() || !isSystemHalted()) return;
    const unprotected = getUnprotectedPositions();
    let confirmCheck = null;
    let confirmBtn = null;

    openModal({
      id: "resume-modal",
      title: "Confirm Trading Resume",
      buildBody: (bodyEl) => {
        if (unprotected.length > 0) {
          const tickers = unprotected.map((p) => p.ticker).join(", ");
          bodyEl.appendChild(
            el(
              "div",
              { className: "wb-banner wb-banner--danger" },
              el("strong", { text: `Warning: ${unprotected.length} unprotected / panic-failed position(s): ` }),
              el("span", { className: "wb-mono", text: tickers }),
              el("p", {
                style: "margin:0.35rem 0 0",
                text: "Backend resume verifies and re-arms GTC stops; confirm explicit acknowledgment if unprotected positions are present.",
              })
            )
          );
          confirmCheck = el("input", {
            id: "resume-confirm-unprotected",
            type: "checkbox",
            onChange: () => {
              if (confirmBtn) confirmBtn.disabled = !confirmCheck.checked;
            },
          });
          bodyEl.appendChild(
            el(
              "label",
              { className: "wb-check-row", htmlFor: "resume-confirm-unprotected" },
              confirmCheck,
              "I acknowledge unprotected / panic-failed positions and request GTC stop verification before resuming."
            )
          );
        } else {
          bodyEl.appendChild(
            el("p", {
              text: "Resume automated position monitoring and universe scanning? All open positions will be verified for active broker GTC stop protection.",
            })
          );
        }
      },
      buildFooter: (footEl) => {
        footEl.appendChild(
          el("button", { type: "button", className: "wb-btn wb-btn--ghost", text: "Cancel", onClick: closeModal })
        );
        confirmBtn = el("button", {
          type: "button",
          id: "resume-confirm-btn",
          className: "wb-btn wb-btn--emerald",
          disabled: unprotected.length > 0,
          text: "Confirm Resume",
          onClick: executeResume,
        });
        footEl.appendChild(confirmBtn);
      },
    });

    async function executeResume() {
      const confirmUnprotected = Boolean(confirmCheck && confirmCheck.checked);
      if (confirmBtn) confirmBtn.disabled = true;
      const res = await apiRequest("/api/resume", {
        method: "POST",
        body: { confirm_unprotected: confirmUnprotected },
        retryFactory: () => executeResume(),
      });
      if (confirmBtn) confirmBtn.disabled = false;
      if (res.ok) {
        closeModal();
        applyMutationSnapshotResponse(res.data);
        await fetchCatchUpData();
        showToast("Automated trading resumed.", "success");
      } else if (res.status === 409) {
        if (res.data) {
          applyMutationSnapshotResponse(res.data);
        }
        closeModal();
        showToast((res.data && res.data.error) || "Resume blocked: unprotected positions require confirmation.", "error");
        openResumeModal();
      } else if (res.status !== 401 && res.status !== 403) {
        closeModal();
        showToast((res.data && res.data.error) || `Resume failed (${res.status})`, "error");
      }
    }
  }

  function openRetryLiquidationModal(pos) {
    if (!pos || !pos.ticker) return;
    if (isReadOnlyMode()) {
      showToast("Retry Liquidation is disabled in read-only mode.", "error");
      return;
    }
    let confirmBtn = null;
    openModal({
      id: "retry-liquidation-modal",
      title: `Retry Liquidation (${pos.ticker})`,
      buildBody: (bodyEl) => {
        bodyEl.appendChild(
          el("div", {
            className: "wb-banner wb-banner--danger",
            text: `Submits a targeted emergency market sell for ${pos.ticker} via /api/positions/${pos.ticker}/retry-liquidation without affecting other open positions.`,
          })
        );
      },
      buildFooter: (footEl) => {
        footEl.appendChild(
          el("button", { type: "button", className: "wb-btn wb-btn--ghost", text: "Cancel", onClick: closeModal })
        );
        confirmBtn = el("button", {
          type: "button",
          id: "retry-liquidation-confirm-btn",
          className: "wb-btn wb-btn--danger",
          text: `Retry Liquidation (${pos.ticker})`,
          onClick: executeRetry,
        });
        footEl.appendChild(confirmBtn);
      },
    });

    async function executeRetry() {
      if (confirmBtn) confirmBtn.disabled = true;
      const res = await apiRequest(`/api/positions/${encodeURIComponent(pos.ticker)}/retry-liquidation`, {
        method: "POST",
        body: { confirm: true },
        retryFactory: () => executeRetry(),
      });
      if (confirmBtn) confirmBtn.disabled = false;
      if (res.ok && res.data && res.data.ok !== false && !res.data.panic_failed) {
        closeModal();
        applyMutationSnapshotResponse(res.data);
        await fetchCatchUpData();
        showToast(`Retry liquidation succeeded for ${pos.ticker}.`, "success");
      } else if (res.status !== 401 && res.status !== 403) {
        if (res.data) applyMutationSnapshotResponse(res.data);
        closeModal();
        showToast(
          (res.data && (res.data.error || res.data.message)) || `Retry liquidation failed for ${pos.ticker} (${res.status})`,
          "error"
        );
      }
    }
  }

  async function openReEpochModal(posOrTicker) {
    if (isReadOnlyMode()) {
      showToast("Re-Epoch is disabled in read-only mode.", "error");
      return;
    }
    if (state.reEpochModalLoading) return;
    const pos = typeof posOrTicker === "string"
      ? (state.data.active_signals || []).find((p) => p && p.ticker === posOrTicker) || { ticker: posOrTicker }
      : posOrTicker;
    if (!pos || !pos.ticker) return;
    const targetEpoch = getActiveEpochId();
    const expectedPosEpoch = pos.config_epoch_id || null;
    state.reEpochModalLoading = true;
    let previewRes = null;
    try {
      previewRes = await apiRequest(`/api/positions/${encodeURIComponent(pos.ticker)}/re-epoch/preview`, {
        method: "POST",
        body: {
          expected_position_epoch_id: expectedPosEpoch,
          target_epoch_id: targetEpoch,
          confirm_pure_tsl_mode_change: false,
        },
        retryFactory: () => openReEpochModal(pos),
      });
    } finally {
      state.reEpochModalLoading = false;
    }

    if (!previewRes || !previewRes.ok) {
      if (previewRes && previewRes.status !== 401 && previewRes.status !== 403) {
        showToast((previewRes.data && previewRes.data.error) || `Re-Epoch preview failed (${previewRes.status})`, "error");
      }
      return;
    }

    const preview = previewRes.data || {};
    const action = preview.action || "no_change";
    const needsTslConfirm = Boolean(
      preview.pure_tsl_mode_changed || preview.requires_tsl_confirmation || preview.requires_confirmation
    );
    const blockedByMarket = action === "market_closed" || action === "blocked";
    let tslCheck = null;
    let confirmBtn = null;

    openModal({
      id: "re-epoch-modal",
      title: `Re-Epoch ${pos.ticker} -> ${targetEpoch}`,
      buildBody: (bodyEl) => {
        const badgeCls = action === "tighten"
          ? "wb-badge wb-badge--emerald"
          : (blockedByMarket ? "wb-badge wb-badge--rose" : "wb-badge wb-badge--sky");
        const oldSl = preview.old_stop_loss ?? preview.current_stop ?? preview.current_sl ?? pos.stop_loss;
        const newSl = preview.new_stop_loss ?? preview.recommended_sl ?? preview.candidate_sl ?? oldSl;

        bodyEl.appendChild(
          el(
            "div",
            { className: "wb-pos-rows", style: "margin-bottom:0.75rem" },
            el("div", { className: "wb-pos-row" }, el("span", { text: "Position Epoch" }), el("span", { className: "wb-mono", text: `${expectedPosEpoch || "Legacy (Unstamped)"} -> ${targetEpoch}` })),
            el("div", { className: "wb-pos-row" }, el("span", { text: "Stop Policy Action" }), el("span", { className: badgeCls, text: formatReEpochActionLabel(action) })),
            el("div", { className: "wb-pos-row" }, el("span", { text: "GTC Stop Projection" }), el("span", { className: "wb-mono", text: `${oldSl ?? "—"} -> ${newSl ?? "—"}` }))
          )
        );
        if (preview.reason || preview.message || preview.warning) {
          bodyEl.appendChild(
            el("div", {
              className: blockedByMarket ? "wb-banner wb-banner--danger" : "wb-banner wb-banner--info",
              text: preview.reason || preview.message || preview.warning,
            })
          );
        }
        if (needsTslConfirm) {
          tslCheck = el("input", {
            id: "re-epoch-tsl-confirm",
            type: "checkbox",
            onChange: () => {
              if (confirmBtn && !blockedByMarket) confirmBtn.disabled = !tslCheck.checked;
            },
          });
          bodyEl.appendChild(
            el(
              "div",
              { className: "wb-banner wb-banner--warn" },
              el("strong", { text: "PURE_TSL_MODE Transition Warning: " }),
              el("span", { text: "Migrating this position alters partial TP vs 100% Pure TSL broker order synchronization." }),
              el("div", { style: "margin-top:0.5rem;width:100%" }, el("label", { className: "wb-check-row", htmlFor: "re-epoch-tsl-confirm" }, tslCheck, "Confirm PURE_TSL_MODE transition"))
            )
          );
        }
      },
      buildFooter: (footEl) => {
        footEl.appendChild(
          el("button", { type: "button", className: "wb-btn wb-btn--ghost", text: "Cancel", onClick: closeModal })
        );
        confirmBtn = el("button", {
          type: "button",
          id: "re-epoch-confirm-btn",
          className: "wb-btn wb-btn--primary",
          disabled: blockedByMarket || needsTslConfirm,
          text: "Confirm Re-Epoch",
          onClick: executeSingleReEpoch,
        });
        footEl.appendChild(confirmBtn);
      },
    });

    async function executeSingleReEpoch() {
      if (confirmBtn) confirmBtn.disabled = true;
      const res = await apiRequest(`/api/positions/${encodeURIComponent(pos.ticker)}/re-epoch`, {
        method: "POST",
        body: {
          expected_position_epoch_id: expectedPosEpoch,
          target_epoch_id: targetEpoch,
          confirm_pure_tsl_mode_change: Boolean(tslCheck && tslCheck.checked),
        },
        retryFactory: () => executeSingleReEpoch(),
      });
      if (confirmBtn) confirmBtn.disabled = false;
      if (res.ok) {
        closeModal();
        applyMutationSnapshotResponse(res.data);
        await fetchCatchUpData();
        showToast(`${pos.ticker} migrated to ${targetEpoch}.`, "success");
      } else if (res.status !== 401) {
        closeModal();
        showToast((res.data && res.data.error) || `Re-Epoch failed (${res.status})`, "error");
      }
    }
  }

  async function openReEpochAllModal() {
    if (isReadOnlyMode()) {
      showToast("Re-Epoch All is disabled in read-only mode.", "error");
      return;
    }
    if (state.reEpochModalLoading) return;
    state.reEpochModalLoading = true;
    const targetEpoch = getActiveEpochId();
    let previewRes = null;
    try {
      previewRes = await apiRequest("/api/positions/re-epoch-all/preview", {
        method: "POST",
        body: { target_epoch_id: targetEpoch, confirm_pure_tsl_mode_change: false },
        retryFactory: () => openReEpochAllModal(),
      });
    } finally {
      state.reEpochModalLoading = false;
    }
    if (!previewRes || !previewRes.ok) {
      if (previewRes && previewRes.status !== 401 && previewRes.status !== 403) {
        showToast((previewRes.data && previewRes.data.error) || `Batch preview failed (${previewRes.status})`, "error");
      }
      return;
    }

    const preview = previewRes.data || {};
    const items = preview.positions || preview.previews || preview.items || [];
    const needsTslConfirm = Boolean(
      preview.pure_tsl_mode_changed || preview.requires_tsl_confirmation || preview.requires_confirmation
    );
    let tslCheck = null;
    let confirmBtn = null;
    let modalBody = null;
    let modalFoot = null;

    openModal({
      id: "re-epoch-all-modal",
      title: `Batch Re-Epoch All -> ${targetEpoch}`,
      wide: true,
      buildBody: (bodyEl) => {
        modalBody = bodyEl;
        if (items.length === 0) {
          bodyEl.appendChild(el("div", { className: "wb-empty", text: "All open positions are already on the active epoch." }));
        } else {
          const list = el("div", { className: "wb-pos-rows" });
          items.forEach((it) => {
            const act = it.action || "no_change";
            list.appendChild(
              el(
                "div",
                { className: "wb-pos-row" },
                el("span", { className: "wb-mono", text: it.ticker || "—" }),
                el("span", { className: "wb-badge wb-badge--sky", text: formatReEpochActionLabel(act) }),
                el("span", { className: "wb-muted", text: it.reason || it.message || "" })
              )
            );
          });
          bodyEl.appendChild(list);
        }
        if (needsTslConfirm) {
          tslCheck = el("input", {
            id: "re-epoch-all-tsl-confirm",
            type: "checkbox",
            onChange: () => {
              if (confirmBtn) confirmBtn.disabled = !tslCheck.checked;
            },
          });
          bodyEl.appendChild(
            el(
              "div",
              { className: "wb-banner wb-banner--warn", style: "margin-top:0.75rem" },
              el("label", { className: "wb-check-row", htmlFor: "re-epoch-all-tsl-confirm" }, tslCheck, "Confirm PURE_TSL_MODE transition across affected positions")
            )
          );
        }
      },
      buildFooter: (footEl) => {
        modalFoot = footEl;
        footEl.appendChild(
          el("button", { type: "button", className: "wb-btn wb-btn--ghost", text: "Cancel", onClick: closeModal })
        );
        confirmBtn = el("button", {
          type: "button",
          id: "re-epoch-all-confirm-btn",
          className: "wb-btn wb-btn--primary",
          disabled: items.length === 0 || needsTslConfirm,
          text: "Execute Re-Epoch All",
          onClick: executeBatch,
        });
        footEl.appendChild(confirmBtn);
      },
    });

    async function executeBatch() {
      if (confirmBtn) confirmBtn.disabled = true;
      const res = await apiRequest("/api/positions/re-epoch-all", {
        method: "POST",
        body: {
          target_epoch_id: targetEpoch,
          confirm_pure_tsl_mode_change: Boolean(tslCheck && tslCheck.checked),
        },
        retryFactory: () => executeBatch(),
      });
      if (confirmBtn) confirmBtn.disabled = false;
      if (res.ok) {
        applyMutationSnapshotResponse(res.data);
        await fetchCatchUpData();
        const summary = res.data || {};
        if (modalBody && modalFoot) {
          clearNode(modalBody);
          const results = summary.results || summary.migrated || [];
          modalBody.appendChild(
            el("div", {
              className: summary.failed_count > 0 || summary.halted_preempted ? "wb-banner wb-banner--warn" : "wb-banner wb-banner--info",
              text: `Batch complete. Succeeded: ${summary.succeeded_count ?? summary.updated_count ?? results.length}, Failed/Skipped: ${summary.failed_count ?? 0}${summary.halted_preempted ? " (preempted by system halt)" : ""}.`,
            })
          );
          clearNode(modalFoot);
          modalFoot.appendChild(
            el("button", { type: "button", className: "wb-btn wb-btn--primary", text: "Done", onClick: closeModal })
          );
        } else {
          closeModal();
        }
      } else if (res.status !== 401) {
        closeModal();
        showToast((res.data && res.data.error) || `Batch Re-Epoch failed (${res.status})`, "error");
      }
    }
  }

  function formatReEpochActionLabel(action) {
    if (action === "tighten") return "Tighten Stop";
    if (action === "keep_existing_stop") return "Keep Existing Stop (Never Loosen)";
    if (action === "market_closed") return "Market Closed (Blocked)";
    if (action === "blocked") return "Blocked";
    return "Policy Align (No Stop Change)";
  }

  /* ── Snapshot Application & Focus-Safe DOM Reconciliation ──────────────── */

  function applyMutationSnapshotResponse(payload) {
    if (!payload || typeof payload !== "object") return;
    if (payload.dashboard && typeof payload.dashboard === "object") {
      applySnapshotData(payload.dashboard);
      return;
    }
    if (payload.snapshot && typeof payload.snapshot === "object") {
      state.data.config = payload.snapshot;
      if (payload.snapshot.config_epoch_id) {
        state.data.config_epoch_id = payload.snapshot.config_epoch_id;
      }
      if (payload.snapshot.pending_restart_keys) {
        state.data.pending_restart_keys = payload.snapshot.pending_restart_keys;
      }
      if (payload.snapshot.config_quarantine_warnings) {
        state.data.config_quarantine_warnings = payload.snapshot.config_quarantine_warnings;
      }
      if (payload.schema) state.data.schema = payload.schema;
    } else if (payload.config && typeof payload.config === "object") {
      state.data.config = payload.config;
      if (payload.config.config_epoch_id) state.data.config_epoch_id = payload.config.config_epoch_id;
    }
    if (payload.config_epoch_id) state.data.config_epoch_id = payload.config_epoch_id;
    renderAllVisible();
  }

  function applySnapshotData(incoming) {
    if (!incoming || typeof incoming !== "object") return;
    const prevEpoch = getActiveEpochId();
    // Merge top-level fields without destroying preserved static fields if omitted in partial SSE events
    Object.keys(incoming).forEach((k) => {
      if (incoming[k] !== undefined) {
        state.data[k] = incoming[k];
      }
    });
    const nextEpoch = getActiveEpochId();

    // Update equity chart whenever equity payload arrives
    if (incoming.equity_history || incoming.current_equity !== undefined) {
      renderEquityChart(state.data.equity_history, state.data.current_equity);
    }

    // Reconcile visible workspaces cleanly WITHOUT wiping open modals, focused inputs, or stagedChanges
    renderBanners();
    renderHeader();

    if (state.route.tab === "monitor") {
      renderMonitorWorkspace();
    } else if (state.route.tab === "config") {
      renderConfigWorkspace(false);
      if (nextEpoch !== prevEpoch && Object.keys(state.stagedChanges).length > 0) {
        renderDiffDrawer();
        scheduleServerPreview();
      } else if (nextEpoch !== prevEpoch) {
        renderDiffDrawer();
      }
    }
  }

  /* ── Render: Banners & Header ──────────────────────────────────────────── */

  function renderBanners() {
    const doc = getDoc();
    if (!doc) return;

    const roEl = doc.getElementById("readonly-banner");
    if (roEl) {
      clearNode(roEl);
      if (isFileProtocol()) {
        roEl.hidden = false;
        roEl.className = "wb-banner wb-banner--warn";
        roEl.appendChild(
          el(
            "div",
            { className: "wb-banner-body" },
            el("span", { className: "wb-banner-title", text: "Read-Only Offline Viewer (file://)" }),
            "Viewing static embedded snapshot. Network streaming, live refresh, and configuration or panic mutations are disabled."
          )
        );
      } else if (!isControllerAttached()) {
        roEl.hidden = false;
        roEl.className = "wb-banner wb-banner--warn";
        roEl.appendChild(
          el(
            "div",
            { className: "wb-banner-body" },
            el("span", { className: "wb-banner-title", text: "Detached Read-Only Server" }),
            state.data.read_only_reason || "Standalone dashboard process has no attached Orchestrator controller (HTTP 503 on mutations)."
          )
        );
      } else {
        roEl.hidden = true;
      }
    }

    const qEl = doc.getElementById("quarantine-banner");
    if (qEl) {
      clearNode(qEl);
      const warnings = getQuarantineWarnings();
      if (warnings.length > 0) {
        qEl.hidden = false;
        qEl.className = "wb-banner wb-banner--danger";
        qEl.appendChild(
          el(
            "div",
            { className: "wb-banner-body" },
            el("span", { className: "wb-banner-title", text: "Config Quarantine Warnings" }),
            el("ul", { className: "wb-banner-list" }, warnings.map((w) => el("li", { text: w })))
          )
        );
      } else {
        qEl.hidden = true;
      }
    }

    const hEl = doc.getElementById("halted-banner");
    if (hEl) {
      clearNode(hEl);
      if (isSystemHalted()) {
        hEl.hidden = false;
        hEl.className = "wb-banner wb-banner--danger";
        const unprotectedCount = getUnprotectedPositions().length;
        const msg = state.data.halt_reason || "Automated entries and exit monitoring are halted (.halted active).";
        hEl.appendChild(
          el(
            "div",
            { className: "wb-banner-body" },
            el("span", { className: "wb-banner-title", text: "SYSTEM HALTED" }),
            el("span", {
              text: unprotectedCount > 0
                ? `${msg} ${unprotectedCount} unprotected/panic-failed position(s) require verification.`
                : msg,
            })
          )
        );
        if (!isReadOnlyMode()) {
          hEl.appendChild(
            el("button", {
              type: "button",
              className: "wb-btn wb-btn--emerald wb-btn--sm",
              text: "Resume Trading...",
              onClick: openResumeModal,
            })
          );
        }
      } else {
        hEl.hidden = true;
      }
    }
  }

  function renderHeader() {
    const doc = getDoc();
    const header = doc && doc.getElementById("workbench-header");
    if (!header) return;
    clearNode(header);

    const d = state.data || {};
    const broker = d.broker_status || {};
    const telemetry = d.rate_limit_telemetry || broker.rate_limit_telemetry || {};
    const epochId = getActiveEpochId();
    const pendingRestart = getPendingRestartKeys();
    const readOnly = isReadOnlyMode();
    const halted = isSystemHalted();
    const unlocked = isUnlocked();

    const pairsCount = broker.monitored_assets ?? (Array.isArray(d.pairs_watchlist) ? d.pairs_watchlist.length : 0);
    const eurRatio = broker.eur_ratio || computeEurRatio(d.pairs_watchlist);

    const pills = el(
      "div",
      { className: "wb-pill-row" },
      el(
        "span",
        { className: halted ? "wb-pill wb-pill--rose" : "wb-pill wb-pill--emerald" },
        el("span", { className: halted ? "wb-dot" : "wb-dot wb-dot--pulse" }),
        halted ? "HALTED" : (broker.runner_status || "ONLINE")
      ),
      el(
        "span",
        { className: "wb-pill wb-pill--indigo" },
        `${broker.broker_name || "Broker"}: ${broker.is_connected === false ? "Disconnected" : "Connected"}`
      ),
      el(
        "span",
        {
          id: "header-epoch-pill",
          className: "wb-pill wb-pill--sky wb-mono",
          hidden: state.viewMode === "simple",
        },
        `Epoch: ${epochId}`
      ),
      el(
        "button",
        {
          type: "button",
          id: "header-view-mode-btn",
          className: state.viewMode === "simple" ? "wb-pill wb-pill--emerald" : "wb-pill wb-pill--sky",
          style: "cursor: pointer; border: none;",
          title: "Toggle Simple / Advanced Control Mode",
          onClick: () => {
            state.viewMode = state.viewMode === "simple" ? "advanced" : "simple";
            try {
              const win = getWin();
              if (win && win.sessionStorage) {
                win.sessionStorage.setItem("quant_workbench_view_mode", state.viewMode);
              }
            } catch (_) {}
            renderAllVisible();
          },
        },
        state.viewMode === "simple" ? "Simple Mode" : "Advanced Mode"
      ),
      el(
        "span",
        { className: "wb-pill wb-mono" },
        `${pairsCount} Pairs${eurRatio ? ` • ${eurRatio} EUR` : ""}`
      )
    );

    if (pendingRestart.length > 0) {
      pills.appendChild(
        el("span", {
          id: "header-pending-restart-pill",
          className: "wb-pill wb-pill--amber",
          title: `Restart required for: ${pendingRestart.join(", ")}`,
          text: `Pending Restart (${pendingRestart.length}): ${pendingRestart.join(", ")}`,
        })
      );
    }

    const waitCount = Number(telemetry.wait_count || 0);
    const rate429 = Number(telemetry.rate_limit_429_count || 0);
    const globalCooldown = Number(telemetry.global_cooldown_remaining || 0);

    const telemetryStrip = el(
      "div",
      { id: "header-rate-telemetry", className: "wb-telemetry-strip" },
      el("span", {
        className: rate429 > 0 ? "wb-badge wb-badge--rose wb-mono" : "wb-badge wb-mono",
        text: `Rate-Lock: waits ${waitCount} | 429s ${rate429} | cooldown ${globalCooldown.toFixed(1)}s`,
      }),
      el("span", { id: "header-sse-health", className: "wb-badge wb-mono", text: formatSseHealthText() }),
      el("span", {
        id: "timestamp-text",
        className: "wb-muted wb-mono",
        text: d.generated_at ? `Snapshot: ${d.generated_at}` : "Snapshot: —",
      })
    );

    const authBtn = el("button", {
      type: "button",
      id: "auth-status-btn",
      className: unlocked ? "wb-btn wb-btn--sm wb-btn--emerald" : "wb-btn wb-btn--sm",
      disabled: readOnly,
      text: unlocked ? "Unlocked" : "Locked (Read-Only)",
      onClick: () => {
        if (!unlocked) openAuthModal();
      },
    });

    const lockBtn = unlocked
      ? el("button", {
          type: "button",
          id: "auth-lock-btn",
          className: "wb-btn wb-btn--sm wb-btn--ghost",
          text: "Lock",
          onClick: () => {
            clearStoredToken();
            state.previewData = null;
            renderHeader();
            renderDiffDrawer();
            showToast("Session token cleared (Locked).", "info");
          },
        })
      : null;

    const refreshBtn = el("button", {
      type: "button",
      id: "refresh-btn",
      className: "wb-btn wb-btn--sm",
      disabled: isFileProtocol(),
      text: "Refresh Now",
      onClick: () => fetchCatchUpData(true),
    });

    const panicBtn = el("button", {
      type: "button",
      id: "panic-flatten-btn",
      className: "wb-btn wb-btn--sm wb-btn--danger",
      disabled: readOnly,
      text: "Panic Flatten",
      onClick: () => openPanicModal(),
    });

    const resumeBtn = el("button", {
      type: "button",
      id: "resume-trading-btn",
      className: "wb-btn wb-btn--sm wb-btn--emerald",
      disabled: readOnly || !halted,
      text: "Resume",
      onClick: openResumeModal,
    });

    header.appendChild(
      el(
        "div",
        { className: "wb-header-left" },
        pills,
        el("h1", { className: "wb-title", text: "Trading 212 Quantitative Control Workbench" }),
        el("p", {
          className: "wb-subtitle",
          text: broker.execution_mode || "Epoch-versioned directional trend engine with native server GTC stops.",
        })
      )
    );
    header.appendChild(
      el(
        "div",
        { className: "wb-header-right" },
        el("div", { className: "wb-action-row" }, authBtn, lockBtn, refreshBtn, panicBtn, resumeBtn),
        telemetryStrip
      )
    );
  }

  function computeEurRatio(pairs) {
    if (!Array.isArray(pairs) || pairs.length === 0) return "";
    let eur = 0;
    pairs.forEach((p) => {
      if ((p.currency || p.long_currency || "").toUpperCase() === "EUR") eur += 1;
    });
    return `${Math.round((eur / pairs.length) * 100)}%`;
  }

  function formatSseHealthText() {
    if (isFileProtocol()) return "SSE: Offline (file://)";
    const now = Date.now();
    const snapAge = Math.max(0, Math.floor((now - state.sse.lastSnapshotAt) / 1000));
    const hbAge = Math.max(0, Math.floor((now - state.sse.lastHeartbeatAt) / 1000));
    if (hbAge > HEARTBEAT_STALE_MS / 1000) {
      return `Last Snapshot ${snapAge}s | Heartbeat Stale (>25s)`;
    }
    return `Last Snapshot ${snapAge}s | SSE ${state.sse.status === "connected" ? "Live" : state.sse.status}`;
  }

  function updateSseHealthBadge() {
    const doc = getDoc();
    const badge = doc && doc.getElementById("header-sse-health");
    const now = Date.now();
    const isStale = !isFileProtocol() && (now - state.sse.lastHeartbeatAt) > HEARTBEAT_STALE_MS;
    if (badge) {
      badge.className = isStale ? "wb-badge wb-badge--rose wb-mono" : "wb-badge wb-mono";
      badge.textContent = formatSseHealthText();
    }
    if (!isFileProtocol() && state.sse.status === "connected" && (now - state.sse.lastHeartbeatAt) > HEARTBEAT_STALE_MS * 2) {
      state.sse.hasDisconnected = true;
      state.sse.status = "reconnecting";
      connectSSE();
    }
  }

  /* ── Render: Equity Chart (Preserved Compatibility) ────────────────────── */

  function renderEquityChart(history, currentEquity) {
    const doc = getDoc();
    if (!doc) return;
    const g = doc.getElementById("equity-chart-group");
    const hasHist = Array.isArray(history) && history.length > 0;
    const hasCurr = currentEquity !== undefined && currentEquity !== null && Number.isFinite(Number(currentEquity));

    let points = [];
    if (hasHist) {
      points = history;
    } else if (hasCurr) {
      const val = Number(currentEquity);
      points = [{ time: "Start", equity: val }, { time: "Now", equity: val }];
    }

    const curEl = doc.getElementById("equity-val-current");
    const deltaEl = doc.getElementById("equity-val-delta");
    const startEl = doc.getElementById("equity-val-start");
    const highEl = doc.getElementById("equity-val-high");
    const lowEl = doc.getElementById("equity-val-low");

    if (points.length === 0) {
      if (curEl) curEl.textContent = "—";
      if (deltaEl) deltaEl.textContent = "—";
      if (startEl) startEl.textContent = "—";
      if (highEl) highEl.textContent = "—";
      if (lowEl) lowEl.textContent = "—";
      if (g) clearNode(g);
      return;
    }

    const equities = points.map((p) => Number(p.equity || 0));
    const startEq = equities[0] || 0;
    const currEq = hasCurr ? Number(currentEquity) : (equities[equities.length - 1] || startEq);
    const minEq = equities.reduce((m, v) => (v < m ? v : m), currEq);
    const maxEq = equities.reduce((m, v) => (v > m ? v : m), currEq);
    const delta = currEq - startEq;
    const deltaPct = startEq > 0 ? (delta / startEq) * 100 : 0;
    const isPos = delta >= 0;

    if (curEl) curEl.textContent = `€${formatNumber(currEq, 2)}`;
    if (deltaEl) {
      deltaEl.textContent = `${isPos ? "+" : "-"}€${Math.abs(delta).toFixed(2)} (${isPos ? "+" : ""}${deltaPct.toFixed(2)}%)`;
      deltaEl.className = `wb-equity-delta ${isPos ? "wb-pill--emerald" : "wb-pill--rose"}`;
    }
    if (startEl) startEl.textContent = `€${formatNumber(startEq, 2)}`;
    if (highEl) highEl.textContent = `€${formatNumber(maxEq, 2)}`;
    if (lowEl) lowEl.textContent = `€${formatNumber(minEq, 2)}`;

    if (!g) return;
    clearNode(g);

    const W = 800;
    const H = 100;
    const padY = 12;
    const range = (maxEq - minEq) || 1.0;

    const coords = points.map((p, idx) => {
      const x = points.length === 1 ? W / 2 : (idx / (points.length - 1)) * W;
      const norm = (Number(p.equity || startEq) - minEq) / range;
      const y = H - padY - (norm * (H - 2 * padY));
      return { x: Number(x.toFixed(1)), y: Number(y.toFixed(1)), equity: Number(p.equity || 0), time: p.time || "" };
    });

    const lineD = coords.reduce((acc, pt, i) => acc + (i === 0 ? `M ${pt.x} ${pt.y}` : ` L ${pt.x} ${pt.y}`), "");
    const areaD = `${lineD} L ${W} ${H} L 0 ${H} Z`;
    const stroke = isPos ? "#10b981" : "#f43f5e";
    const grad = isPos ? "url(#equity-gradient-emerald)" : "url(#equity-gradient-rose)";

    g.appendChild(svgEl("line", { x1: 0, y1: H, x2: W, y2: H, stroke: "#334155", "stroke-width": 1, "stroke-dasharray": 4 }));
    g.appendChild(svgEl("path", { d: areaD, fill: grad }));
    g.appendChild(svgEl("path", { d: lineD, fill: "none", stroke, "stroke-width": 2.5, "stroke-linecap": "round", "stroke-linejoin": "round" }));

    coords.forEach((pt) => {
      const titleNode = svgEl("title", { text: `${pt.time}: €${pt.equity.toFixed(2)}` });
      const circle = svgEl("circle", { cx: pt.x, cy: pt.y, r: 3.5, fill: stroke, stroke: "#0f172a", "stroke-width": 1.5 }, titleNode);
      g.appendChild(circle);
    });
  }

  /* ── Render: #monitor Workspace ────────────────────────────────────────── */

  function renderMonitorWorkspace() {
    const doc = getDoc();
    const root = doc && doc.getElementById("workspace-monitor");
    if (!root) return;

    const d = state.data || {};
    const broker = d.broker_status || {};
    const cfg = getActiveConfigValues();
    const signals = Array.isArray(d.active_signals) ? d.active_signals : [];
    const activeStops = broker.active_server_stops ?? signals.filter((s) => s.broker_stop_order_id).length;
    const pureTsl = cfg.PURE_TSL_MODE !== undefined ? Boolean(cfg.PURE_TSL_MODE) : true;
    const cutoffStr = cfg.SESSION_CLOSE_CUTOFF_UTC || "16:20";

    // KPI strip
    const kpiRoot = ensureChildSection(root, "monitor-kpis", () => el("div", { id: "monitor-kpis", className: "wb-kpi-grid" }));
    clearNode(kpiRoot);
    kpiRoot.appendChild(
      el(
        "div",
        { className: "wb-kpi-card" },
        el("div", { className: "wb-kpi-label", text: "Active Positions" }),
        el("div", { id: "metric-active", className: "wb-kpi-value", text: `${signals.length} Open Trades` }),
        el("div", { className: "wb-kpi-sub", text: broker.broker_name || "Broker Portfolio" })
      )
    );
    kpiRoot.appendChild(
      el(
        "div",
        { className: "wb-kpi-card" },
        el("div", { className: "wb-kpi-label", text: "Native Broker Stops" }),
        el("div", { id: "metric-stops", className: "wb-kpi-value", style: "color:var(--emerald)", text: `${activeStops} Active` }),
        el("div", { className: "wb-kpi-sub", text: "Server-Side GTC" })
      )
    );
    kpiRoot.appendChild(
      el(
        "div",
        { className: "wb-kpi-card" },
        el("div", { className: "wb-kpi-label", text: "Exit Protocol" }),
        el("div", { className: "wb-kpi-value", text: pureTsl ? "100% Pure TSL" : "TP1/2/3 + TSL" }),
        el("div", { className: "wb-kpi-sub", text: "Per-position epoch governed" })
      )
    );
    kpiRoot.appendChild(
      el(
        "div",
        { className: "wb-kpi-card" },
        el("div", { className: "wb-kpi-label", text: `${cutoffStr} UTC Gap Cutoff` }),
        el("div", { id: "cutoff-countdown", className: "wb-kpi-value wb-mono", style: "color:var(--amber)", text: "--:--:--" }),
        el("div", { className: "wb-kpi-sub", text: "Asymmetric Session Protocol" })
      )
    );

    // Ensure equity card stays right after KPIs and render chart
    const eqShell = doc.getElementById("monitor-equity-shell");
    if (eqShell && eqShell.parentNode === root) {
      root.appendChild(eqShell);
    }
    renderEquityChart(d.equity_history, d.current_equity);

    // Funnel card
    const funnelSection = ensureChildSection(root, "funnel-panel", () => el("section", { id: "funnel-panel", className: "wb-card" }));
    root.appendChild(funnelSection);
    renderFunnelCard(funnelSection);

    // Active Positions section
    const posSection = ensureChildSection(root, "active-trades-section", () => el("section", { id: "active-trades-section", className: "wb-card" }));
    root.appendChild(posSection);
    renderPositionsSection(posSection);

    // Alerts table
    const alertsSection = ensureChildSection(root, "alerts-section", () => el("section", { id: "alerts-section", className: "wb-card" }));
    root.appendChild(alertsSection);
    renderAlertsSection(alertsSection);

    // Pairs directory
    const pairsSection = ensureChildSection(root, "pairs-section", () => el("details", { id: "pairs-section", className: "wb-details" }));
    root.appendChild(pairsSection);
    renderPairsSection(pairsSection);

    updateCountdown();
  }

  function ensureChildSection(parent, id, factory) {
    const doc = getDoc();
    let node = doc.getElementById(id);
    if (!node) {
      node = factory();
      parent.appendChild(node);
    }
    return node;
  }

  function normalizeFunnelData() {
    const d = state.data || {};
    const f = d.funnel || d.cycle_funnel || d.funnel_report || null;
    if (!f || typeof f !== "object") return null;
    const stagesObj = f.stage_counts || f.stages || f;
    const getCount = (k, alt) => Number(stagesObj[k] ?? stagesObj[alt] ?? f[k] ?? f[alt] ?? 0);
    const counts = {
      universe: getCount("universe", "universe_count"),
      screened: getCount("screened", "screened_count"),
      strategy: getCount("strategy", "strategy_count"),
      debate: getCount("debate", "debate_count"),
      gates: getCount("gates", "gates_count"),
      executed: getCount("executed", "executed_count"),
    };
    const breakdownRaw = f.gate_breakdown || f.rejections || f.gate_rejections || [];
    let breakdown = [];
    if (Array.isArray(breakdownRaw)) {
      breakdown = breakdownRaw;
    } else if (breakdownRaw && typeof breakdownRaw === "object") {
      breakdown = Object.keys(breakdownRaw).map((gid) => {
        const entry = breakdownRaw[gid];
        if (typeof entry === "number") {
          return { gate_id: gid, label: gid, count: entry, param_keys: [] };
        }
        return Object.assign({ gate_id: gid }, entry);
      });
    }
    return {
      status: f.status || f.cycle_state || "",
      epochId: f.config_epoch_id || getActiveEpochId(),
      timestamp: f.timestamp || f.generated_at || f.captured_at || "",
      counts,
      breakdown,
      reason: f.reason || f.message || "",
    };
  }

  function renderMissionControlCard(container) {
    clearNode(container);
    container.classList.add("wb-mc-card");
    const funnel = normalizeFunnelData();
    const progress = state.pipelineProgress || (state.data && state.data.pipeline_progress) || null;
    const isHalted = isSystemHalted();

    // Determine active 5-step index (1..5)
    let currentStep = 5;
    let stepStatus = "standby";
    if (progress && Number.isInteger(progress.step)) {
      currentStep = Math.min(5, Math.max(1, progress.step));
      stepStatus = progress.status || (currentStep === 5 ? "standby" : "active");
    } else if (funnel && funnel.counts) {
      if (funnel.status === "off_hours" || isHalted) {
        currentStep = 5;
        stepStatus = isHalted ? "halted" : "standby";
      } else if (funnel.counts.executed > 0) {
        currentStep = 5;
        stepStatus = "standby";
      } else if (funnel.counts.gates > 0 || funnel.counts.strategy > 0) {
        currentStep = 4;
        stepStatus = "active";
      } else if (funnel.counts.screened > 0) {
        currentStep = 3;
        stepStatus = "active";
      } else if (funnel.counts.universe > 0) {
        currentStep = 2;
        stepStatus = "active";
      }
    }

    const stepDefs = [
      [1, "Market Check"],
      [2, "Ingest Candles"],
      [3, "Strategy Screen"],
      [4, "Risk Sizing"],
      [5, "Order Route"],
    ];

    // Header
    const statusBadge = isHalted
      ? el("span", { className: "wb-badge wb-badge--rose wb-mono", text: "SYSTEM HALTED" })
      : (stepStatus === "active"
        ? el("span", { className: "wb-badge wb-badge--emerald wb-mono", text: "● SCANNING" })
        : el("span", { className: "wb-badge wb-badge--sky wb-mono", text: "● STANDBY / MONITORING" }));

    const header = el(
      "div",
      { className: "wb-mc-header" },
      el(
        "div",
        { className: "wb-mc-title-group" },
        el("h2", { className: "wb-card-title", text: "Mission Control — Live Pipeline Workflow" }),
        statusBadge
      ),
      el(
        "div",
        { className: "wb-pill-row" },
        funnel && funnel.epochId && state.viewMode === "advanced"
          ? el("span", { className: "wb-badge wb-badge--sky wb-mono", text: `Cycle Epoch: ${funnel.epochId}` })
          : null,
        el(
          "button",
          {
            type: "button",
            id: "quick-settings-toggle-btn",
            className: "wb-btn wb-btn--sm wb-btn--primary",
            style: "font-size: 0.75rem; padding: 0.25rem 0.65rem; display: inline-flex; align-items: center; gap: 0.35rem;",
            onClick: () => toggleQuickSettingsDrawer(true),
          },
          "⚙️ Quick Settings"
        ),
        el(
          "button",
          {
            type: "button",
            id: "mc-mode-toggle-btn",
            className: "wb-btn wb-btn--sm wb-btn--ghost",
            style: "font-size: 0.72rem; padding: 0.2rem 0.5rem;",
            onClick: () => {
              state.viewMode = state.viewMode === "simple" ? "advanced" : "simple";
              try {
                const win = getWin();
                if (win && win.sessionStorage) win.sessionStorage.setItem("quant_workbench_view_mode", state.viewMode);
              } catch (_) {}
              renderAllVisible();
            },
          },
          state.viewMode === "simple" ? "Switch to Advanced" : "Switch to Simple"
        )
      )
    );
    container.appendChild(header);

    // 5-Step Stepper
    const stepperEl = el("div", { id: "workflow-stepper", className: "wb-stepper", role: "progressbar", "aria-valuenow": String(currentStep), "aria-valuemin": "1", "aria-valuemax": "5" });
    stepDefs.forEach(([num, label]) => {
      let stateClass = "wb-step--pending";
      let iconText = String(num);

      if (num < currentStep || (currentStep === 5 && stepStatus === "standby" && funnel && funnel.counts && funnel.counts.universe > 0)) {
        stateClass = "wb-step--completed";
        iconText = "✓";
      } else if (num === currentStep) {
        stateClass = (stepStatus === "standby" ? "wb-step--completed" : "wb-step--active");
        iconText = (stepStatus === "standby" ? "✓" : String(num));
      }

      stepperEl.appendChild(
        el(
          "div",
          { id: `step-${num}`, className: `wb-step ${stateClass}`, dataset: { step: String(num) } },
          el("div", { className: "wb-step-num", text: iconText }),
          el(
            "div",
            { className: "wb-step-info" },
            el("span", { className: "wb-step-name", text: label })
          )
        )
      );
    });
    container.appendChild(stepperEl);

    // Dynamic "Now / Where / What's Next" Box
    const nowDetail = (progress && progress.detail)
      ? progress.detail
      : (isHalted
        ? "Trading engine is currently HALTED following emergency panic flatten."
        : (currentStep === 5 || stepStatus === "standby"
          ? "Cycle complete. Position monitor actively watching stops and ratchets."
          : `Running step ${currentStep}: ${stepDefs[currentStep - 1] ? stepDefs[currentStep - 1][1] : ""}`));

    const whereDetail = (funnel && funnel.counts && funnel.counts.universe > 0)
      ? `${funnel.counts.executed || 0} executed • ${funnel.counts.screened || 0} screened of ${funnel.counts.universe} total`
      : `${(state.data && state.data.pairs_watchlist ? state.data.pairs_watchlist.length : 24)} universe pairs monitored`;

    const nextDetail = (currentStep === 5 || stepStatus === "standby")
      ? "Next 5-minute automated universe scan cycle"
      : `Step ${(currentStep % 5) + 1}: ${stepDefs[currentStep % 5][1]}`;

    const nowBox = el(
      "div",
      { className: "wb-mc-now-box" },
      el(
        "div",
        { className: "wb-mc-col" },
        el("span", { className: "wb-mc-col-label", text: "NOW" }),
        el("span", { id: "workflow-now-val", className: "wb-mc-col-val", text: nowDetail })
      ),
      el(
        "div",
        { className: "wb-mc-col" },
        el("span", { className: "wb-mc-col-label", text: "WHERE" }),
        el("span", { id: "workflow-where-val", className: "wb-mc-col-val", text: whereDetail })
      ),
      el(
        "div",
        { className: "wb-mc-col" },
        el("span", { className: "wb-mc-col-label", text: "WHAT'S NEXT" }),
        el("span", { id: "workflow-next-val", className: "wb-mc-col-val", text: nextDetail })
      )
    );
    container.appendChild(nowBox);

    // Live Countdowns Tray
    const countdownsEl = el(
      "div",
      { className: "wb-mc-countdowns" },
      el(
        "div",
        { className: "wb-countdown-chip" },
        el("span", { className: "wb-muted", text: "Next Scan:" }),
        el("span", { id: "next-scan-countdown", className: "wb-countdown-val", text: "--:--" })
      ),
      el(
        "div",
        { className: "wb-countdown-chip" },
        el("span", { className: "wb-muted", text: "Stop Monitor:" }),
        el("span", { id: "next-monitor-countdown", className: "wb-countdown-val", text: "--s" })
      ),
      el(
        "div",
        { className: "wb-countdown-chip" },
        el("span", { className: "wb-muted", text: "LSE Close Cutoff:" }),
        el("span", { id: "cutoff-countdown-mc", className: "wb-countdown-val", text: "--:--:--" })
      )
    );
    container.appendChild(countdownsEl);

    // Decision Funnel Stages (preserving stage-count metrics compatibility)
    if (funnel && funnel.counts && funnel.counts.universe > 0 && funnel.status !== "pre_scan" && funnel.status !== "off_hours") {
      const stageDefsList = (state.viewMode === "advanced")
        ? [
            ["universe", "Universe"],
            ["screened", "Screened"],
            ["strategy", "Strategy"],
            ["debate", "Debate"],
            ["gates", "Risk Gates"],
            ["executed", "Executed"],
          ]
        : [
            ["universe", "Universe Assets"],
            ["screened", "Data Ingested"],
            ["strategy", "Strategy Screened"],
            ["gates", "Risk Sized"],
            ["executed", "Orders Routed"],
          ];
      const total = Math.max(1, funnel.counts.universe);
      const stagesEl = el("div", { className: "wb-funnel-stages", style: "margin-bottom:1rem;" });
      stageDefsList.forEach(([key, label]) => {
        const val = funnel.counts[key] || 0;
        const pct = Math.round((val / total) * 100);
        stagesEl.appendChild(
          el(
            "div",
            { className: "wb-funnel-stage", dataset: { stage: key } },
            el("span", { className: "wb-funnel-stage-name", text: label }),
            el("span", { className: "wb-funnel-stage-count", text: String(val) }),
            el("span", { className: "wb-funnel-stage-pct", text: `${pct}% of universe` })
          )
        );
      });
      container.appendChild(stagesEl);
    } else if (!funnel || funnel.counts.universe === 0 || funnel.status === "pre_scan" || funnel.status === "off_hours") {
      const msg = funnel && funnel.reason
        ? funnel.reason
        : (funnel && funnel.status === "off_hours"
          ? "Off-Hours / Market Closed: Next funnel scan runs when exchange sessions open."
          : "Pre-Scan State: No completed universe scan cycle recorded yet.");
      container.appendChild(el("div", { id: "funnel-empty-state", className: "wb-empty", style: "margin-bottom:1rem;", text: msg }));
    }

    // Plain-English Rejections List
    if (funnel && Array.isArray(funnel.breakdown) && funnel.breakdown.length > 0) {
      const rejTitle = el("div", { className: "wb-label", style: "margin-bottom:0.5rem", text: "Universe Filter Results (Plain English)" });
      container.appendChild(rejTitle);
      const rejectionsList = el("div", { className: "wb-mc-rejections" });
      const legacyGateList = el("div", { className: "wb-gate-list", style: "margin-top:0.5rem;" });

      funnel.breakdown.forEach((g) => {
        const gid = g.gate_id || "gate";
        const label = g.label || gid;
        const count = g.count ?? 0;
        const paramKeys = Array.isArray(g.param_keys) ? g.param_keys.filter(Boolean) : [];
        const isStructural = Boolean(g.structural) || STRUCTURAL_GATES.has(gid) || paramKeys.length === 0;
        const translatedReason = translateGateRejection(gid, label);

        const card = el(
          "div",
          { className: "wb-mc-rejection-card", dataset: { gateId: gid } },
          el(
            "div",
            { className: "wb-mc-rejection-text" },
            el("span", { className: "wb-badge wb-badge--rose wb-mono", text: `${count} stocks` }),
            el("span", { text: translatedReason })
          )
        );

        if (!isStructural && paramKeys.length > 0) {
          const targetParam = paramKeys[0];
          card.appendChild(
            el("button", {
              type: "button",
              className: "wb-btn wb-btn--sm wb-btn--ghost",
              dataset: { deepLink: "config", param: targetParam, gate: gid },
              text: `Adjust ${targetParam}`,
              onClick: () => {
                state.viewMode = "advanced";
                navigateTo("config", { param: targetParam, gate: gid });
              },
            })
          );
          legacyGateList.appendChild(
            el("button", {
              type: "button",
              className: "wb-gate-chip wb-gate-chip--link",
              dataset: { deepLink: "config", param: targetParam, gate: gid },
              text: `${label}: ${count} -> Tune ${targetParam}`,
              onClick: () => navigateTo("config", { param: targetParam, gate: gid }),
            })
          );
        } else {
          legacyGateList.appendChild(
            el("span", {
              className: "wb-gate-chip wb-gate-chip--structural",
              dataset: { gateId: gid, structural: "true" },
              text: `${label}: ${count} (Structural)`,
            })
          );
        }
        rejectionsList.appendChild(card);
      });
      container.appendChild(rejectionsList);
      container.appendChild(legacyGateList);
    }

    // Expandable 24-Stock Pipeline Tracker
    const watchlist = (state.data && Array.isArray(state.data.pairs_watchlist)) ? state.data.pairs_watchlist : [];
    const activeSignals = (state.data && Array.isArray(state.data.active_signals)) ? state.data.active_signals : [];
    const activeTickerSet = new Set(activeSignals.map((s) => s.ticker));

    const trackerSection = el("div", { className: "wb-tracker-tray" });
    const toggleBtn = el(
      "button",
      {
        type: "button",
        id: "tracker-toggle-btn",
        className: "wb-btn wb-btn--sm wb-btn--ghost",
        style: "display: flex; align-items: center; gap: 0.4rem; font-weight: 600;",
        onClick: () => {
          state.trackerExpanded = !state.trackerExpanded;
          renderMissionControlCard(container);
        },
      },
      el("span", { text: state.trackerExpanded ? "▲ Hide Universe Stock Tracker" : `▼ Show Universe Stock Tracker (${watchlist.length || 24} assets)` })
    );
    trackerSection.appendChild(toggleBtn);

    if (state.trackerExpanded) {
      const grid = el("div", { className: "wb-tracker-grid" });

      // Column 1: Filtered Out
      const colFiltered = el("div", { className: "wb-tracker-col" });
      colFiltered.appendChild(el("div", { className: "wb-tracker-col-header" }, el("span", { text: "1. Filtered Out" })));

      // Column 2: In Debate
      const colDebate = el("div", { className: "wb-tracker-col" });
      colDebate.appendChild(el("div", { className: "wb-tracker-col-header" }, el("span", { text: "2. In Debate" })));

      // Column 3: Risk Sizing
      const colSizing = el("div", { className: "wb-tracker-col" });
      colSizing.appendChild(el("div", { className: "wb-tracker-col-header" }, el("span", { text: "3. Risk Sizing" })));

      // Column 4: Active Trade
      const colActive = el("div", { className: "wb-tracker-col" });
      colActive.appendChild(el("div", { className: "wb-tracker-col-header" }, el("span", { text: "4. Active Trade" })));

      // Populate Active Trades
      activeSignals.forEach((sig) => {
        const item = el(
          "div",
          { className: "wb-tracker-item" },
          el("span", { className: "wb-tracker-item-ticker", text: sig.ticker || sig.display_name }),
          el("span", { className: "wb-badge wb-badge--emerald", text: `${sig.direction} • ${sig.pnl_pct ? sig.pnl_pct.toFixed(1) + "%" : "ACTIVE"}` })
        );
        colActive.appendChild(item);
      });

      // Populate other watchlist items
      watchlist.forEach((pair) => {
        const tk = pair.long_ticker || pair.underlying;
        if (activeTickerSet.has(tk)) return;

        // If currently in debate/progress
        if (progress && Array.isArray(progress.active_assets) && progress.active_assets.includes(tk)) {
          if (progress.step === 4) {
            colDebate.appendChild(el("div", { className: "wb-tracker-item" }, el("span", { className: "wb-tracker-item-ticker", text: tk }), el("span", { className: "wb-badge wb-badge--indigo", text: "Debating" })));
            return;
          } else if (progress.step === 5 || progress.step === 6) {
            colSizing.appendChild(el("div", { className: "wb-tracker-item" }, el("span", { className: "wb-tracker-item-ticker", text: tk }), el("span", { className: "wb-badge wb-badge--sky", text: "Sizing" })));
            return;
          }
        }

        // Filtered out item
        colFiltered.appendChild(
          el(
            "div",
            { className: "wb-tracker-item" },
            el("span", { className: "wb-tracker-item-ticker", text: tk }),
            el("span", { className: "wb-muted", style: "font-size:0.7rem", text: pair.adx ? `ADX ${pair.adx}` : "Skipped" })
          )
        );
      });

      grid.appendChild(colFiltered);
      grid.appendChild(colDebate);
      grid.appendChild(colSizing);
      grid.appendChild(colActive);
      trackerSection.appendChild(grid);
    }

    container.appendChild(trackerSection);
  }

  function renderFunnelCard(container) {
    return renderMissionControlCard(container);
  }

  function renderPositionsSection(container) {
    clearNode(container);
    const d = state.data || {};
    const allSignals = Array.isArray(d.active_signals) ? d.active_signals : [];
    const activeEpoch = getActiveEpochId();
    const readOnly = isReadOnlyMode();
    const behindPositions = allSignals.filter((s) => !s.config_epoch_id || s.config_epoch_id !== activeEpoch);

    const filterButtons = ["ALL", "LONG", "SHORT"].map((dir) =>
      el("button", {
        type: "button",
        id: `filter-${dir.toLowerCase()}`,
        className: state.directionFilter === dir ? "wb-btn wb-btn--sm wb-btn--primary" : "wb-btn wb-btn--sm",
        text: dir === "ALL" ? "All" : `${dir[0]}${dir.slice(1).toLowerCase()} Only`,
        onClick: () => {
          state.directionFilter = dir;
          renderPositionsSection(container);
        },
      })
    );

    const reEpochAllBtn = el("button", {
      type: "button",
      id: "re-epoch-all-btn",
      className: "wb-btn wb-btn--sm wb-btn--amber",
      disabled: readOnly || behindPositions.length === 0,
      text: `Re-Epoch All (${behindPositions.length})`,
      onClick: openReEpochAllModal,
    });

    container.appendChild(
      el(
        "div",
        { className: "wb-card-header" },
        el(
          "h2",
          { className: "wb-card-title" },
          "Active Positions / Open Trades",
          el("span", { id: "signals-badge", className: "wb-badge wb-mono", text: String(allSignals.length) })
        ),
        el("div", { className: "wb-action-row" }, ...filterButtons, reEpochAllBtn)
      )
    );

    const filtered = state.directionFilter === "ALL"
      ? allSignals
      : allSignals.filter((s) => String(s.direction || "").toUpperCase() === state.directionFilter);

    const grid = el("div", { id: "signals-grid", className: "wb-positions-grid" });
    if (filtered.length === 0) {
      grid.appendChild(
        el("div", {
          className: "wb-empty",
          style: "grid-column:1 / -1",
          text: allSignals.length === 0
            ? "No open positions currently tracked."
            : `No ${state.directionFilter} positions match the current filter.`,
        })
      );
      container.appendChild(grid);
      return;
    }

    filtered.forEach((s) => {
      grid.appendChild(buildPositionCard(s, activeEpoch, readOnly));
    });
    container.appendChild(grid);
  }

  function buildPositionCard(s, activeEpoch, readOnly) {
    const isLong = String(s.direction || "LONG").toUpperCase() === "LONG";
    const lev = s.leverage || (s.ticker && String(s.ticker).includes("5") ? 5 : 3);
    const prefix = s.currency_prefix !== undefined && s.currency_prefix !== null ? s.currency_prefix : "€";
    const suffix = s.currency_suffix || "";
    const curCode = (s.currency_code || (suffix === "p" ? "GBX" : (prefix === "$" ? "USD" : "EUR"))).toUpperCase();
    const remQty = Number(s.remaining_quantity !== undefined ? s.remaining_quantity : (s.quantity || 0));
    const entryPx = Number(s.entry_price || 0);
    const curPx = Number(s.current_price || entryPx);
    const pnlAmt = Number(s.pnl_amount || 0);
    const pnlPct = Number(s.pnl_pct || 0);
    const pnlEur = Number(s.pnl_amount_eur ?? pnlAmt);
    const isPanicFailed = Boolean(s.panic_failed || s.unprotected_reason);
    const posEpoch = s.config_epoch_id || null;
    const isLegacy = !posEpoch;
    const isBehind = !isLegacy && posEpoch !== activeEpoch;

    const badges = [
      el("span", {
        className: isLong ? "wb-badge wb-badge--emerald" : "wb-badge wb-badge--rose",
        text: `${isLong ? "LONG" : "SHORT"} ${lev}x`,
      }),
    ];
    if (state.viewMode === "advanced") {
      if (isLegacy) {
        badges.push(el("span", { className: "wb-badge wb-badge--amber", text: "Legacy (Unstamped)" }));
      } else if (isBehind) {
        badges.push(el("span", { className: "wb-badge wb-badge--amber wb-mono", text: `Behind ${posEpoch} -> ${activeEpoch}` }));
      } else {
        badges.push(el("span", { className: "wb-badge wb-badge--emerald wb-mono", text: posEpoch }));
      }
    } else {
      badges.push(el("span", { className: "wb-badge wb-badge--emerald wb-mono", text: "● Stop Armed" }));
    }
    if (isPanicFailed) {
      badges.push(el("span", { className: "wb-badge wb-badge--rose", text: "UNPROTECTED / PANIC FAILED" }));
    }

    const actionButtons = [];
    if (isPanicFailed) {
      actionButtons.push(
        el("button", {
          type: "button",
          className: "wb-btn wb-btn--sm wb-btn--danger",
          disabled: readOnly,
          text: "Retry Liquidation",
          onClick: () => openRetryLiquidationModal(s),
        })
      );
    }
    if (isLegacy || isBehind) {
      actionButtons.push(
        el("button", {
          type: "button",
          className: "wb-btn wb-btn--sm wb-btn--amber",
          disabled: readOnly,
          text: `Re-Epoch to ${activeEpoch}`,
          onClick: () => openReEpochModal(s),
        })
      );
    }

    const pnlStyle = pnlEur > 0 ? "color:var(--emerald)" : (pnlEur < 0 ? "color:var(--rose)" : "");
    const eurSign = pnlEur > 0 ? "+" : (pnlEur < 0 ? "-" : "");
    const eurFormatted = `${eurSign}€${Math.abs(pnlEur).toFixed(2)}`;
    const pctFormatted = `(${pnlPct >= 0 ? "+" : ""}${pnlPct.toFixed(2)}%)`;

    let posValueStr = `${remQty.toFixed(2)} shs`;
    if (curCode === "GBX" || suffix === "p") {
      const totalGbp = (remQty * curPx) / 100.0;
      const totalEur = totalGbp * 1.204;
      posValueStr = `${remQty.toFixed(2)} shs • £${formatNumber(totalGbp, 2)} (≈ €${formatNumber(totalEur, 2)})`;
    } else if (curCode === "USD" || prefix === "$") {
      const totalUsd = remQty * curPx;
      const totalEur = totalUsd / 1.085;
      posValueStr = `${remQty.toFixed(2)} shs • $${formatNumber(totalUsd, 2)} (≈ €${formatNumber(totalEur, 2)})`;
    } else if (curCode === "EUR") {
      const totalEur = remQty * curPx;
      posValueStr = `${remQty.toFixed(2)} shs • €${formatNumber(totalEur, 2)}`;
    }

    return el(
      "article",
      { className: isPanicFailed ? "wb-pos-card wb-pos-card--unprotected" : "wb-pos-card" },
      el("div", { className: isLong ? "wb-pos-accent wb-pos-accent--long" : "wb-pos-accent wb-pos-accent--short" }),
      el(
        "div",
        null,
        el(
          "div",
          { className: "wb-card-header", style: "margin-bottom:0.5rem" },
          el(
            "div",
            null,
            el("div", { style: "font-weight:800;font-size:1rem", text: s.name || s.underlying || s.ticker }),
            el("div", { className: "wb-mono wb-muted", style: "font-size:0.75rem", text: `${s.ticker} • ${curCode === "GBX" ? "GBX (London Pence • 100p = £1)" : curCode}` })
          ),
          el("div", { className: "wb-pill-row" }, badges)
        ),
        isPanicFailed && s.unprotected_reason
          ? el("div", { className: "wb-banner wb-banner--danger", style: "padding:0.45rem 0.65rem;margin-bottom:0.6rem;font-size:0.75rem", text: s.unprotected_reason })
          : null,
        el(
          "div",
          { className: "wb-pos-pnl-box" },
          el(
            "div",
            { style: "display:flex;justify-content:space-between;align-items:baseline;gap:0.5rem" },
            el("div", { className: "wb-mono", style: `font-weight:800;font-size:1.15rem;${pnlStyle}`, text: `${eurFormatted} ${pctFormatted}` }),
            el("div", { className: "wb-mono wb-muted", style: "font-size:0.75rem", text: posValueStr })
          ),
          curCode !== "EUR"
            ? el(
                "div",
                { className: "wb-mono wb-muted", style: "font-size:0.74rem;margin-top:0.25rem;display:flex;justify-content:space-between" },
                el("span", { text: "Native Broker P&L:" }),
                el("span", { style: `font-weight:600;${pnlAmt > 0 ? "color:var(--emerald)" : (pnlAmt < 0 ? "color:var(--rose)" : "")}`, text: formatCurrencyAmount(pnlAmt, prefix, suffix) })
              )
            : null,
          el(
            "div",
            { className: "wb-mono wb-muted", style: "font-size:0.74rem;margin-top:0.35rem;display:flex;justify-content:space-between;flex-wrap:wrap;gap:0.25rem" },
            el("span", { text: `Entry: ${formatPriceWithCurrency(entryPx, curCode, prefix, suffix, { compact: true })}` }),
            el("span", { text: `Live: ${formatPriceWithCurrency(curPx, curCode, prefix, suffix, { compact: true })}` })
          )
        ),
        el(
          "div",
          { className: "wb-pos-rows", style: "margin-top:0.65rem" },
          el("div", { className: "wb-pos-row" }, el("span", { text: "Stop Loss (SL)" }), el("span", { className: "wb-mono", text: formatPriceWithCurrency(s.stop_loss || 0, curCode, prefix, suffix, { compact: false }) })),
          el("div", { className: "wb-pos-row" }, el("span", { text: "Trailing Stop (TSL)" }), el("span", { className: "wb-mono", text: `${formatPriceWithCurrency(s.trailing_stop_loss || s.stop_loss || 0, curCode, prefix, suffix, { compact: false })} (${s.tsl_status || "Active"})` })),
          el("div", { className: "wb-pos-row" }, el("span", { text: "Broker GTC Stop" }), el("span", { className: "wb-mono", text: s.broker_stop_order_id ? `#${s.broker_stop_order_id}` : "Unverified / Missing" }))
        )
      ),
      el(
        "div",
        { className: "wb-pos-actions" },
        el("span", { className: "wb-muted wb-mono", style: "font-size:0.72rem;margin-right:auto", text: s.decay_label || s.status || "ACTIVE" }),
        actionButtons
      )
    );
  }

  function renderAlertsSection(container) {
    clearNode(container);
    const notifs = Array.isArray(state.data.recent_notifications) ? state.data.recent_notifications : [];
    container.appendChild(
      el(
        "div",
        { className: "wb-card-header" },
        el(
          "h3",
          { className: "wb-card-title" },
          "Recent Trade & Alert History",
          el("span", { id: "alerts-badge", className: "wb-badge wb-mono", text: String(notifs.length) })
        ),
        el("span", { className: "wb-muted wb-mono", style: "font-size:0.75rem", text: "Real-Time Event Stream" })
      )
    );

    const tbody = el("tbody", { id: "alerts-table-body" });
    if (notifs.length === 0) {
      tbody.appendChild(
        el("tr", null, el("td", { colSpan: "6", className: "wb-muted", style: "text-align:center;padding:1.25rem", text: "No recent alerts or execution events recorded." }))
      );
    } else {
      notifs.forEach((n) => {
        const ev = String(n.event_type || "ALERT").toUpperCase();
        const priceStr = n.price !== null && n.price !== undefined ? Number(n.price).toFixed(2) : "—";
        const pnlStr = n.pnl !== null && n.pnl !== undefined ? `${Number(n.pnl).toFixed(2)}${n.pnl_pct !== null && n.pnl_pct !== undefined ? ` (${Number(n.pnl_pct).toFixed(2)}%)` : ""}` : "—";
        tbody.appendChild(
          el(
            "tr",
            null,
            el("td", null, el("span", { className: "wb-badge wb-mono", text: ev })),
            el("td", { className: "wb-mono", text: `${n.ticker || "—"} ${n.underlying ? `(${n.underlying})` : ""}` }),
            el("td", { className: "wb-mono", text: priceStr }),
            el("td", { className: "wb-mono", text: pnlStr }),
            el("td", { text: n.rationale || "—" }),
            el("td", { className: "wb-mono wb-muted", text: n.timestamp || "—" })
          )
        );
      });
    }

    container.appendChild(
      el(
        "div",
        { className: "wb-table-scroll" },
        el(
          "table",
          { className: "wb-table" },
          el(
            "thead",
            null,
            el(
              "tr",
              null,
              el("th", { text: "Event" }),
              el("th", { text: "Ticker / Asset" }),
              el("th", { text: "Price" }),
              el("th", { text: "P&L" }),
              el("th", { text: "Rationale" }),
              el("th", { text: "Timestamp (UTC)" })
            )
          ),
          tbody
        )
      )
    );
  }

  function renderPairsSection(detailsEl) {
    const wasOpen = detailsEl.open;
    clearNode(detailsEl);
    const pairs = Array.isArray(state.data.pairs_watchlist) ? state.data.pairs_watchlist : [];
    const broker = state.data.broker_status || {};
    const eurRatio = broker.eur_ratio || computeEurRatio(pairs);

    detailsEl.appendChild(
      el(
        "summary",
        null,
        el("span", { text: `Monitored Asset Directory (${pairs.length} Pairs${eurRatio ? ` • ${eurRatio} EUR` : ""})` }),
        el("span", { className: "wb-badge wb-badge--indigo", text: "Toggle Directory" })
      )
    );

    const tbody = el("tbody", { id: "pairs-table-body" });
    if (pairs.length === 0) {
      tbody.appendChild(el("tr", null, el("td", { colSpan: "6", className: "wb-muted", text: "No universe pairs loaded." })));
    } else {
      pairs.forEach((p) => {
        tbody.appendChild(
          el(
            "tr",
            null,
            el("td", { className: "wb-mono", text: p.underlying || "—" }),
            el("td", { text: `${p.name || ""} (${p.category || ""})` }),
            el("td", { className: "wb-mono", style: "color:var(--emerald)", text: p.long_ticker || "—" }),
            el("td", { className: "wb-mono", style: "color:var(--rose)", text: p.short_ticker || "—" }),
            el("td", null, el("span", { className: p.has_5x ? "wb-badge wb-badge--amber" : "wb-badge", text: p.has_5x ? "5x Available" : "3x Standard" })),
            el("td", { className: "wb-mono", text: p.currency || p.long_currency || "EUR" })
          )
        );
      });
    }

    detailsEl.appendChild(
      el(
        "div",
        { className: "wb-details-body" },
        el(
          "div",
          { className: "wb-table-scroll" },
          el(
            "table",
            { className: "wb-table" },
            el(
              "thead",
              null,
              el(
                "tr",
                null,
                el("th", { text: "Underlying" }),
                el("th", { text: "Name / Sector" }),
                el("th", { text: "Long ETP" }),
                el("th", { text: "Short (Inverse) ETP" }),
                el("th", { text: "Leverage Policy" }),
                el("th", { text: "Currency" })
              )
            ),
            tbody
          )
        )
      )
    );
    if (wasOpen) detailsEl.open = true;
  }

  /* ── Render: #config Workspace & Sticky Diff Drawer ────────────────────── */

  function getCrossSectionalMetricsMap() {
    const d = state.data || {};
    const funnel = d.funnel || d.cycle_funnel || d.funnel_report || {};
    const audit = d.audit || d.improvement_audit || {};
    return (
      funnel.cross_sectional_metrics ||
      funnel.indicator_summaries ||
      funnel.metric_summaries ||
      audit.cross_sectional_metrics ||
      d.cross_sectional_metrics ||
      {}
    );
  }

  function findRelatedMetrics(paramKey, gateId, schemaByKey) {
    const metrics = new Set();
    const spec = schemaByKey[paramKey];
    if (spec && spec.screener_metric_gate && spec.screener_metric_gate.metric) {
      metrics.add(spec.screener_metric_gate.metric);
    }
    if (PARAM_TO_METRIC_HINTS[paramKey]) {
      PARAM_TO_METRIC_HINTS[paramKey].forEach((m) => metrics.add(m));
    }
    if (gateId && GATE_TO_METRIC_HINTS[gateId]) {
      GATE_TO_METRIC_HINTS[gateId].forEach((m) => metrics.add(m));
    }
    return Array.from(metrics);
  }

  function ensureConfigToolbar(controlsRoot, readOnly) {
    const doc = getDoc();
    const profiles = getBuiltinProfiles();
    const profileIds = Object.keys(profiles);
    const profilesSig = profileIds.map((pid) => `${pid}:${(profiles[pid] && profiles[pid].label) || pid}`).join("|");

    let toolbarCard = doc.getElementById("config-toolbar-card");
    if (!toolbarCard) {
      const profileSelect = el("select", { id: "profile-select", className: "wb-select" });
      const loadProfileBtn = el("button", {
        type: "button",
        id: "load-profile-btn",
        className: "wb-btn wb-btn--sm",
        text: "Load Profile",
        onClick: () => {
          if (profileSelect.value) loadBuiltinProfileIntoDraft(profileSelect.value);
        },
      });
      const addParamBtn = el("button", {
        type: "button",
        id: "add-param-btn",
        className: "wb-btn wb-btn--sm wb-btn--primary",
        text: "+ Add Parameter",
        onClick: openAddParameterModal,
      });
      const searchInput = el("input", {
        id: "config-search-input",
        type: "search",
        className: "wb-input",
        placeholder: "Filter parameters by key or label...",
        value: state.configSearch,
        onInput: (ev) => {
          state.configSearch = ev.target.value;
          const params = getSchemaParameters();
          renderConfigParamGroups(controlsRoot, params, isReadOnlyMode());
          state.lastSchemaSignature = computeConfigSignature(params, isReadOnlyMode());
        },
      });
      const groupSelect = el(
        "select",
        {
          id: "config-group-select",
          className: "wb-select",
          onChange: (ev) => {
            state.configGroup = ev.target.value;
            const params = getSchemaParameters();
            renderConfigParamGroups(controlsRoot, params, isReadOnlyMode());
            state.lastSchemaSignature = computeConfigSignature(params, isReadOnlyMode());
          },
        },
        el("option", { value: "all", text: "All Groups" }),
        GROUP_ORDER.map((g) => el("option", { value: g, text: GROUP_LABELS[g] || g })),
        el("option", { value: "dynamic", text: "Custom Dynamic Gates" })
      );
      groupSelect.value = state.configGroup;

      toolbarCard = el(
        "div",
        { id: "config-toolbar-card", className: "wb-card" },
        el(
          "div",
          { className: "wb-config-toolbar" },
          el(
            "div",
            { className: "wb-field", style: "flex:1 1 200px" },
            el("label", { className: "wb-label", htmlFor: "config-search-input", text: "Search Parameters" }),
            searchInput
          ),
          el(
            "div",
            { className: "wb-field", style: "flex:0 1 200px" },
            el("label", { className: "wb-label", htmlFor: "config-group-select", text: "Group" }),
            groupSelect
          ),
          el(
            "div",
            { className: "wb-field", style: "flex:0 1 220px" },
            el("label", { className: "wb-label", htmlFor: "profile-select", text: "Built-in Profiles" }),
            el("div", { className: "wb-action-row" }, profileSelect, loadProfileBtn)
          ),
          addParamBtn
        )
      );
      controlsRoot.appendChild(toolbarCard);
    }

    const profileSelectEl = doc.getElementById("profile-select");
    if (profileSelectEl && toolbarCard.dataset.profilesSig !== profilesSig) {
      const prevVal = profileSelectEl.value;
      clearNode(profileSelectEl);
      if (profileIds.length === 0) {
        profileSelectEl.appendChild(el("option", { value: "", text: "No profiles available" }));
      } else {
        profileIds.forEach((pid) => {
          profileSelectEl.appendChild(el("option", { value: pid, text: profiles[pid].label || pid }));
        });
        profileSelectEl.value = prevVal && profileIds.includes(prevVal) ? prevVal : profileIds[0];
      }
      toolbarCard.dataset.profilesSig = profilesSig;
    }

    const loadProfileBtnEl = doc.getElementById("load-profile-btn");
    if (loadProfileBtnEl) loadProfileBtnEl.disabled = readOnly || profileIds.length === 0;
    const addParamBtnEl = doc.getElementById("add-param-btn");
    if (addParamBtnEl) addParamBtnEl.disabled = readOnly;
    const groupSelectEl = doc.getElementById("config-group-select");
    if (groupSelectEl && groupSelectEl.value !== state.configGroup) {
      groupSelectEl.value = state.configGroup;
    }
  }

  function renderConfigDeepLinkSlot(controlsRoot, schemaByKey) {
    const doc = getDoc();
    let slot = doc.getElementById("config-deeplink-slot");
    if (!slot) {
      slot = el("div", { id: "config-deeplink-slot" });
      controlsRoot.appendChild(slot);
    }
    clearNode(slot);

    if (!state.route.param && !state.route.gate) return;

    const targetKey = state.route.param;
    const targetSpec = targetKey ? schemaByKey[targetKey] : null;
    if (!targetSpec || (state.route.gate && STRUCTURAL_GATES.has(state.route.gate))) {
      slot.appendChild(
        el("div", {
          id: "config-deeplink-fallback",
          className: "wb-banner wb-banner--warn",
          style: "margin-top:0.85rem",
          text: `Deep-link notice: parameter '${targetKey || state.route.gate}' is unavailable (deleted custom parameter or non-tunable structural gate).`,
        })
      );
      return;
    }

    const crossMetrics = getCrossSectionalMetricsMap();
    const related = findRelatedMetrics(targetKey, state.route.gate, schemaByKey);
    const summaryLines = [];
    related.forEach((m) => {
      const stats = crossMetrics[m];
      if (stats && typeof stats === "object") {
        summaryLines.push(
          `${m}: median ${formatStat(stats.median)} (p25 ${formatStat(stats.p25)}, p75 ${formatStat(stats.p75)}, min ${formatStat(stats.min)}, max ${formatStat(stats.max)})`
        );
      }
    });
    slot.appendChild(
      el(
        "div",
        { id: "config-deeplink-callout", className: "wb-callout", style: "margin-top:0.85rem" },
        el("strong", { text: `Focused Gate Parameter: ${targetSpec.label || targetKey} (${targetKey})` }),
        summaryLines.length > 0
          ? el("div", { className: "wb-mono", style: "margin-top:0.35rem;font-size:0.78rem", text: summaryLines.join(" | ") })
          : el("div", { className: "wb-muted", style: "margin-top:0.25rem;font-size:0.78rem", text: "Cross-sectional distribution will appear after a completed universe scan cycle." })
      )
    );
  }

  function renderConfigParamGroups(controlsRoot, params, readOnly) {
    const doc = getDoc();
    let slot = doc.getElementById("config-param-groups-slot");
    if (!slot) {
      slot = el("div", { id: "config-param-groups-slot" });
      controlsRoot.appendChild(slot);
    }
    state.paramDomCache = {};
    clearNode(slot);

    const q = state.configSearch.trim().toLowerCase();
    const filtered = params.filter((p) => {
      if (!p || !p.key) return false;
      if (state.configGroup === "dynamic" && !p.is_dynamic) return false;
      if (state.configGroup !== "all" && state.configGroup !== "dynamic" && p.group !== state.configGroup) return false;
      if (q) {
        const hay = `${p.key} ${p.label || ""} ${p.impact_hint || ""}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });

    const groupsWrap = el("div", { className: "wb-param-groups" });
    if (filtered.length === 0) {
      groupsWrap.appendChild(
        el("div", {
          className: "wb-empty",
          text: params.length === 0
            ? "No schema metadata loaded in snapshot."
            : "No parameters match your current filter.",
        })
      );
      slot.appendChild(groupsWrap);
      return;
    }

    const byGroup = {};
    filtered.forEach((p) => {
      const g = p.group || "screener";
      if (!byGroup[g]) byGroup[g] = [];
      byGroup[g].push(p);
    });

    const orderedGroups = GROUP_ORDER.filter((g) => byGroup[g]).concat(
      Object.keys(byGroup).filter((g) => !GROUP_ORDER.includes(g))
    );

    orderedGroups.forEach((g) => {
      const groupGrid = el("div", { className: "wb-group-grid" });
      byGroup[g].forEach((spec) => {
        groupGrid.appendChild(buildParameterCard(spec, readOnly));
      });
      groupsWrap.appendChild(
        el(
          "section",
          { className: "wb-group-card", dataset: { group: g } },
          el("h2", { className: "wb-card-title", style: "margin-bottom:0.75rem", text: GROUP_LABELS[g] || g }),
          groupGrid
        )
      );
    });

    slot.appendChild(groupsWrap);
  }

  function renderQuickControls(container, readOnly) {
    const doc = getDoc();
    if (!doc) return;
    let card = doc.getElementById("quick-controls-card");
    if (!card) {
      card = el("section", { id: "quick-controls-card", className: "wb-quick-controls" });
      if (typeof container.insertBefore === "function" && container.firstChild) {
        container.insertBefore(card, container.firstChild);
      } else {
        container.appendChild(card);
      }
    }
    clearNode(card);

    const activeStrat = getEffectiveConfigValue("ACTIVE_STRATEGY") || "vwap";
    const currentRisk = Number(getEffectiveConfigValue("MAX_RISK_PER_TRADE") || 0.015);
    const currentSlMult = Number(getEffectiveConfigValue("ATR_SL_MULTIPLIER") || 2.2);
    const currentCutoff = String(getEffectiveConfigValue("SESSION_CLOSE_CUTOFF_UTC") || "16:20").trim();

    const header = el(
      "div",
      { className: "wb-mc-header" },
      el(
        "div",
        { className: "wb-mc-title-group" },
        el("h2", { className: "wb-card-title", text: "Quick Controls (Simple Mode)" }),
        el("span", { className: "wb-badge wb-badge--emerald", text: "Core 4 Controls" })
      ),
      el(
        "button",
        {
          type: "button",
          id: "toggle-advanced-from-quick-btn",
          className: "wb-btn wb-btn--sm wb-btn--primary",
          onClick: () => {
            state.viewMode = "advanced";
            try {
              const win = getWin();
              if (win && win.sessionStorage) win.sessionStorage.setItem("quant_workbench_view_mode", "advanced");
            } catch (_) {}
            renderConfigWorkspace(true);
            renderHeader();
          },
        },
        "Show Advanced Settings (50+ Parameters & Diff Drawer)"
      )
    );
    card.appendChild(header);

    const grid = el("div", { className: "wb-quick-grid" });

    // 1. Active Strategy
    const stratSelect = el(
      "select",
      {
        id: "quick-strat-select",
        className: "wb-input",
        disabled: readOnly,
        onChange: (e) => {
          stageParameterValue("ACTIVE_STRATEGY", e.target.value);
          renderQuickControls(container, readOnly);
        },
      },
      el("option", { value: "vwap", selected: activeStrat === "vwap", text: "VWAP Session Trend (Breakout & Volume)" }),
      el("option", { value: "ml_alpha", selected: activeStrat === "ml_alpha", text: "ML Alpha Engine (Multi-Factor ONNX)" }),
      el("option", { value: "supertrend", selected: activeStrat === "supertrend", text: "Supertrend (Experimental)" })
    );
    grid.appendChild(
      el(
        "div",
        { className: "wb-field" },
        el("label", { className: "wb-label", htmlFor: "quick-strat-select", text: "1. Active Strategy" }),
        stratSelect,
        el("div", { className: "wb-kpi-sub", text: "Governs technical screening & momentum rules" })
      )
    );

    // 2. Risk Level (Segmented Control: Conservative, Balanced, Aggressive)
    let riskTier = "balanced";
    if (currentRisk <= 0.008) riskTier = "conservative";
    else if (currentRisk >= 0.018) riskTier = "aggressive";

    const segControl = el("div", { className: "wb-segmented-control" });
    const tiers = [
      ["conservative", "Conservative (0.5%)", 0.0075, 2.5],
      ["balanced", "Balanced (1.0%)", 0.015, 2.2],
      ["aggressive", "Aggressive (1.5%)", 0.02, 1.8],
    ];
    tiers.forEach(([tierKey, tierLabel, riskVal, slVal]) => {
      segControl.appendChild(
        el(
          "button",
          {
            type: "button",
            id: `quick-risk-${tierKey}`,
            className: `wb-seg-btn ${riskTier === tierKey ? "wb-seg-btn--active" : ""}`,
            disabled: readOnly,
            onClick: () => {
              stageParameterValue("MAX_RISK_PER_TRADE", riskVal);
              stageParameterValue("ATR_SL_MULTIPLIER", slVal);
              renderQuickControls(container, readOnly);
            },
          },
          tierLabel
        )
      );
    });
    grid.appendChild(
      el(
        "div",
        { className: "wb-field" },
        el("label", { className: "wb-label", text: "2. Risk Profile" }),
        segControl,
        el("div", { className: "wb-kpi-sub", text: "Sets position sizing cap & stop distance" })
      )
    );

    // 3. Stop Loss Buffer (ATR Multiplier)
    const slSelect = el(
      "select",
      {
        id: "quick-sl-select",
        className: "wb-input",
        disabled: readOnly,
        onChange: (e) => {
          stageParameterValue("ATR_SL_MULTIPLIER", Number(e.target.value));
          renderQuickControls(container, readOnly);
        },
      },
      [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0].map((mult) =>
        el("option", {
          value: String(mult),
          selected: Math.abs(currentSlMult - mult) < 0.05,
          text: `${mult.toFixed(1)}x ATR Noise Buffer`,
        })
      )
    );
    grid.appendChild(
      el(
        "div",
        { className: "wb-field" },
        el("label", { className: "wb-label", htmlFor: "quick-sl-select", text: "3. Stop Loss Buffer" }),
        slSelect,
        el("div", { className: "wb-kpi-sub", text: "Multiplier above market noise" })
      )
    );

    // 4. Session Cutoff
    const cutoffInput = el("input", {
      type: "text",
      id: "quick-cutoff-input",
      className: "wb-input wb-mono",
      value: currentCutoff,
      disabled: readOnly,
      placeholder: "16:20",
      onInput: (e) => {
        stageParameterValue("SESSION_CLOSE_CUTOFF_UTC", e.target.value.trim());
      },
    });
    grid.appendChild(
      el(
        "div",
        { className: "wb-field" },
        el("label", { className: "wb-label", htmlFor: "quick-cutoff-input", text: "4. LSE Gap Cutoff (UTC)" }),
        cutoffInput,
        el("div", { className: "wb-kpi-sub", text: "Session flatten time (e.g. 16:20)" })
      )
    );

    card.appendChild(grid);

    // If staged changes exist, show a prominent Save bar
    const stagedCount = Object.keys(state.stagedChanges).length;
    if (stagedCount > 0) {
      const saveBar = el(
        "div",
        {
          className: "wb-pill-row",
          style: "margin-top: 1rem; padding-top: 0.75rem; border-top: 1px solid var(--border-subtle); justify-content: space-between;",
        },
        el("span", { className: "wb-badge wb-badge--amber", text: `${stagedCount} setting(s) staged for save` }),
        el(
          "div",
          { className: "wb-pill-row" },
          el(
            "button",
            {
              type: "button",
              className: "wb-btn wb-btn--sm wb-btn--ghost",
              onClick: () => {
                discardStagedDraft();
                renderQuickControls(container, readOnly);
              },
            },
            "Discard"
          ),
          el(
            "button",
            {
              type: "button",
              id: "quick-save-btn",
              className: "wb-btn wb-btn--sm wb-btn--emerald",
              disabled: readOnly || state.isCommitting,
              onClick: async () => {
                await submitConfigCommit();
                renderQuickControls(container, readOnly);
              },
            },
            state.isCommitting ? "Saving..." : "Apply & Save Settings"
          )
        )
      );
      card.appendChild(saveBar);
    }
  }

  function applyConfigViewModeVisibility(controlsRoot, drawer, readOnly) {
    const doc = getDoc();
    if (!doc) return;
    if (state.route.param) {
      state.viewMode = "advanced";
    }
    const slot = doc.getElementById("config-param-groups-slot");
    const tb = doc.getElementById("config-toolbar");
    const qc = doc.getElementById("quick-controls-card");

    if (state.viewMode === "simple") {
      renderQuickControls(controlsRoot, readOnly);
      if (drawer) drawer.hidden = true;
      if (slot) slot.hidden = true;
      if (tb) tb.hidden = true;
      if (qc) qc.hidden = false;
    } else {
      if (qc) qc.hidden = true;
      if (drawer) drawer.hidden = false;
      if (slot) slot.hidden = false;
      if (tb) tb.hidden = false;
    }
  }

  function computeConfigSignature(params, readOnly) {
    return (
      params.map((p) => p.key).join("|") +
      "::" +
      (state.route.param || "") +
      "::" +
      (state.route.gate || "") +
      "::" +
      state.configGroup +
      "::" +
      state.configSearch +
      "::" +
      (readOnly ? "ro" : "rw")
    );
  }

  function renderConfigWorkspace(forceRebuildControls) {
    const doc = getDoc();
    const root = doc && doc.getElementById("workspace-config");
    if (!root) return;

    let layout = doc.getElementById("config-layout");
    if (!layout) {
      layout = el("div", { id: "config-layout", className: "wb-config-layout" });
      root.appendChild(layout);
    }

    let controlsRoot = doc.getElementById("config-controls-root");
    if (!controlsRoot) {
      controlsRoot = el("div", { id: "config-controls-root" });
      layout.appendChild(controlsRoot);
    }

    let drawer = doc.getElementById("diff-drawer");
    if (!drawer) {
      drawer = el("aside", { id: "diff-drawer", className: "wb-diff-drawer", "aria-label": "Staged-Intent Diff Drawer" });
      layout.appendChild(drawer);
    } else if (drawer.parentNode !== layout) {
      layout.appendChild(drawer);
    }

    const params = getSchemaParameters();
    const readOnly = isReadOnlyMode();
    const schemaByKey = {};
    params.forEach((p) => {
      if (p && p.key) schemaByKey[p.key] = p;
    });

    ensureConfigToolbar(controlsRoot, readOnly);

    const focused = getFocusedElement();
    const groupsSlot = doc.getElementById("config-param-groups-slot");
    const focusInsideGroups = Boolean(focused && groupsSlot && isElementInside(focused, groupsSlot));
    const schemaSig = computeConfigSignature(params, readOnly);

    if (!forceRebuildControls && focusInsideGroups && state.lastSchemaSignature === schemaSig) {
      renderConfigDeepLinkSlot(controlsRoot, schemaByKey);
      patchConfigBadgesInPlace(params);
      applyConfigViewModeVisibility(controlsRoot, drawer, readOnly);
      return;
    }

    if (!forceRebuildControls && state.lastSchemaSignature === schemaSig && groupsSlot) {
      renderConfigDeepLinkSlot(controlsRoot, schemaByKey);
      patchConfigBadgesInPlace(params);
      applyConfigViewModeVisibility(controlsRoot, drawer, readOnly);
      return;
    }

    state.lastSchemaSignature = schemaSig;
    renderConfigDeepLinkSlot(controlsRoot, schemaByKey);
    renderConfigParamGroups(controlsRoot, params, readOnly);
    applyConfigViewModeVisibility(controlsRoot, drawer, readOnly);
  }

  function formatStat(val) {
    const num = Number(val);
    return Number.isFinite(num) ? num.toFixed(2) : "—";
  }

  function buildParameterCard(spec, readOnly) {
    const key = spec.key;
    const inputId = `cfg-input-${key}`;
    const pendingSet = new Set(getPendingRestartKeys());
    const isStaged = Object.prototype.hasOwnProperty.call(state.stagedChanges, key);
    const activeVal = getEffectiveConfigValue(key);
    const displayVal = isStaged ? state.stagedChanges[key] : activeVal;
    const isHighlighted = state.route.param === key;

    const restartBadge = spec.reload_mode === "restart_required"
      ? el("span", { className: "wb-badge wb-badge--amber", text: "restart_required" })
      : el("span", { className: "wb-badge wb-badge--emerald", text: "hot_reload" });

    const pendingBadge = el("span", {
      className: "wb-badge wb-badge--rose",
      hidden: !pendingSet.has(key),
      text: "Pending Restart",
    });

    const stagedBadge = el("span", {
      className: "wb-badge wb-badge--sky",
      hidden: !isStaged,
      text: "Staged",
    });

    const resetBtn = el("button", {
      type: "button",
      className: "wb-btn wb-btn--sm wb-btn--ghost",
      hidden: !isStaged,
      text: "Revert",
      onClick: () => {
        stageParameterValue(key, activeVal);
        if (state.paramDomCache[key] && state.paramDomCache[key].input) {
          syncControlValue(state.paramDomCache[key].input, spec, activeVal);
        }
      },
    });

    const inputEl = buildParameterInput(inputId, spec, displayVal, readOnly);

    const card = el(
      "div",
      {
        className: `wb-param-card${isStaged ? " is-staged" : ""}${isHighlighted ? " is-highlighted" : ""}`,
        dataset: { paramKey: key },
      },
      el(
        "div",
        { className: "wb-param-head" },
        el(
          "div",
          null,
          el("label", { className: "wb-label", htmlFor: inputId, text: spec.label || key }),
          el("div", { className: "wb-mono wb-muted", style: "font-size:0.72rem", text: `${key}${spec.unit ? ` (${spec.unit})` : ""}` })
        ),
        el("div", { className: "wb-param-meta" }, restartBadge, pendingBadge, stagedBadge)
      ),
      inputEl,
      spec.impact_hint ? el("p", { className: "wb-param-hint", text: spec.impact_hint }) : null,
      spec.is_dynamic && spec.screener_metric_gate
        ? el("div", {
            className: "wb-mono",
            style: "font-size:0.72rem;color:var(--sky)",
            text: `Gate: ${spec.screener_metric_gate.metric} ${spec.screener_metric_gate.op} ${spec.screener_metric_gate.threshold} (missing: ${spec.screener_metric_gate.on_missing || "reject"})`,
          })
        : null,
      el(
        "div",
        { style: "display:flex;justify-content:space-between;align-items:center;gap:0.5rem;margin-top:auto" },
        el("span", {
          className: "wb-mono wb-muted",
          style: "font-size:0.72rem",
          text: `Active: ${formatShortValue(activeVal)}${spec.min !== null && spec.min !== undefined ? ` [${spec.min}..${spec.max}]` : ""}`,
        }),
        el(
          "div",
          { className: "wb-action-row" },
          resetBtn,
          spec.is_dynamic
            ? el("button", {
                type: "button",
                className: "wb-btn wb-btn--sm wb-btn--danger",
                disabled: readOnly,
                text: "Delete",
                onClick: () => openDeleteParamModal(key),
              })
            : null
        )
      )
    );

    state.paramDomCache[key] = { card, input: inputEl, pendingBadge, stagedBadge, resetBtn };
    return card;
  }

  function buildParameterInput(inputId, spec, value, readOnly) {
    if (spec.type === "bool") {
      const cb = el("input", {
        id: inputId,
        type: "checkbox",
        disabled: readOnly,
        onChange: (ev) => stageParameterValue(spec.key, Boolean(ev.target.checked)),
      });
      cb.checked = Boolean(value);
      return el("label", { className: "wb-check-row", htmlFor: inputId }, cb, Boolean(value) ? "Enabled" : "Disabled");
    }

    if (Array.isArray(spec.allowed_values) && spec.allowed_values.length > 0) {
      const sel = el(
        "select",
        {
          id: inputId,
          className: "wb-select",
          disabled: readOnly,
          onChange: (ev) => {
            const raw = ev.target.value;
            const parsed = spec.type === "int" ? parseInt(raw, 10) : (spec.type === "float" ? Number(raw) : raw);
            stageParameterValue(spec.key, parsed);
          },
        },
        spec.allowed_values.map((av) => el("option", { value: String(av), text: String(av) }))
      );
      sel.value = String(value !== undefined && value !== null ? value : spec.default);
      return sel;
    }

    if (spec.type === "int" || spec.type === "float") {
      const numIn = el("input", {
        id: inputId,
        type: "number",
        className: "wb-input wb-input--mono",
        disabled: readOnly,
        min: spec.min !== null && spec.min !== undefined ? spec.min : undefined,
        max: spec.max !== null && spec.max !== undefined ? spec.max : undefined,
        step: spec.step !== null && spec.step !== undefined ? spec.step : (spec.type === "int" ? 1 : "any"),
        value: value !== undefined && value !== null ? String(value) : "",
        onInput: (ev) => {
          const raw = ev.target.value.trim();
          if (raw === "") return;
          const parsed = spec.type === "int" ? Number(raw) : Number(raw);
          stageParameterValue(spec.key, parsed);
        },
      });
      return numIn;
    }

    if (spec.type === "string_list") {
      const listStr = Array.isArray(value) ? value.join(", ") : "";
      return el("input", {
        id: inputId,
        type: "text",
        className: "wb-input wb-input--mono",
        disabled: readOnly,
        placeholder: "Comma-separated symbols (blank = default)",
        value: listStr,
        onInput: (ev) => {
          const raw = ev.target.value.trim();
          const parsed = raw ? raw.split(",").map((t) => t.trim().toUpperCase()).filter(Boolean) : null;
          stageParameterValue(spec.key, parsed);
        },
      });
    }

    return el("input", {
      id: inputId,
      type: "text",
      className: "wb-input wb-input--mono",
      disabled: readOnly,
      value: value !== undefined && value !== null ? String(value) : "",
      onInput: (ev) => stageParameterValue(spec.key, ev.target.value),
    });
  }

  function syncControlValue(inputEl, spec, val) {
    if (!inputEl) return;
    if (spec.type === "bool") {
      const cb = inputEl.querySelector ? inputEl.querySelector('input[type="checkbox"]') : inputEl;
      if (cb) cb.checked = Boolean(val);
    } else if (spec.type === "string_list") {
      inputEl.value = Array.isArray(val) ? val.join(", ") : "";
    } else {
      inputEl.value = val !== undefined && val !== null ? String(val) : "";
    }
  }

  function patchConfigBadgesInPlace(params) {
    const pendingSet = new Set(getPendingRestartKeys());
    params.forEach((spec) => {
      const cached = state.paramDomCache[spec.key];
      if (!cached) return;
      if (cached.pendingBadge) cached.pendingBadge.hidden = !pendingSet.has(spec.key);
      updateConfigCardStagingUI(spec.key);
      const isFocused = cached.input === getFocusedElement() || isElementInside(getFocusedElement(), cached.card);
      const isStaged = Object.prototype.hasOwnProperty.call(state.stagedChanges, spec.key);
      if (!isFocused && !isStaged) {
        syncControlValue(cached.input, spec, getEffectiveConfigValue(spec.key));
      }
    });
  }

  function renderDiffDrawer() {
    const doc = getDoc();
    const drawer = doc && doc.getElementById("diff-drawer");
    if (!drawer) return;
    const prevReasonEl = doc.getElementById("commit-reason-input");
    const savedReasonVal = prevReasonEl ? prevReasonEl.value : "";
    const reasonFocused = Boolean(prevReasonEl && getFocusedElement() === prevReasonEl);
    const selStart = reasonFocused && typeof prevReasonEl.selectionStart === "number" ? prevReasonEl.selectionStart : null;
    const selEnd = reasonFocused && typeof prevReasonEl.selectionEnd === "number" ? prevReasonEl.selectionEnd : null;
    clearNode(drawer);

    if (state.drawerCollapsedMobile) {
      drawer.classList.add("is-collapsed");
    } else {
      drawer.classList.remove("is-collapsed");
    }

    const stagedKeys = Object.keys(state.stagedChanges);
    const activeEpoch = getActiveEpochId();
    const readOnly = isReadOnlyMode();
    const unlocked = isUnlocked();
    const params = getSchemaParameters();
    const localResult = computeLocalValidationAndDiff(getActiveConfigValues(), state.stagedChanges, params);

    drawer.appendChild(
      el(
        "div",
        { className: "wb-drawer-head" },
        el(
          "div",
          null,
          el("strong", { text: "Staged-Intent Diff Drawer" }),
          " ",
          el("span", { className: "wb-badge wb-badge--sky wb-mono", text: `${stagedKeys.length} staged` })
        ),
        el("button", {
          type: "button",
          className: "wb-btn wb-btn--sm wb-drawer-toggle",
          "aria-expanded": state.drawerCollapsedMobile ? "false" : "true",
          text: state.drawerCollapsedMobile ? "Expand" : "Minimize",
          onClick: () => {
            state.drawerCollapsedMobile = !state.drawerCollapsedMobile;
            renderDiffDrawer();
          },
        })
      )
    );

    const body = el("div", { className: "wb-drawer-body" });

    // Non-destructive Remote Epoch Changed banner
    if (stagedKeys.length > 0 && state.draftBaseEpochId && state.draftBaseEpochId !== activeEpoch && !state.rebaseState) {
      body.appendChild(
        el(
          "div",
          { id: "remote-epoch-changed-banner", className: "wb-banner wb-banner--warn", style: "margin-bottom:0" },
          el(
            "div",
            { className: "wb-banner-body" },
            el("span", { className: "wb-banner-title", text: "Remote Epoch Changed" }),
            `Draft pinned to ${state.draftBaseEpochId}, while active epoch is now ${activeEpoch}. Your staged edits are preserved.`
          ),
          el(
            "div",
            { className: "wb-action-row" },
            el("button", {
              type: "button",
              className: "wb-btn wb-btn--sm wb-btn--primary",
              text: "Rebase onto Active",
              onClick: () => {
                const choices = {};
                Object.keys(state.stagedChanges).forEach((k) => { choices[k] = "mine"; });
                state.rebaseState = {
                  activeEpochId: activeEpoch,
                  remoteConfig: getActiveConfigValues(),
                  errorMessage: `Remote updated to ${activeEpoch} while editing ${state.draftBaseEpochId}.`,
                  choices,
                };
                renderDiffDrawer();
              },
            })
          )
        )
      );
    }

    // 409 Conflict Rebase Panel
    if (state.rebaseState) {
      const rb = state.rebaseState;
      const rbBox = el(
        "div",
        { id: "conflict-rebase-panel", className: "wb-rebase-box" },
        el("strong", { text: `409 Conflict Rebase (${state.draftBaseEpochId} vs ${rb.activeEpochId})` }),
        el("div", { className: "wb-muted", style: "font-size:0.75rem", text: rb.errorMessage || "Choose Keep Mine or Use Remote per key." })
      );
      const remoteValues = extractConfigValues(rb.remoteConfig);
      Object.keys(state.stagedChanges).forEach((key) => {
        const mineVal = state.stagedChanges[key];
        const remoteVal = Object.prototype.hasOwnProperty.call(remoteValues, key)
          ? remoteValues[key]
          : getEffectiveConfigValue(key);
        const choice = rb.choices[key] || "mine";
        rbBox.appendChild(
          el(
            "div",
            { className: "wb-rebase-row", dataset: { rebaseKey: key } },
            el("strong", { className: "wb-mono", text: key }),
            el("div", { className: "wb-mono", text: `Mine: ${formatShortValue(mineVal)} | Remote: ${formatShortValue(remoteVal)}` }),
            el(
              "div",
              { className: "wb-action-row" },
              el("button", {
                type: "button",
                className: choice === "mine" ? "wb-btn wb-btn--sm wb-btn--primary" : "wb-btn wb-btn--sm",
                text: "Keep Mine",
                onClick: () => {
                  rb.choices[key] = "mine";
                  renderDiffDrawer();
                },
              }),
              el("button", {
                type: "button",
                className: choice === "remote" ? "wb-btn wb-btn--sm wb-btn--emerald" : "wb-btn wb-btn--sm",
                text: "Use Remote",
                onClick: () => {
                  rb.choices[key] = "remote";
                  renderDiffDrawer();
                },
              })
            )
          )
        );
      });
      rbBox.appendChild(
        el("button", {
          type: "button",
          id: "apply-rebase-btn",
          className: "wb-btn wb-btn--sm wb-btn--primary",
          text: "Apply Rebase to Remote Epoch",
          onClick: () => applyRebaseChoices(),
        })
      );
      body.appendChild(rbBox);
    }

    if (stagedKeys.length === 0) {
      body.appendChild(
        el("div", {
          className: "wb-empty",
          text: "Edit any parameter or load a built-in profile to stage an epoch diff.",
        })
      );
      drawer.appendChild(body);
      return;
    }

    body.appendChild(
      el("div", {
        className: "wb-mono wb-muted",
        style: "font-size:0.74rem",
        text: `Base Epoch: ${state.draftBaseEpochId || activeEpoch}`,
      })
    );

    // Diff entries
    localResult.diff.forEach((d) => {
      body.appendChild(
        el(
          "div",
          { className: "wb-diff-item" },
          el(
            "div",
            { style: "display:flex;justify-content:space-between;gap:0.4rem" },
            el("span", { className: "wb-mono", style: "font-weight:700", text: d.key }),
            el("span", { className: d.reload_mode === "restart_required" ? "wb-badge wb-badge--amber" : "wb-badge wb-badge--emerald", text: d.reload_mode })
          ),
          el("div", { className: "wb-mono", style: "margin-top:0.2rem", text: `${formatShortValue(d.old_value)} -> ${formatShortValue(d.new_value)}` })
        )
      );
    });

    // Local + server validation errors
    const serverErrors = (state.previewData && Array.isArray(state.previewData.validation_errors))
      ? state.previewData.validation_errors
      : [];
    const allErrors = serverErrors.length > 0 ? serverErrors : localResult.errors;
    if (allErrors.length > 0) {
      body.appendChild(
        el(
          "div",
          { className: "wb-banner wb-banner--danger", style: "margin-bottom:0;padding:0.65rem" },
          el("strong", { text: "Validation Errors:" }),
          el("ul", { className: "wb-banner-list" }, allErrors.map((e) => el("li", { text: `${e.key}: ${e.message}` })))
        )
      );
    }

    // Non-simulatable downstream risk/TSL/off-hours warning
    const nonSim = (state.previewData && Array.isArray(state.previewData.non_simulatable_keys) && state.previewData.non_simulatable_keys.length > 0)
      ? state.previewData.non_simulatable_keys
      : localResult.nonSimulatable.map((x) => x.key);
    if (nonSim.length > 0) {
      body.appendChild(
        el("div", {
          className: "wb-banner wb-banner--warn",
          style: "margin-bottom:0;padding:0.6rem",
          text: `Non-simulatable downstream changes (${nonSim.join(", ")}): risk, TSL, session, and runtime edits are validated for bounds/invariants but not replayed on candidate-gate telemetry.`,
        })
      );
    }

    // Locked vs Unlocked server preview section
    if (!unlocked && !readOnly) {
      body.appendChild(
        el(
          "div",
          { id: "diff-locked-hint", className: "wb-callout" },
          el("div", { text: "Locked mode: showing local bounds & cross-field checks only. Unlock to run server Phase-2 gate impact preview." }),
          el("button", {
            type: "button",
            className: "wb-btn wb-btn--sm wb-btn--primary",
            style: "margin-top:0.45rem",
            text: "Unlock to Preview Server Impact",
            onClick: () => openAuthModal(),
          })
        )
      );
    } else if (state.previewLoading) {
      body.appendChild(el("div", { className: "wb-muted wb-mono", style: "font-size:0.75rem", text: "Running server preview..." }));
    } else if (state.previewError) {
      body.appendChild(el("div", { className: "wb-banner wb-banner--danger", style: "margin-bottom:0;padding:0.55rem", text: state.previewError }));
    } else if (state.previewData) {
      const impact = state.previewData.gate_impact || state.previewData.screener_impact || state.previewData.phase2_impact || state.previewData.simulation;
      if (impact && typeof impact === "object") {
        const impactText = impact.available === false
          ? (impact.reason || "Off-hours / empty indicator cache: Phase-2 gate simulation skipped.")
          : `Phase-2 Gate Impact: passed ${impact.baseline_passed ?? impact.before_count ?? "?"} -> ${impact.candidate_passed ?? impact.after_count ?? "?"} (delta ${impact.delta ?? 0})`;
        body.appendChild(el("div", { id: "diff-server-impact", className: "wb-callout", text: impactText }));
      }
    }

    const openBehindCount = (state.data.active_signals || []).length;
    if (openBehindCount > 0) {
      body.appendChild(
        el("div", {
          className: "wb-muted",
          style: "font-size:0.74rem",
          text: `Note: ${openBehindCount} open position(s) remain pinned to their entry epoch until explicitly Re-Epoched.`,
        })
      );
    }

    const reasonInput = el("input", {
      id: "commit-reason-input",
      type: "text",
      className: "wb-input",
      disabled: readOnly || state.isCommitting,
      placeholder: "Commit reason (optional)...",
    });
    if (savedReasonVal) {
      reasonInput.value = savedReasonVal;
    }

    const commitBtn = el("button", {
      type: "button",
      id: "commit-config-btn",
      className: "wb-btn wb-btn--primary",
      disabled: readOnly || state.isCommitting || localResult.errors.length > 0 || serverErrors.length > 0 || Boolean(state.rebaseState),
      text: state.isCommitting ? "Committing..." : "Commit Epoch",
      onClick: () => submitConfigCommit(reasonInput.value.trim()),
    });

    const discardBtn = el("button", {
      type: "button",
      id: "discard-draft-btn",
      className: "wb-btn wb-btn--ghost",
      text: "Discard Draft",
      onClick: discardStagedDraft,
    });

    body.appendChild(
      el(
        "div",
        { className: "wb-field" },
        el("label", { className: "wb-label", htmlFor: "commit-reason-input", text: "Commit Reason" }),
        reasonInput
      )
    );
    body.appendChild(el("div", { className: "wb-action-row" }, commitBtn, discardBtn));

    drawer.appendChild(body);
    if (reasonFocused && typeof reasonInput.focus === "function") {
      try {
        reasonInput.focus();
        if (selStart !== null && selEnd !== null && typeof reasonInput.setSelectionRange === "function") {
          reasonInput.setSelectionRange(selStart, selEnd);
        }
      } catch (_) {
        // ignore focus restore failures
      }
    }
  }

  /* ── Audit & Regime (Disabled & Deprecated) ───────────────────────────── */

  function renderAuditWorkspace() {
    const doc = getDoc();
    const root = doc && doc.getElementById("workspace-audit");
    if (root) clearNode(root);
  }

  /* ── Timers, Countdown & SSE Engine ────────────────────────────────────── */

  function updateCountdown() {
    const doc = getDoc();
    if (!doc) return;
    const countdownEl = doc.getElementById("cutoff-countdown");
    const mcCutoffEl = doc.getElementById("cutoff-countdown-mc");
    const scanCountdownEl = doc.getElementById("next-scan-countdown");
    const monCountdownEl = doc.getElementById("next-monitor-countdown");

    const cfg = getActiveConfigValues();
    const rawCutoff = String(cfg.SESSION_CLOSE_CUTOFF_UTC || "16:20").trim();
    const m = HHMM_RE.exec(rawCutoff);
    const ch = m ? parseInt(m[1], 10) : 16;
    const cm = m ? parseInt(m[2], 10) : 20;

    const now = new Date();
    const nowUtc = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate(), now.getUTCHours(), now.getUTCMinutes(), now.getUTCSeconds());
    let targetUtc = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate(), ch, cm, 0);
    let diff = targetUtc - nowUtc;
    let cutoffStr = "--:--:--";
    if (diff < 0) {
      if (diff > -600000) {
        cutoffStr = "FLATTENING";
      } else {
        targetUtc += 86400000;
        diff = targetUtc - nowUtc;
        const hours = Math.floor(diff / 3600000);
        const mins = Math.floor((diff % 3600000) / 60000);
        const secs = Math.floor((diff % 60000) / 1000);
        const pad = (n) => String(n).padStart(2, "0");
        cutoffStr = `${pad(hours)}:${pad(mins)}:${pad(secs)}`;
      }
    } else {
      const hours = Math.floor(diff / 3600000);
      const mins = Math.floor((diff % 3600000) / 60000);
      const secs = Math.floor((diff % 60000) / 1000);
      const pad = (n) => String(n).padStart(2, "0");
      cutoffStr = `${pad(hours)}:${pad(mins)}:${pad(secs)}`;
    }

    if (countdownEl) countdownEl.textContent = cutoffStr;
    if (mcCutoffEl) mcCutoffEl.textContent = cutoffStr;

    // Next Scan (5m = 300s) and Next Monitor (30s) timers
    const nowSec = Math.floor(Date.now() / 1000);
    const remScan = 300 - (nowSec % 300);
    const remMon = 30 - (nowSec % 30);
    const pad = (n) => String(n).padStart(2, "0");

    if (scanCountdownEl) {
      const sm = Math.floor(remScan / 60);
      const ss = remScan % 60;
      scanCountdownEl.textContent = `${pad(sm)}:${pad(ss)}`;
    }
    if (monCountdownEl) {
      monCountdownEl.textContent = `${remMon}s`;
    }
  }

  async function fetchCatchUpData(isManual) {
    if (isFileProtocol()) return;
    const doc = getDoc();
    const btn = isManual && doc ? doc.getElementById("refresh-btn") : null;
    if (btn) btn.textContent = "Syncing...";
    const res = await apiRequest(`/api/data?_t=${Date.now()}`, { method: "GET" });
    if (res.ok && res.data) {
      state.sse.lastSnapshotAt = Date.now();
      state.sse.lastHeartbeatAt = Date.now();
      applySnapshotData(res.data);
    }
    if (btn) btn.textContent = "Refresh Now";
  }

  function connectSSE() {
    if (state.sse.reconnectTimer) {
      clearTimeout(state.sse.reconnectTimer);
      state.sse.reconnectTimer = null;
    }
    if (isFileProtocol() || typeof EventSource === "undefined") {
      state.sse.status = "offline";
      updateSseHealthBadge();
      return;
    }
    if (state.sse.source) {
      try {
        state.sse.source.close();
      } catch (_) {
        // ignore
      }
      state.sse.source = null;
    }
    state.sse.status = "connecting";
    updateSseHealthBadge();

    try {
      const es = new EventSource("/stream");
      state.sse.source = es;

      es.onopen = () => {
        const reconnected = state.sse.hasDisconnected;
        state.sse.status = "connected";
        state.sse.backoffMs = SSE_MIN_BACKOFF_MS;
        state.sse.lastHeartbeatAt = Date.now();
        updateSseHealthBadge();
        if (reconnected) {
          state.sse.hasDisconnected = false;
          fetchCatchUpData(false);
        }
      };

      const handlePayloadEvent = (ev) => {
        state.sse.lastSnapshotAt = Date.now();
        state.sse.lastHeartbeatAt = Date.now();
        try {
          const parsed = JSON.parse(ev.data);
          applySnapshotData(parsed);
        } catch (_) {
          // ignore malformed frames
        }
      };

      es.addEventListener("snapshot", handlePayloadEvent);
      es.addEventListener("dashboard_update", handlePayloadEvent);
      es.addEventListener("pipeline_progress", (ev) => {
        state.sse.lastHeartbeatAt = Date.now();
        try {
          const parsed = JSON.parse(ev.data);
          state.pipelineProgress = parsed;
          const doc = getDoc();
          const mc = doc && (doc.getElementById("mission-control-panel") || doc.getElementById("funnel-panel"));
          if (mc) renderMissionControlCard(mc);
        } catch (_) {
          // ignore malformed frames
        }
      });
      es.addEventListener("config_epoch_changed", (ev) => {
        state.sse.lastHeartbeatAt = Date.now();
        try {
          const parsed = JSON.parse(ev.data);
          if (parsed && typeof parsed === "object" && parsed.dashboard && typeof parsed.dashboard === "object") {
            applySnapshotData(parsed.dashboard);
            return;
          }
          if (parsed && typeof parsed === "object" && parsed.config) {
            applySnapshotData(parsed);
            return;
          }
          if (parsed && typeof parsed === "object" && (parsed.config_epoch_id || parsed.active_epoch_id || parsed.epoch)) {
            state.data.config_epoch_id = String(parsed.config_epoch_id || parsed.active_epoch_id || parsed.epoch);
            renderHeader();
            if (state.route.tab === "config") renderDiffDrawer();
          }
          fetchCatchUpData(false);
        } catch (_) {
          fetchCatchUpData(false);
        }
      });
      es.addEventListener("heartbeat", (ev) => {
        state.sse.lastHeartbeatAt = Date.now();
        try {
          const hb = JSON.parse(ev.data || "{}");
          if (hb && hb.epoch && hb.epoch !== getActiveEpochId()) {
            fetchCatchUpData(false);
          }
        } catch (_) {
          // ignore
        }
        updateSseHealthBadge();
      });

      es.onerror = () => {
        try {
          es.close();
        } catch (_) {
          // ignore
        }
        state.sse.hasDisconnected = true;
        state.sse.status = "reconnecting";
        updateSseHealthBadge();
        const jitter = Math.floor(Math.random() * 400);
        const delay = Math.min(SSE_MAX_BACKOFF_MS, state.sse.backoffMs) + jitter;
        state.sse.backoffMs = Math.min(SSE_MAX_BACKOFF_MS, state.sse.backoffMs * 2);
        if (state.sse.reconnectTimer) clearTimeout(state.sse.reconnectTimer);
        state.sse.reconnectTimer = setTimeout(connectSSE, delay);
      };
    } catch (_) {
      state.sse.status = "offline";
      updateSseHealthBadge();
    }
  }

  /* ── Shell Mounting & Hydration ────────────────────────────────────────── */

  function ensureMountShell(mountRoot) {
    const doc = getDoc();
    let app = doc.getElementById("workbench-app") || mountRoot || doc.getElementById("app") || doc.body;
    if (!app) return null;
    if (!app.id) app.id = "workbench-app";
    if (!app.classList.contains("wb-app")) app.classList.add("wb-app");

    const ensureNode = (id, factory) => {
      let existing = doc.getElementById(id);
      if (!existing) {
        existing = factory();
        app.appendChild(existing);
      }
      return existing;
    };

    ensureNode("readonly-banner", () => el("div", { id: "readonly-banner", className: "wb-banner wb-banner--warn", hidden: true }));
    ensureNode("quarantine-banner", () => el("div", { id: "quarantine-banner", className: "wb-banner wb-banner--danger", hidden: true }));
    ensureNode("halted-banner", () => el("div", { id: "halted-banner", className: "wb-banner wb-banner--danger", hidden: true }));
    ensureNode("workbench-header", () => el("header", { id: "workbench-header", className: "wb-header" }));

    const tabsEl = ensureNode("workbench-tabs", () => el("nav", { id: "workbench-tabs", className: "wb-tabs", role: "tablist", "aria-label": "Control Plane Workspaces" }));
    if (tabsEl.children.length === 0) {
      const tabDefs = [
        ["monitor", "Live Monitor"],
        ["config", "Configuration"],
      ];
      tabDefs.forEach(([id, label], idx) => {
        const btn = el("button", {
          type: "button",
          id: `tab-${id}`,
          className: "wb-tab",
          role: "tab",
          "aria-selected": idx === 0 ? "true" : "false",
          "aria-controls": `workspace-${id}`,
          tabindex: idx === 0 ? "0" : "-1",
          text: label,
          onClick: () => navigateTo(id),
          onKeyDown: (ev) => {
            if (ev.key === "ArrowRight" || ev.key === "ArrowLeft") {
              ev.preventDefault();
              const order = ["monitor", "config"];
              const curIdx = order.indexOf(id);
              const nextIdx = ev.key === "ArrowRight" ? (curIdx + 1) % order.length : (curIdx - 1 + order.length) % order.length;
              const nextId = order[nextIdx];
              navigateTo(nextId);
              const nextBtn = doc.getElementById(`tab-${nextId}`);
              if (nextBtn && typeof nextBtn.focus === "function") nextBtn.focus();
            }
          },
        });
        tabsEl.appendChild(btn);
      });
    }

    const mainEl = ensureNode("workbench-main", () => el("main", { id: "workbench-main", className: "wb-main" }));
    const ensureWorkspace = (name, hiddenByDefault) => {
      let ws = doc.getElementById(`workspace-${name}`);
      if (!ws) {
        ws = el("section", {
          id: `workspace-${name}`,
          className: name === "config" ? "wb-workspace wb-config-workspace--has-padding" : "wb-workspace",
          role: "tabpanel",
          "aria-labelledby": `tab-${name}`,
          hidden: hiddenByDefault,
        });
        mainEl.appendChild(ws);
      }
      if (!doc.getElementById(name)) {
        ws.appendChild(el("div", { id: name, className: "wb-anchor" }));
      }
      return ws;
    };

    const monitorWs = ensureWorkspace("monitor", false);
    if (!doc.getElementById("equity-chart-svg")) {
      const eqCard = el(
        "section",
        { id: "monitor-equity-shell", className: "wb-card" },
        el(
          "div",
          { className: "wb-equity-header" },
          el(
            "div",
            null,
            el("div", { className: "wb-kpi-label", text: "Portfolio Equity Curve & Session Trajectory" }),
            el(
              "div",
              { style: "margin-top:0.25rem" },
              el("span", { id: "equity-val-current", className: "wb-equity-value", text: "€0.00" }),
              el("span", { id: "equity-val-delta", className: "wb-equity-delta", text: "+€0.00 (+0.00%)" })
            )
          ),
          el(
            "div",
            { className: "wb-equity-stats" },
            el("span", null, "Start: ", el("strong", { id: "equity-val-start", text: "€0.00" })),
            el("span", null, "High: ", el("strong", { id: "equity-val-high", text: "—" })),
            el("span", null, "Low: ", el("strong", { id: "equity-val-low", text: "—" }))
          )
        ),
        el(
          "div",
          { className: "wb-equity-wrap" },
          svgEl(
            "svg",
            { id: "equity-chart-svg", className: "wb-equity-svg", preserveAspectRatio: "none", viewBox: "0 0 800 100" },
            svgEl(
              "defs",
              null,
              svgEl(
                "linearGradient",
                { id: "equity-gradient-emerald", x1: "0", y1: "0", x2: "0", y2: "1" },
                svgEl("stop", { offset: "0%", "stop-color": "#10b981", "stop-opacity": "0.35" }),
                svgEl("stop", { offset: "100%", "stop-color": "#10b981", "stop-opacity": "0.0" })
              ),
              svgEl(
                "linearGradient",
                { id: "equity-gradient-rose", x1: "0", y1: "0", x2: "0", y2: "1" },
                svgEl("stop", { offset: "0%", "stop-color": "#f43f5e", "stop-opacity": "0.35" }),
                svgEl("stop", { offset: "100%", "stop-color": "#f43f5e", "stop-opacity": "0.0" })
              )
            ),
            svgEl("g", { id: "equity-chart-group" })
          )
        )
      );
      monitorWs.appendChild(eqCard);
    }

    ensureWorkspace("config", true);
    ensureWorkspace("audit", true);

    ensureNode("modal-root", () => el("div", { id: "modal-root" }));
    ensureNode("toast-root", () => el("div", { id: "toast-root", className: "wb-toast-root", "aria-live": "polite", "aria-atomic": "true" }));

    state.root = app;
    return app;
  }

  function renderAllVisible() {
    renderBanners();
    renderHeader();
    applyRouteVisibility();
  }

  function initWorkbench(mountRoot) {
    const doc = getDoc();
    const win = getWin();
    if (!doc) return;

    state.data = Object.assign({}, readInitialData());
    state.route = parseHashRoute();
    state.sse.lastSnapshotAt = Date.now();
    state.sse.lastHeartbeatAt = Date.now();

    ensureMountShell(mountRoot);
    renderAllVisible();

    if (!state.initialized) {
      state.initialized = true;
      if (win && win.addEventListener) {
        state.hashChangeHandler = () => {
          state.route = parseHashRoute();
          applyRouteVisibility();
        };
        win.addEventListener("hashchange", state.hashChangeHandler);
      }
      // 1s UI timer for session-close countdown and >25s SSE heartbeat staleness badge (zero network polling)
      state.uiTimer = setInterval(() => {
        updateCountdown();
        updateSseHealthBadge();
      }, 1000);
      connectSSE();
    }
  }

  function destroyWorkbench() {
    const win = getWin();
    if (state.hashChangeHandler && win && win.removeEventListener) {
      win.removeEventListener("hashchange", state.hashChangeHandler);
      state.hashChangeHandler = null;
    }
    if (state.uiTimer) {
      clearInterval(state.uiTimer);
      state.uiTimer = null;
    }
    if (state.sse.reconnectTimer) {
      clearTimeout(state.sse.reconnectTimer);
      state.sse.reconnectTimer = null;
    }
    if (state.previewTimer) {
      clearTimeout(state.previewTimer);
      state.previewTimer = null;
    }
    if (state.sse.source) {
      try {
        state.sse.source.close();
      } catch (_) {
        // ignore
      }
      state.sse.source = null;
    }
    state.initialized = false;
  }

  // Expose compatibility globals and modular API
  const api = {
    initWorkbench,
    destroyWorkbench,
    renderEquityChart,
    applySnapshotData,
    stageParameterValue,
    discardStagedDraft,
    computeLocalValidationAndDiff,
    submitConfigCommit,
    applyRebaseChoices,
    loadBuiltinProfileIntoDraft,
    navigateTo,
    parseHashRoute,
    openAuthModal,
    openQuickSettingsDrawer,
    toggleQuickSettingsDrawer,
    openPanicModal,
    openResumeModal,
    openAddParameterModal,
    openDeleteParamModal,
    openReEpochModal,
    openReEpochAllModal,
    fetchCatchUpData,
    connectSSE,
    getState: () => state,
    TOKEN_STORAGE_KEY,
  };

  if (globalScope) {
    globalScope.renderEquityChart = renderEquityChart;
    globalScope.QuantWorkbench = api;
    globalScope.filterSignals = function (direction) {
      state.directionFilter = direction || "ALL";
      const doc = getDoc();
      const sec = doc && doc.getElementById("active-trades-section");
      if (sec) renderPositionsSection(sec);
    };
  }

  // Auto-mount in browser context
  const doc = getDoc();
  if (doc && typeof doc.addEventListener === "function") {
    if (doc.readyState === "loading") {
      doc.addEventListener("DOMContentLoaded", () => initWorkbench());
    } else if (doc.getElementById("workbench-app") || doc.getElementById("app") || doc.getElementById("equity-chart-svg")) {
      initWorkbench();
    }
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
