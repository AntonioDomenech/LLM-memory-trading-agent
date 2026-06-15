import React, { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertCircle,
  BarChart3,
  BookOpen,
  Brain,
  CheckCircle2,
  ChevronRight,
  Clock,
  CirclePause,
  CirclePlay,
  Database,
  Eye,
  Info,
  KeyRound,
  Layers3,
  LineChart,
  Pause,
  Play,
  RefreshCw,
  Save,
  Settings,
  ShieldCheck,
  Square,
  Zap,
} from "lucide-react";

const API_BASE = "http://127.0.0.1:8000";

const emptyConfig = {
  benchmark: {
    mode: "balanced_50_portfolio",
    run_preset: "balanced_50_mini",
    symbol: "AAPL",
    company_name: "Apple",
    selected_symbols: [],
    start_date: "2025-01-02",
    end_date: "2025-03-31",
    train_start: "2024-12-02",
    train_end: "2024-12-09",
    test_start: "2025-01-02",
    test_end: "2025-01-08",
    max_train_days: 5,
    max_test_days: 5,
    historical_cadence: "daily",
    fill_timing: "next_open",
    live_frequency: "hourly",
    model: "",
    endpoint: "responses",
    initial_cash: 1000,
    max_days: 20,
    allow_short: true,
    max_leverage: 1,
    max_gross_exposure: 1,
    slippage_bps: 5,
    commission_per_trade: 0,
    commission_per_share: 0,
    temperature: 0,
    max_output_tokens: 900,
    use_cached_llm: true,
    stage1_chunk_size: 10,
    max_news_per_symbol: 2,
    memory_mode: "deterministic_market_cases",
    memory_retrieval: "deterministic_similarity",
    deterministic_memory_per_symbol: 1,
    deterministic_memory_max_items: 50,
    memory_k_neighbors: 50,
    memory_examples_per_symbol: 2,
    prompt_detail_level: "compact",
    embedding_provider: "local",
    decision_process: "two_stage_llm",
    strict_preflight: true,
    require_paid_micro_pilot: true,
    max_nonzero_positions: 12,
    max_daily_turnover: 0.2,
    turnover_edge_multiplier: 3,
    invalid_run_abort_count: 3,
    invalid_run_abort_rate: 0.05,
    macro_policy: "omit_if_missing",
    news_policy: "real_titles_or_aggregate_events",
    data_sources: {
      news_sources: ["gdelt"],
      max_news_per_day: 8,
      include_sec_fundamentals: true,
      include_fred_macro: false,
      include_index_context: true,
      index_symbols: ["SPY", "QQQ", "IWM", "^VIX", "^TNX"],
      rss_feeds: [],
    },
  },
  secrets: {
    openai_api_key: "",
    openai_base_url: "https://api.openai.com/v1",
    openai_embedding_model: "text-embedding-3-small",
    marketaux_key: "",
    newsapi_key: "",
    finnhub_key: "",
    fred_api_key: "",
    sec_user_agent: "",
  },
  secret_status: {},
};

const help = {
  apiKey: {
    title: "OpenAI API key",
    body: "This lets the app call the selected model. It stays in your local ignored config file.",
    more: "Without this key you can still run dry pilots, but real benchmark decisions will be placeholders.",
  },
  model: {
    title: "Model",
    body: "The model is the portfolio manager being tested. Different models can build different memories and results.",
    more: "Use Refresh models after saving your API key. For fair comparisons, run models on the same preset and dates.",
  },
  preset: {
    title: "Run preset",
    body: "Presets change scope and cost. Mini proves mechanics. Budget Official is the recommended affordable benchmark.",
    more: "Budget Official builds deterministic memory from 2000-2024 and only calls the model for 2025 decisions.",
  },
  mode: {
    title: "Benchmark mode",
    body: "Single-stock is easier to debug. Balanced 50 is the official portfolio benchmark.",
    more: "Balanced 50 asks the model to compare stocks and allocate one portfolio instead of judging one stock alone.",
  },
  trainDates: {
    title: "Training replay dates",
    body: "The historical period used to build deterministic market memory.",
    more: "No model calls are made for this phase in Budget Official mode. The warehouse computes cases and later outcomes locally.",
  },
  testDates: {
    title: "Test dates",
    body: "The period used to judge whether the trained model can make money without future leakage.",
    more: "Default official split is 2000-2024 training and 2025 testing.",
  },
  dayCaps: {
    title: "Day caps",
    body: "Limits how many trading days are used. Set 0 for uncapped full replay.",
    more: "A mini pilot uses 5 training days and 5 test days. A full official run can make many thousands of calls.",
  },
  initialCash: {
    title: "Initial cash",
    body: "The fake starting money in the paper portfolio.",
    more: "This does not affect model intelligence, but it affects share counts and dollar P&L.",
  },
  shorting: {
    title: "Shorting",
    body: "Allows the model to bet that a stock will fall.",
    more: "Shorting can improve expression but adds risk. The simulator logs blocked shorts if this is disabled.",
  },
  exposure: {
    title: "Max gross exposure",
    body: "The total absolute portfolio weight allowed. 1.0 means no leverage.",
    more: "Example: 60% long plus 40% short equals 100% gross exposure.",
  },
  slippage: {
    title: "Slippage",
    body: "A small simulated fill penalty, measured in basis points.",
    more: "5 bps means a buy fills about 0.05% above the reference price and a sell below it.",
  },
  fees: {
    title: "Fees",
    body: "Simulated trading costs. Keep at 0 for a clean model benchmark or raise for realism.",
    more: "Fees are mechanical; they do not change the model decision, but they reduce portfolio value.",
  },
  chunk: {
    title: "Stage 1 chunk size",
    body: "How many stocks the analyst pass reviews per model call.",
    more: "Smaller chunks give more room per stock. Larger chunks reduce call count.",
  },
  memory: {
    title: "Memory retrieval",
    body: "How the model sees history before making a new decision.",
    more: "Deterministic similarity retrieves historical market cases and later outcomes from the warehouse without model-written lessons.",
  },
  embeddings: {
    title: "Embedding mode",
    body: "Legacy semantic memory setting.",
    more: "The current budget benchmark uses deterministic similarity, so embeddings are not required.",
  },
  temperature: {
    title: "Temperature",
    body: "Controls randomness in model answers. 0 is most repeatable.",
    more: "For benchmarks, 0 is usually best because it improves reproducibility.",
  },
  outputTokens: {
    title: "Output tokens",
    body: "Maximum response length for each model call.",
    more: "The default is now compact. Raise this only if the model returns truncated JSON.",
  },
  cache: {
    title: "LLM cache",
    body: "Reuses identical model calls from local disk.",
    more: "Caching saves tokens and makes reruns faster, but disable it if you want a fresh model response every time.",
  },
  preflight: {
    title: "Preflight",
    body: "Checks data, memory, macro, news, prompts, and estimated call count before a paid official run can start.",
    more: "A failed preflight blocks the run so you do not spend tokens on known-bad inputs.",
  },
  microPilot: {
    title: "Paid micro-pilot",
    body: "Requires a clean low-cost 5-day paid run before launching an uncapped official benchmark.",
    more: "This catches broken JSON, invalid allocations, and runaway turnover while the token bill is still tiny.",
  },
  maxPositions: {
    title: "Max positions",
    body: "Limits how many non-zero stock weights the model can hold at once.",
    more: "This keeps the portfolio interpretable. The simulator rejects invalid outputs instead of scaling them.",
  },
  turnover: {
    title: "Daily turnover limit",
    body: "Maximum portfolio weight the model can change in one day without proving enough expected edge.",
    more: "20% means moving from 10% Apple to 20% Apple counts as 10% turnover. High turnover must beat estimated trading cost by the configured multiplier.",
  },
  edgeMultiplier: {
    title: "Cost edge multiplier",
    body: "How much expected edge is required when a trade exceeds the daily turnover limit.",
    more: "A value of 3 means the model must claim expected edge at least three times estimated slippage cost.",
  },
  memoryNeighbors: {
    title: "Memory neighbors",
    body: "How many similar historical cases are aggregated internally per symbol.",
    more: "The prompt sees compact statistics and only a few examples, reducing noise while preserving historical evidence.",
  },
  macroPolicy: {
    title: "Macro policy",
    body: "Controls what happens when FRED macro data is unavailable.",
    more: "Omit if missing prevents the model from citing rates, inflation, or GDP when those values are not actually available.",
  },
  newsPolicy: {
    title: "News policy",
    body: "Controls how GDELT news/event rows are sent to the model.",
    more: "Synthetic event labels are aggregated as event features and never passed as fake article headlines.",
  },
  dataSources: {
    title: "News sources",
    body: "Sources used to collect context. GDELT is the default no-cost historical backbone.",
    more: "Optional providers can improve coverage when keys are configured, but the benchmark records source status.",
  },
  live: {
    title: "Live mode",
    body: "Runs a local hourly paper benchmark during regular US market hours.",
    more: "It uses fresh market snapshots when available and never connects to a brokerage account. Outside market hours it waits instead of making stale decisions.",
  },
};

const sourceHelp = {
  gdelt: {
    title: "GDELT",
    body: "Free global news metadata and headlines. This is the default historical backbone.",
    more: "Best for the under-10 EUR budget. It is broad and resumable, but titles/metadata are less clean than paid feeds.",
  },
  marketaux: {
    title: "Marketaux",
    body: "Optional market-news API with ticker tagging when you provide a key.",
    more: "Useful for live or recent news quality. Free tiers are limited, so the benchmark records source coverage.",
  },
  finnhub: {
    title: "Finnhub",
    body: "Optional financial news source for company-specific articles.",
    more: "Can improve coverage for recent company news if your free plan has enough allowance.",
  },
  newsapi: {
    title: "NewsAPI",
    body: "Optional general news API that can search ticker and company names.",
    more: "Useful as a fallback source, but free historical depth and commercial terms are limited.",
  },
  rss: {
    title: "RSS",
    body: "Optional custom feeds you control, filtered by ticker and company name.",
    more: "Good for adding trusted free sources later without changing the benchmark engine.",
  },
};

const presetInfo = {
  balanced_50_mini: {
    title: "Balanced 50 Mini Pilot",
    subtitle: "Default first run",
    body: "A tiny official-mode run that proves the two-stage portfolio, memory, and result screens work.",
    patch: {
      mode: "balanced_50_portfolio",
      run_preset: "balanced_50_mini",
      train_start: "2024-12-02",
      train_end: "2024-12-09",
      test_start: "2025-01-02",
      test_end: "2025-01-08",
      max_train_days: 5,
      max_test_days: 5,
      initial_cash: 1000,
      allow_short: true,
      max_gross_exposure: 1,
      max_output_tokens: 900,
      max_news_per_symbol: 2,
      stage1_chunk_size: 25,
      memory_mode: "deterministic_market_cases",
      memory_retrieval: "deterministic_similarity",
      memory_k_neighbors: 50,
      memory_examples_per_symbol: 2,
      strict_preflight: true,
      require_paid_micro_pilot: true,
      max_nonzero_positions: 12,
      max_daily_turnover: 0.2,
      turnover_edge_multiplier: 3,
      invalid_run_abort_count: 3,
      invalid_run_abort_rate: 0.05,
      macro_policy: "omit_if_missing",
      news_policy: "real_titles_or_aggregate_events",
      prompt_detail_level: "compact",
    },
  },
  single_stock_diagnostic: {
    title: "Single-stock Diagnostic",
    subtitle: "Cheapest debugging mode",
    body: "Runs one stock so prompts, data quality, memory, and decisions are easier to inspect.",
    patch: {
      mode: "single_stock",
      run_preset: "single_stock_diagnostic",
      symbol: "AAPL",
      company_name: "Apple",
      train_start: "2024-12-02",
      train_end: "2024-12-09",
      test_start: "2025-01-02",
      test_end: "2025-01-08",
      max_train_days: 5,
      max_test_days: 5,
      initial_cash: 1000,
      max_output_tokens: 900,
      max_news_per_symbol: 2,
      stage1_chunk_size: 25,
      memory_mode: "deterministic_market_cases",
      memory_retrieval: "deterministic_similarity",
      memory_k_neighbors: 50,
      memory_examples_per_symbol: 2,
      strict_preflight: true,
      require_paid_micro_pilot: false,
      max_nonzero_positions: 12,
      max_daily_turnover: 0.2,
      turnover_edge_multiplier: 3,
      invalid_run_abort_count: 3,
      invalid_run_abort_rate: 0.05,
      macro_policy: "omit_if_missing",
      news_policy: "real_titles_or_aggregate_events",
      prompt_detail_level: "compact",
    },
  },
  budget_official: {
    title: "Budget Official",
    subtitle: "Recommended full test",
    body: "Uses all 2000-2024 history as deterministic memory, then calls the model only for 2025 decisions.",
    patch: {
      mode: "balanced_50_portfolio",
      run_preset: "budget_official",
      train_start: "2000-01-01",
      train_end: "2024-12-31",
      test_start: "2025-01-01",
      test_end: "2025-12-31",
      max_train_days: 0,
      max_test_days: 0,
      initial_cash: 1000,
      allow_short: true,
      max_gross_exposure: 1,
      max_output_tokens: 900,
      max_news_per_symbol: 2,
      stage1_chunk_size: 25,
      memory_mode: "deterministic_market_cases",
      memory_retrieval: "deterministic_similarity",
      deterministic_memory_per_symbol: 1,
      deterministic_memory_max_items: 50,
      memory_k_neighbors: 50,
      memory_examples_per_symbol: 2,
      strict_preflight: true,
      require_paid_micro_pilot: true,
      max_nonzero_positions: 12,
      max_daily_turnover: 0.2,
      turnover_edge_multiplier: 3,
      invalid_run_abort_count: 3,
      invalid_run_abort_rate: 0.05,
      macro_policy: "omit_if_missing",
      news_policy: "real_titles_or_aggregate_events",
      prompt_detail_level: "compact",
    },
  },
  full_official: {
    title: "Full Official Benchmark",
    subtitle: "Same dates, compact memory",
    body: "Uses the official 2000-2024 train and 2025 test split with deterministic memory and compact prompts.",
    patch: {
      mode: "balanced_50_portfolio",
      run_preset: "full_official",
      train_start: "2000-01-01",
      train_end: "2024-12-31",
      test_start: "2025-01-01",
      test_end: "2025-12-31",
      max_train_days: 0,
      max_test_days: 0,
      initial_cash: 1000,
      allow_short: true,
      max_gross_exposure: 1,
      max_output_tokens: 900,
      max_news_per_symbol: 2,
      stage1_chunk_size: 25,
      memory_mode: "deterministic_market_cases",
      memory_retrieval: "deterministic_similarity",
      deterministic_memory_per_symbol: 1,
      deterministic_memory_max_items: 50,
      memory_k_neighbors: 50,
      memory_examples_per_symbol: 2,
      strict_preflight: true,
      require_paid_micro_pilot: true,
      max_nonzero_positions: 12,
      max_daily_turnover: 0.2,
      turnover_edge_multiplier: 3,
      invalid_run_abort_count: 3,
      invalid_run_abort_rate: 0.05,
      macro_policy: "omit_if_missing",
      news_policy: "real_titles_or_aggregate_events",
      prompt_detail_level: "compact",
    },
  },
};

function deepMerge(base, incoming) {
  if (!incoming || typeof incoming !== "object") return base;
  const output = Array.isArray(base) ? [...base] : { ...base };
  for (const [key, value] of Object.entries(incoming)) {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      output[key] = deepMerge(base?.[key] || {}, value);
    } else {
      output[key] = value;
    }
  }
  return output;
}

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || data.error || "Request failed");
  return data;
}

function InfoPopover({ item }) {
  if (!item) return null;
  return (
    <span className="info-popover">
      <button type="button" aria-label={`Explain ${item.title}`}>
        <Info size={14} />
      </button>
      <span className="popover-panel" role="tooltip">
        <strong>{item.title}</strong>
        <span>{item.body}</span>
        <details>
          <summary>More</summary>
          <p>{item.more}</p>
        </details>
      </span>
    </span>
  );
}

function ExplainedField({ label, helpKey, children }) {
  return (
    <label className="field">
      <span className="field-label">
        {label}
        <InfoPopover item={help[helpKey]} />
      </span>
      {children}
    </label>
  );
}

function IconButton({ icon: Icon, label, onClick, disabled, variant = "secondary", type = "button" }) {
  return (
    <button type={type} className={`button ${variant}`} onClick={onClick} disabled={disabled} title={label}>
      <Icon size={17} />
      <span>{label}</span>
    </button>
  );
}

function StatusPill({ tone = "muted", label, icon: Icon = CheckCircle2 }) {
  return (
    <span className={`status-pill ${tone}`}>
      <Icon size={14} />
      {label}
    </span>
  );
}

function formatMoney(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 });
}

function formatPct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function formatSignedPct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const numeric = Number(value) * 100;
  const sign = numeric > 0 ? "+" : "";
  return `${sign}${numeric.toFixed(2)}%`;
}

function toneForNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "neutral";
  if (Number(value) > 0) return "positive";
  if (Number(value) < 0) return "negative";
  return "neutral";
}

function formatDate(value) {
  if (!value) return "-";
  return String(value).slice(0, 10);
}

function formatDateTime(value) {
  if (!value) return "-";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value);
  return parsed.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function runLabel(run) {
  if (!run) return "Select a run";
  const preset = presetInfo[run.preset]?.title || run.preset || run.mode || "Benchmark run";
  const status = run.status || "unknown";
  const model = run.model || "model not selected";
  return `${status} | ${preset} | ${model} | ${formatDateTime(run.created_at)}`;
}

function parseMemoryJson(content) {
  if (!content || typeof content !== "string") return null;
  const trimmed = content.trim();
  if (!trimmed.startsWith("{")) return null;
  try {
    return JSON.parse(trimmed);
  } catch {
    return null;
  }
}

function summarizeLegacyMemory(item) {
  const parsed = parseMemoryJson(item.content);
  const outcome = parsed?.outcome || {};
  const weights = outcome.target_weights || {};
  const nonZeroWeights = Object.entries(weights).filter(([, value]) => Math.abs(Number(value) || 0) > 0.000001);
  const allCashFallback = Object.keys(weights).length > 0 && nonZeroWeights.length === 0;
  const date = formatDate(item.knowledge_timestamp || item.decision_timestamp);
  const horizon = item.outcome_horizon || item.portfolio_scope || "memory";

  if (allCashFallback) {
    return {
      id: item.id,
      label: "Legacy lesson",
      date,
      horizon,
      detail: "Older fallback/no-position lesson. It recorded an all-cash decision, so every target weight is 0.0. Budget Official uses deterministic market cases instead.",
      tone: "warn",
    };
  }

  if (parsed) {
    const selected = nonZeroWeights
      .slice(0, 4)
      .map(([symbol, value]) => `${symbol} ${formatPct(value)}`)
      .join(", ");
    return {
      id: item.id,
      label: item.memory_type || "Memory",
      date,
      horizon,
      detail: selected ? `Non-zero weights: ${selected}` : `Stored outcome for ${formatDate(outcome.decision_date)}.`,
      tone: "neutral",
    };
  }

  return {
    id: item.id,
    label: item.memory_type || "Memory",
    date,
    horizon,
    detail: String(item.content || "").slice(0, 280),
    tone: "neutral",
  };
}

function LineSvg({ data, valueKey = "equity", height = 260 }) {
  if (!data?.length) return <div className="empty-chart">No data yet.</div>;
  const width = 900;
  const pad = 32;
  const values = data.map((point) => Number(point[valueKey] || 0));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(1e-9, max - min);
  const step = data.length > 1 ? (width - pad * 2) / (data.length - 1) : 0;
  const points = values
    .map((value, index) => {
      const x = pad + index * step;
      const y = height - pad - ((value - min) / span) * (height - pad * 2);
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
  return (
    <svg className="line-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Result chart">
      <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} />
      <line x1={pad} y1={pad} x2={pad} y2={height - pad} />
      <polyline points={points} />
      <text x={pad + 4} y={pad - 10}>{formatMoney(max)}</text>
      <text x={pad + 4} y={height - pad - 8}>{formatMoney(min)}</text>
    </svg>
  );
}

function Metric({ label, value, helpKey, tone = "" }) {
  return (
    <div className={`metric ${tone}`}>
      <span>
        {label}
        {helpKey && <InfoPopover item={help[helpKey]} />}
      </span>
      <strong>{value}</strong>
    </div>
  );
}

function PreflightPanel({ report, onRun, loading }) {
  const checks = report?.checks || [];
  const tone = report?.status === "pass" ? "ok" : report?.status === "warn" ? "warn" : report?.status === "fail" ? "bad" : "muted";
  const label = report ? `Preflight ${report.status}` : "Preflight not run";
  return (
    <div className="section-band preflight-band">
      <div className="section-title">
        <div>
          <h2>Preflight gate</h2>
          <p className="subtle">Required before Budget Official or Full Official can spend tokens. It checks the inputs, not the model.</p>
        </div>
        <div className="button-row">
          <StatusPill tone={tone === "bad" ? "warn" : tone} icon={tone === "ok" ? CheckCircle2 : AlertCircle} label={label} />
          <IconButton icon={ShieldCheck} label="Run preflight" onClick={onRun} disabled={loading === "preflight"} />
        </div>
      </div>
      {report?.estimate && (
        <div className="metrics-row compact">
          <Metric label="Test days" value={report.estimate.test_trading_days ?? "-"} />
          <Metric label="Calls per day" value={report.estimate.decision_calls_per_day ?? "-"} />
          <Metric label="Estimated calls" value={report.estimate.estimated_model_calls ?? "-"} />
          <Metric label="Symbols" value={report.symbols ?? "-"} />
        </div>
      )}
      <div className="check-grid">
        {checks.length === 0 && <div className="empty-state compact-empty">Run preflight to see the official readiness checklist.</div>}
        {checks.map((check) => (
          <article key={check.id} className={`check-card ${check.status}`}>
            <span>{check.status}</span>
            <strong>{check.id?.replaceAll("_", " ")}</strong>
            <p>{check.message}</p>
            {check.missing_by_symbol && Object.keys(check.missing_by_symbol).length > 0 && <small>{Object.keys(check.missing_by_symbol).slice(0, 8).join(", ")}</small>}
            {check.policy && <small>Policy: {check.policy}</small>}
          </article>
        ))}
      </div>
    </div>
  );
}

function DiagnosticsPanel({ report, onRefresh, loading }) {
  const metrics = report?.metrics || {};
  return (
    <div className="section-band diagnostics-band">
      <div className="section-title">
        <div>
          <h2>Run diagnostics</h2>
          <p className="subtle">Explains performance using costs, turnover, exposure, invalid decisions, and worst dates.</p>
        </div>
        <IconButton icon={RefreshCw} label="Refresh diagnostics" onClick={onRefresh} disabled={loading === "diagnostics"} />
      </div>
      {!report || report.status === "empty" ? (
        <div className="empty-state compact-empty">{report?.message || "Select a run to compute diagnostics."}</div>
      ) : (
        <>
          {report.official_status === "diagnostic" && (
            <div className="notice soft">
              <AlertCircle size={16} /> This run is diagnostic, not an official score: {(report.diagnostic_reasons || []).join(", ") || "quality issue"}.
            </div>
          )}
          <div className="metrics-row wrap">
            <Metric label="Net return" value={formatPct(metrics.net_return)} />
            <Metric label="Gross of cost" value={formatPct(metrics.gross_of_cost_return)} />
            <Metric label="Slippage drag" value={formatPct(metrics.slippage_drag)} />
            <Metric label="Total turnover" value={formatPct(metrics.total_turnover)} />
            <Metric label="Avg gross exposure" value={formatPct(metrics.avg_gross_exposure)} />
            <Metric label="Avg net exposure" value={formatPct(metrics.avg_net_exposure)} />
            <Metric label="Invalid allocations" value={metrics.invalid_allocations ?? "-"} />
            <Metric label="Legacy schema days" value={metrics.legacy_stage2_schema_days ?? "-"} />
            <Metric label="Avg positions" value={metrics.avg_nonzero_positions?.toFixed?.(1) ?? "-"} />
          </div>
          <div className="diagnostic-grid">
            <div>
              <h3>Worst dates</h3>
              <table>
                <tbody>
                  {(report.worst_days || []).slice(0, 6).map((row) => (
                    <tr key={`${row.decision_date}-${row.fill_date}`}>
                      <td>{formatDate(row.fill_date || row.decision_date)}</td>
                      <td>{formatSignedPct(row.daily_return)}</td>
                      <td>{formatPct(row.gross_exposure)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div>
              <h3>Worst symbols</h3>
              <table>
                <tbody>
                  {(report.symbol_pnl_worst || []).slice(0, 6).map((row) => (
                    <tr key={row.symbol}>
                      <td>{row.symbol}</td>
                      <td>{formatMoney(row.pnl)}</td>
                      <td>{row.trades} trades</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function App() {
  const [config, setConfig] = useState(emptyConfig);
  const [models, setModels] = useState([]);
  const [sourcePlan, setSourcePlan] = useState(null);
  const [preview, setPreview] = useState(null);
  const [activeRun, setActiveRun] = useState(null);
  const [runs, setRuns] = useState([]);
  const [memory, setMemory] = useState([]);
  const [warehouse, setWarehouse] = useState(null);
  const [live, setLive] = useState(null);
  const [preflight, setPreflight] = useState(null);
  const [diagnostics, setDiagnostics] = useState(null);
  const [view, setView] = useState("setup");
  const [setupStep, setSetupStep] = useState(0);
  const [loading, setLoading] = useState("");
  const [error, setError] = useState("");
  const [autoLoadedRunId, setAutoLoadedRunId] = useState("");

  const benchmark = config.benchmark;
  const secrets = config.secrets;
  const selectedPreset = presetInfo[benchmark.run_preset] || presetInfo.balanced_50_mini;

  useEffect(() => {
    refreshAll();
  }, []);

  useEffect(() => {
    const id = setInterval(() => {
      if (activeRun?.id && ["queued", "running", "paused", "cancelling"].includes(activeRun.status)) {
        refreshRun(activeRun.id, false);
      }
      refreshLive(false);
    }, 3000);
    return () => clearInterval(id);
  }, [activeRun?.id, activeRun?.status]);

  useEffect(() => {
    const decisionCount = activeRun?.decisions?.length || 0;
    const needsFullRun = ["results", "memory"].includes(view) && activeRun?.id && decisionCount === 0 && autoLoadedRunId !== activeRun.id;
    if (!needsFullRun) return;
    setAutoLoadedRunId(activeRun.id);
    refreshRun(activeRun.id, false);
  }, [view, activeRun?.id, activeRun?.decisions?.length, autoLoadedRunId]);

  useEffect(() => {
    if (view === "results" && activeRun?.id) {
      loadDiagnostics(activeRun.id, false);
    }
  }, [view, activeRun?.id]);

  async function refreshAll() {
    setError("");
    try {
      const [cfg, plan, history, wh, liveStatus, mem] = await Promise.all([
        api("/api/config"),
        api("/api/source-plan"),
        api("/api/benchmark/runs"),
        api("/api/warehouse/status"),
        api("/api/live/status"),
        api("/api/memory"),
      ]);
      setConfig(deepMerge(emptyConfig, cfg));
      setSourcePlan(plan);
      setRuns(history.runs || []);
      setWarehouse(wh);
      setLive(liveStatus);
      setMemory(mem.items || []);
      if (!activeRun && history.runs?.[0]) setActiveRun(history.runs[0]);
    } catch (err) {
      setError(err.message);
    }
  }

  async function refreshRun(runId = activeRun?.id, loud = true) {
    if (!runId) return;
    if (loud) setLoading("run-refresh");
    try {
      const run = await api(`/api/benchmark/runs/${runId}`);
      setActiveRun(run);
      if (view === "results") await loadDiagnostics(runId, false);
      const history = await api("/api/benchmark/runs");
      setRuns(history.runs || []);
    } catch (err) {
      if (loud) setError(err.message);
    } finally {
      if (loud) setLoading("");
    }
  }

  async function refreshLive(loud = true) {
    try {
      setLive(await api("/api/live/status"));
    } catch (err) {
      if (loud) setError(err.message);
    }
  }

  function updateBenchmark(path, value) {
    setConfig((current) => {
      const next = structuredClone(current);
      let target = next.benchmark;
      for (let i = 0; i < path.length - 1; i += 1) target = target[path[i]];
      target[path[path.length - 1]] = value;
      return next;
    });
  }

  function updateSecret(key, value) {
    setConfig((current) => ({ ...current, secrets: { ...current.secrets, [key]: value } }));
  }

  function applyPreset(key) {
    const preset = presetInfo[key];
    setConfig((current) => ({
      ...current,
      benchmark: { ...current.benchmark, ...preset.patch },
    }));
  }

  async function saveConfig() {
    setLoading("save");
    setError("");
    try {
      const saved = await api("/api/config", {
        method: "PUT",
        body: JSON.stringify({ benchmark, secrets }),
      });
      setConfig(deepMerge(emptyConfig, saved));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading("");
    }
  }

  async function refreshModels() {
    setLoading("models");
    setError("");
    try {
      await saveConfig();
      const data = await api("/api/models");
      if (data.error) setError(data.error);
      setModels(data.models || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading("");
    }
  }

  async function loadPreview() {
    setLoading("preview");
    setError("");
    try {
      await saveConfig();
      const data = await api("/api/benchmark/preview", {
        method: "POST",
        body: JSON.stringify({ config: benchmark, phase: "test" }),
      });
      setPreview(data);
      setView("inputs");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading("");
    }
  }

  async function runPreflight(loud = true) {
    if (loud) setLoading("preflight");
    setError("");
    try {
      await saveConfig();
      const data = await api("/api/benchmark/preflight", {
        method: "POST",
        body: JSON.stringify({ config: benchmark, dry_run: false }),
      });
      setPreflight(data);
      return data;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      if (loud) setLoading("");
    }
  }

  async function loadDiagnostics(runId = activeRun?.id, loud = true) {
    if (!runId) return null;
    if (loud) setLoading("diagnostics");
    try {
      const data = await api(`/api/benchmark/runs/${runId}/diagnostics`);
      setDiagnostics(data);
      return data;
    } catch (err) {
      if (loud) setError(err.message);
      return null;
    } finally {
      if (loud) setLoading("");
    }
  }

  async function startRun(dryRun = false) {
    setLoading(dryRun ? "dry-run" : "run");
    setError("");
    try {
      await saveConfig();
      const needsPreflight = !dryRun && benchmark.strict_preflight && ["budget_official", "full_official"].includes(benchmark.run_preset);
      if (needsPreflight) {
        const report = await api("/api/benchmark/preflight", {
          method: "POST",
          body: JSON.stringify({ config: benchmark, dry_run: false }),
        });
        setPreflight(report);
        if (report.status === "fail") {
          setView("run");
          const labels = (report.blocking_issues || []).map((item) => item.id).slice(0, 4).join(", ");
          setError(`Preflight failed: ${labels || "blocking checks"}. Fix these before spending tokens on an official run.`);
          return;
        }
      }
      const data = await api("/api/benchmark/runs", {
        method: "POST",
        body: JSON.stringify({ config: benchmark, dry_run: dryRun }),
      });
      if (data.preflight) setPreflight(data.preflight);
      setActiveRun(data);
      setView("run");
      await refreshAll();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading("");
    }
  }

  async function runAction(action) {
    if (!activeRun?.id) return;
    setLoading(action);
    setError("");
    try {
      const data = await api(`/api/benchmark/runs/${activeRun.id}/${action}`, { method: "POST" });
      setActiveRun(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading("");
    }
  }

  async function liveAction(action) {
    setLoading(`live-${action}`);
    setError("");
    try {
      if (action !== "stop") {
        const saved = await api("/api/config", {
          method: "PUT",
          body: JSON.stringify({ benchmark, secrets }),
        });
        setConfig(deepMerge(emptyConfig, saved));
      }
      const path = action === "snapshot" ? "/api/live/snapshot" : `/api/live/${action}`;
      const data = await api(path, {
        method: "POST",
        body: action === "stop" ? undefined : JSON.stringify({ config: benchmark, dry_run: false }),
      });
      if (data.id) setActiveRun(data);
      await refreshLive(false);
      setView("live");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading("");
    }
  }

  const estimate = preview?.estimate || activeRun?.progress || {};
  const metrics = activeRun?.summary?.metrics || {};
  const equityCurve = activeRun?.summary?.equity_curve || [];
  const buyHold = activeRun?.summary?.buy_hold_comparison || null;
  const buyHoldBenchmarks = buyHold?.benchmarks || [];
  const progressPercent = activeRun?.progress?.percent ?? 0;
  const hasOpenAIKey = Boolean(config.secret_status?.openai_api_key || secrets.openai_api_key);
  const modelReady = Boolean(benchmark.model && hasOpenAIKey);
  const runAwareViews = new Set(["run", "results"]);
  const headerPreset = runAwareViews.has(view) && activeRun?.preset && presetInfo[activeRun.preset] ? presetInfo[activeRun.preset] : selectedPreset;
  const headerMode = runAwareViews.has(view) && activeRun?.mode ? activeRun.mode : benchmark.mode;
  const headerModel = runAwareViews.has(view) && activeRun?.model ? activeRun.model : benchmark.model;
  const deterministicMemorySummary = activeRun?.summary?.deterministic_memory || null;
  const runMemorySample = useMemo(() => {
    const decisions = activeRun?.decisions || [];
    for (let index = decisions.length - 1; index >= 0; index -= 1) {
      const items = decisions[index]?.input?.memory;
      if (Array.isArray(items) && items.length > 0) return items;
    }
    return [];
  }, [activeRun]);
  const legacyMemoryRows = useMemo(() => memory.slice(0, 80).map(summarizeLegacyMemory), [memory]);

  const navItems = [
    ["setup", "Guided setup", SparkIcon],
    ["config", "Configuration", Settings],
    ["run", "Run control", Activity],
    ["inputs", "Inputs", Eye],
    ["memory", "Memory", Brain],
    ["results", "Results", BarChart3],
    ["live", "Live", Zap],
    ["warehouse", "Warehouse", Database],
  ];

  function SparkIcon(props) {
    return <ShieldCheck {...props} />;
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><LineChart size={22} /></div>
          <div>
            <strong>AI Market Benchmark</strong>
            <span>local research console</span>
          </div>
        </div>
        <nav className="nav">
          {navItems.map(([key, label, Icon]) => (
            <button key={key} className={view === key ? "active" : ""} onClick={() => setView(key)}>
              <Icon size={17} />
              <span>{label}</span>
            </button>
          ))}
        </nav>
        <div className="side-stack">
          <StatusPill tone={config.secret_status?.openai_api_key ? "ok" : "warn"} icon={KeyRound} label={config.secret_status?.openai_api_key ? "API key saved" : "API key needed"} />
          <StatusPill tone={benchmark.model ? "ok" : "warn"} icon={Brain} label={benchmark.model || "Select a model"} />
          <StatusPill tone={activeRun?.status === "running" ? "live" : "muted"} icon={Activity} label={activeRun?.status || "No active run"} />
        </div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">{headerPreset.subtitle}</p>
            <h1>{headerPreset.title}</h1>
            <p className="subtle">{headerMode === "balanced_50_portfolio" ? "Balanced 50 portfolio" : `${benchmark.symbol} diagnostic`} · {headerModel || "model not selected"}</p>
          </div>
          <div className="top-actions">
            <IconButton icon={Save} label="Save" onClick={saveConfig} disabled={loading === "save"} />
            <IconButton icon={Eye} label="Preview inputs" onClick={loadPreview} disabled={loading === "preview"} />
            <IconButton icon={ShieldCheck} label="Preflight" onClick={() => runPreflight()} disabled={loading === "preflight"} />
            <IconButton icon={Play} label="Start dry pilot" onClick={() => startRun(true)} disabled={Boolean(loading)} />
            <IconButton icon={CirclePlay} label="Run model" onClick={() => startRun(false)} disabled={Boolean(loading) || !modelReady} variant="primary" />
          </div>
        </header>

        {error && <div className="notice error"><AlertCircle size={16} /> {error}</div>}

        {view === "setup" && (
          <section className="screen">
            <div className="setup-layout">
              <div className="stepper">
                {["Model", "Preset", "Memory", "Launch"].map((item, index) => (
                  <button key={item} className={setupStep === index ? "active" : ""} onClick={() => setSetupStep(index)}>
                    <span>{index + 1}</span>
                    {item}
                  </button>
                ))}
              </div>
              <div className="setup-panel">
                {setupStep === 0 && (
                  <div className="flow">
                    <h2>Choose the intelligence being tested</h2>
                    <p className="lead">The selected model makes the investment decisions. The simulator only applies market mechanics.</p>
                    <div className="form-grid">
                      <ExplainedField label="OpenAI API key" helpKey="apiKey">
                        <input type="password" value={secrets.openai_api_key} placeholder={config.secret_status?.openai_api_key ? "Configured locally" : "sk-..."} onChange={(e) => updateSecret("openai_api_key", e.target.value)} />
                      </ExplainedField>
                      <ExplainedField label="Base URL" helpKey="apiKey">
                        <input value={secrets.openai_base_url} onChange={(e) => updateSecret("openai_base_url", e.target.value)} />
                      </ExplainedField>
                      <ExplainedField label="Model" helpKey="model">
                        <select value={benchmark.model} onChange={(e) => updateBenchmark(["model"], e.target.value)}>
                          <option value="">Select model</option>
                          {models.map((model) => <option key={model.id} value={model.id}>{model.id}</option>)}
                          {benchmark.model && !models.some((model) => model.id === benchmark.model) && <option value={benchmark.model}>{benchmark.model}</option>}
                        </select>
                      </ExplainedField>
                      <div className="field action-field">
                        <span className="field-label">Model list</span>
                        <IconButton icon={RefreshCw} label="Refresh models" onClick={refreshModels} disabled={loading === "models"} />
                      </div>
                    </div>
                  </div>
                )}
                {setupStep === 1 && (
                  <div className="flow">
                    <h2>Select the run shape</h2>
                    <p className="lead">Start small to verify mechanics, then move to full replay when you are ready for the token cost.</p>
                    <PresetSelector active={benchmark.run_preset} onSelect={applyPreset} />
                  </div>
                )}
                {setupStep === 2 && (
                  <div className="flow">
                    <h2>Set memory and market rules</h2>
                    <p className="lead">These settings define how the model learns from history and how strictly the paper portfolio is simulated.</p>
                    <div className="form-grid">
                      <ExplainedField label="Memory retrieval" helpKey="memory">
                        <select value={benchmark.memory_retrieval} onChange={(e) => updateBenchmark(["memory_retrieval"], e.target.value)}>
                          <option value="deterministic_similarity">Deterministic similarity</option>
                        </select>
                      </ExplainedField>
                      <ExplainedField label="Allow shorting" helpKey="shorting">
                        <select value={benchmark.allow_short ? "yes" : "no"} onChange={(e) => updateBenchmark(["allow_short"], e.target.value === "yes")}>
                          <option value="yes">Yes</option>
                          <option value="no">No</option>
                        </select>
                      </ExplainedField>
                      <ExplainedField label="Max gross exposure" helpKey="exposure">
                        <input type="number" step="0.05" value={benchmark.max_gross_exposure} onChange={(e) => updateBenchmark(["max_gross_exposure"], Number(e.target.value))} />
                      </ExplainedField>
                      <ExplainedField label="Max positions" helpKey="maxPositions">
                        <input type="number" min="1" max="50" value={benchmark.max_nonzero_positions} onChange={(e) => updateBenchmark(["max_nonzero_positions"], Number(e.target.value))} />
                      </ExplainedField>
                      <ExplainedField label="Daily turnover limit" helpKey="turnover">
                        <input type="number" min="0" max="1" step="0.05" value={benchmark.max_daily_turnover} onChange={(e) => updateBenchmark(["max_daily_turnover"], Number(e.target.value))} />
                      </ExplainedField>
                      <ExplainedField label="Strict preflight" helpKey="preflight">
                        <select value={benchmark.strict_preflight ? "yes" : "no"} onChange={(e) => updateBenchmark(["strict_preflight"], e.target.value === "yes")}>
                          <option value="yes">Yes</option>
                          <option value="no">No</option>
                        </select>
                      </ExplainedField>
                    </div>
                  </div>
                )}
                {setupStep === 3 && (
                  <div className="flow">
                    <h2>Review and launch</h2>
                    <p className="lead">A preview estimates call count and shows the exact point-in-time bundle before a real run.</p>
                    <div className="launch-grid">
                      <Metric label="Symbols" value={estimate.symbols ?? (benchmark.mode === "balanced_50_portfolio" ? 50 : 1)} />
                      <Metric label="Training days" value={estimate.train_days ?? benchmark.max_train_days} />
                      <Metric label="Test days" value={estimate.test_days ?? benchmark.max_test_days} />
                      <Metric label="Decision calls" value={estimate.estimated_decision_calls ?? "-"} />
                    </div>
                    <PreflightPanel report={preflight} onRun={() => runPreflight()} loading={loading} />
                    <div className="button-row">
                      <IconButton icon={Eye} label="Preview first" onClick={loadPreview} disabled={loading === "preview"} />
                      <IconButton icon={ShieldCheck} label="Preflight" onClick={() => runPreflight()} disabled={loading === "preflight"} />
                      <IconButton icon={Play} label="Dry run" onClick={() => startRun(true)} disabled={Boolean(loading)} />
                      <IconButton icon={CirclePlay} label="Run selected model" onClick={() => startRun(false)} disabled={Boolean(loading) || !modelReady} variant="primary" />
                    </div>
                  </div>
                )}
                <div className="wizard-actions">
                  <IconButton icon={ChevronRight} label={setupStep < 3 ? "Next" : "Open dashboard"} onClick={() => setupStep < 3 ? setSetupStep(setupStep + 1) : setView("run")} variant="primary" />
                </div>
              </div>
            </div>
          </section>
        )}

        {view === "config" && (
          <section className="screen config-grid">
            <div className="section-band">
              <div className="section-title">
                <h2>Run preset</h2>
                <InfoPopover item={help.preset} />
              </div>
              <PresetSelector active={benchmark.run_preset} onSelect={applyPreset} compact />
            </div>
            <div className="section-band">
              <div className="section-title"><h2>Model</h2></div>
              <div className="form-grid">
                <ExplainedField label="OpenAI API key" helpKey="apiKey">
                  <input type="password" value={secrets.openai_api_key} placeholder={config.secret_status?.openai_api_key ? "Configured locally" : "sk-..."} onChange={(e) => updateSecret("openai_api_key", e.target.value)} />
                </ExplainedField>
                <ExplainedField label="Model" helpKey="model">
                  <select value={benchmark.model} onChange={(e) => updateBenchmark(["model"], e.target.value)}>
                    <option value="">Select model</option>
                    {models.map((model) => <option key={model.id} value={model.id}>{model.id}</option>)}
                    {benchmark.model && !models.some((model) => model.id === benchmark.model) && <option value={benchmark.model}>{benchmark.model}</option>}
                  </select>
                </ExplainedField>
                <ExplainedField label="Temperature" helpKey="temperature">
                  <input type="number" step="0.1" value={benchmark.temperature} onChange={(e) => updateBenchmark(["temperature"], Number(e.target.value))} />
                </ExplainedField>
                <ExplainedField label="Output tokens" helpKey="outputTokens">
                  <input type="number" value={benchmark.max_output_tokens} onChange={(e) => updateBenchmark(["max_output_tokens"], Number(e.target.value))} />
                </ExplainedField>
                <ExplainedField label="Use LLM cache" helpKey="cache">
                  <select value={benchmark.use_cached_llm ? "yes" : "no"} onChange={(e) => updateBenchmark(["use_cached_llm"], e.target.value === "yes")}>
                    <option value="yes">Yes</option>
                    <option value="no">No</option>
                  </select>
                </ExplainedField>
                <div className="field action-field">
                  <span className="field-label">Model tools</span>
                  <IconButton icon={RefreshCw} label="Refresh models" onClick={refreshModels} disabled={loading === "models"} />
                </div>
              </div>
            </div>
            <div className="section-band">
              <div className="section-title"><h2>Benchmark</h2></div>
              <div className="form-grid">
                <ExplainedField label="Mode" helpKey="mode">
                  <select value={benchmark.mode} onChange={(e) => updateBenchmark(["mode"], e.target.value)}>
                    <option value="balanced_50_portfolio">Balanced 50 portfolio</option>
                    <option value="single_stock">Single-stock diagnostic</option>
                  </select>
                </ExplainedField>
                <ExplainedField label="Single-stock symbol" helpKey="mode">
                  <input value={benchmark.symbol} onChange={(e) => updateBenchmark(["symbol"], e.target.value.toUpperCase())} />
                </ExplainedField>
                <ExplainedField label="Training start" helpKey="trainDates">
                  <input type="date" value={benchmark.train_start} onChange={(e) => updateBenchmark(["train_start"], e.target.value)} />
                </ExplainedField>
                <ExplainedField label="Training end" helpKey="trainDates">
                  <input type="date" value={benchmark.train_end} onChange={(e) => updateBenchmark(["train_end"], e.target.value)} />
                </ExplainedField>
                <ExplainedField label="Test start" helpKey="testDates">
                  <input type="date" value={benchmark.test_start} onChange={(e) => updateBenchmark(["test_start"], e.target.value)} />
                </ExplainedField>
                <ExplainedField label="Test end" helpKey="testDates">
                  <input type="date" value={benchmark.test_end} onChange={(e) => updateBenchmark(["test_end"], e.target.value)} />
                </ExplainedField>
                <ExplainedField label="Max training days" helpKey="dayCaps">
                  <input type="number" value={benchmark.max_train_days} onChange={(e) => updateBenchmark(["max_train_days"], Number(e.target.value))} />
                </ExplainedField>
                <ExplainedField label="Max test days" helpKey="dayCaps">
                  <input type="number" value={benchmark.max_test_days} onChange={(e) => updateBenchmark(["max_test_days"], Number(e.target.value))} />
                </ExplainedField>
              </div>
            </div>
            <div className="section-band">
              <div className="section-title"><h2>Trading and memory</h2></div>
              <div className="form-grid">
                <ExplainedField label="Initial cash" helpKey="initialCash">
                  <input type="number" value={benchmark.initial_cash} onChange={(e) => updateBenchmark(["initial_cash"], Number(e.target.value))} />
                </ExplainedField>
                <ExplainedField label="Allow shorting" helpKey="shorting">
                  <select value={benchmark.allow_short ? "yes" : "no"} onChange={(e) => updateBenchmark(["allow_short"], e.target.value === "yes")}>
                    <option value="yes">Yes</option>
                    <option value="no">No</option>
                  </select>
                </ExplainedField>
                <ExplainedField label="Max gross exposure" helpKey="exposure">
                  <input type="number" step="0.05" value={benchmark.max_gross_exposure} onChange={(e) => updateBenchmark(["max_gross_exposure"], Number(e.target.value))} />
                </ExplainedField>
                <ExplainedField label="Max positions" helpKey="maxPositions">
                  <input type="number" min="1" max="50" value={benchmark.max_nonzero_positions} onChange={(e) => updateBenchmark(["max_nonzero_positions"], Number(e.target.value))} />
                </ExplainedField>
                <ExplainedField label="Daily turnover limit" helpKey="turnover">
                  <input type="number" min="0" max="1" step="0.05" value={benchmark.max_daily_turnover} onChange={(e) => updateBenchmark(["max_daily_turnover"], Number(e.target.value))} />
                </ExplainedField>
                <ExplainedField label="Cost edge multiplier" helpKey="edgeMultiplier">
                  <input type="number" min="1" step="0.5" value={benchmark.turnover_edge_multiplier} onChange={(e) => updateBenchmark(["turnover_edge_multiplier"], Number(e.target.value))} />
                </ExplainedField>
                <ExplainedField label="Slippage bps" helpKey="slippage">
                  <input type="number" value={benchmark.slippage_bps} onChange={(e) => updateBenchmark(["slippage_bps"], Number(e.target.value))} />
                </ExplainedField>
                <ExplainedField label="Fee per trade" helpKey="fees">
                  <input type="number" value={benchmark.commission_per_trade} onChange={(e) => updateBenchmark(["commission_per_trade"], Number(e.target.value))} />
                </ExplainedField>
                <ExplainedField label="Fee per share" helpKey="fees">
                  <input type="number" value={benchmark.commission_per_share} onChange={(e) => updateBenchmark(["commission_per_share"], Number(e.target.value))} />
                </ExplainedField>
                <ExplainedField label="Stage 1 chunk size" helpKey="chunk">
                  <input type="number" value={benchmark.stage1_chunk_size} onChange={(e) => updateBenchmark(["stage1_chunk_size"], Number(e.target.value))} />
                </ExplainedField>
                <ExplainedField label="Memory retrieval" helpKey="memory">
                  <select value={benchmark.memory_retrieval} onChange={(e) => updateBenchmark(["memory_retrieval"], e.target.value)}>
                    <option value="deterministic_similarity">Deterministic similarity</option>
                  </select>
                </ExplainedField>
                <ExplainedField label="Memory cases per symbol" helpKey="memory">
                  <input type="number" min="1" max="5" value={benchmark.deterministic_memory_per_symbol} onChange={(e) => updateBenchmark(["deterministic_memory_per_symbol"], Number(e.target.value))} />
                </ExplainedField>
                <ExplainedField label="Max memory items" helpKey="memory">
                  <input type="number" min="10" max="120" value={benchmark.deterministic_memory_max_items} onChange={(e) => updateBenchmark(["deterministic_memory_max_items"], Number(e.target.value))} />
                </ExplainedField>
                <ExplainedField label="Memory neighbors" helpKey="memoryNeighbors">
                  <input type="number" min="5" max="200" value={benchmark.memory_k_neighbors} onChange={(e) => updateBenchmark(["memory_k_neighbors"], Number(e.target.value))} />
                </ExplainedField>
                <ExplainedField label="Examples per symbol" helpKey="memoryNeighbors">
                  <input type="number" min="0" max="5" value={benchmark.memory_examples_per_symbol} onChange={(e) => updateBenchmark(["memory_examples_per_symbol"], Number(e.target.value))} />
                </ExplainedField>
                <ExplainedField label="Strict preflight" helpKey="preflight">
                  <select value={benchmark.strict_preflight ? "yes" : "no"} onChange={(e) => updateBenchmark(["strict_preflight"], e.target.value === "yes")}>
                    <option value="yes">Yes</option>
                    <option value="no">No</option>
                  </select>
                </ExplainedField>
                <ExplainedField label="Require micro-pilot" helpKey="microPilot">
                  <select value={benchmark.require_paid_micro_pilot ? "yes" : "no"} onChange={(e) => updateBenchmark(["require_paid_micro_pilot"], e.target.value === "yes")}>
                    <option value="yes">Yes</option>
                    <option value="no">No</option>
                  </select>
                </ExplainedField>
              </div>
            </div>
            <div className="section-band">
              <div className="section-title"><h2>Data sources</h2><InfoPopover item={help.dataSources} /></div>
              <SourceControls benchmark={benchmark} secrets={secrets} updateBenchmark={updateBenchmark} updateSecret={updateSecret} />
            </div>
          </section>
        )}

        {view === "run" && (
          <section className="screen run-layout">
            <div className="section-band hero-run">
              <div>
                <p className="eyebrow">Run control</p>
                <h2>{activeRun?.status ? activeRun.status.toUpperCase() : "No run started"}</h2>
                <p className="subtle">{activeRun?.progress?.message || "Preview inputs, dry run, or start a real model benchmark."}</p>
              </div>
              <div className="run-buttons">
                <IconButton icon={Eye} label="Preview" onClick={loadPreview} disabled={loading === "preview"} />
                <IconButton icon={ShieldCheck} label="Preflight" onClick={() => runPreflight()} disabled={loading === "preflight"} />
                <IconButton icon={Play} label="Dry run" onClick={() => startRun(true)} disabled={Boolean(loading)} />
                <IconButton icon={CirclePlay} label="Run model" onClick={() => startRun(false)} disabled={Boolean(loading) || !modelReady} variant="primary" />
                <IconButton icon={Pause} label="Pause" onClick={() => runAction("pause")} disabled={!activeRun?.id || activeRun.status !== "running"} />
                <IconButton icon={CirclePause} label="Resume" onClick={() => runAction("resume")} disabled={!activeRun?.id || activeRun.status !== "paused"} />
                <IconButton icon={Square} label="Cancel" onClick={() => runAction("cancel")} disabled={!activeRun?.id || !["running", "paused", "queued"].includes(activeRun.status)} />
              </div>
              <div className="progress-track"><span style={{ width: `${Math.max(0, Math.min(100, progressPercent))}%` }} /></div>
            </div>
            <div className="metrics-row">
              <Metric label="Progress" value={`${progressPercent}%`} />
              <Metric label="Phase" value={activeRun?.phase || "-"} />
              <Metric label="Model calls" value={activeRun?.progress?.model_calls ?? activeRun?.summary?.model_calls ?? "-"} />
              <Metric label="Current date" value={activeRun?.progress?.current_decision_date || "-"} />
            </div>
            <PreflightPanel report={preflight} onRun={() => runPreflight()} loading={loading} />
            <div className="section-band">
              <div className="section-title"><h2>Recent runs</h2><IconButton icon={RefreshCw} label="Refresh" onClick={refreshAll} /></div>
              <div className="run-table">
                {runs.map((run) => (
                  <button key={run.id} onClick={() => { setActiveRun(run); refreshRun(run.id); }}>
                    <span>{run.status}</span>
                    <strong>{run.preset || run.mode}</strong>
                    <em>{run.model}</em>
                    <small>{run.created_at}</small>
                  </button>
                ))}
              </div>
            </div>
          </section>
        )}

        {view === "inputs" && (
          <section className="screen inputs-grid">
            <div className="section-band">
              <div className="section-title"><h2>Input preview</h2><IconButton icon={RefreshCw} label="Refresh preview" onClick={loadPreview} /></div>
              <div className="metrics-row compact">
                <Metric label="Symbols" value={preview?.estimate?.symbols ?? "-"} />
                <Metric label="Train days" value={preview?.estimate?.train_days ?? "-"} />
                <Metric label="Test days" value={preview?.estimate?.test_days ?? "-"} />
                <Metric label="Decision calls" value={preview?.estimate?.estimated_decision_calls ?? "-"} />
              </div>
              <pre className="json-view">{preview ? JSON.stringify(preview.bundle, null, 2) : "No preview loaded."}</pre>
            </div>
            <div className="section-band">
              <div className="section-title"><h2>Prompt preview</h2></div>
              <pre className="json-view">{preview ? JSON.stringify({ stage1: preview.stage1_prompt, stage2: preview.stage2_prompt }, null, 2) : "Run Preview to inspect the exact prompt contracts."}</pre>
            </div>
          </section>
        )}

        {view === "memory" && (
          <section className="screen memory-screen">
            <div className="section-band">
              <div className="section-title">
                <div>
                  <h2>Official benchmark memory</h2>
                  <p className="subtle">What the selected run can use before making decisions. This is the memory contract for Budget Official.</p>
                </div>
                <div className="button-row">
                  <IconButton icon={RefreshCw} label="Load selected run" onClick={() => refreshRun()} disabled={!activeRun?.id || loading === "run-refresh"} />
                  <IconButton icon={RefreshCw} label="Refresh list" onClick={refreshAll} />
                </div>
              </div>
              {deterministicMemorySummary ? (
                <div className="memory-summary-grid">
                  <article className="memory-summary-card">
                    <span>Memory mode</span>
                    <strong>Deterministic cases</strong>
                    <p>Historical market situations are matched locally and sent to the model as examples with later outcomes.</p>
                  </article>
                  <article className="memory-summary-card">
                    <span>Historical cases</span>
                    <strong>{formatMoney(deterministicMemorySummary.cases)}</strong>
                    <p>{deterministicMemorySummary.symbols || "-"} symbols available for retrieval.</p>
                  </article>
                  <article className="memory-summary-card">
                    <span>Training range</span>
                    <strong>{formatDate(deterministicMemorySummary.train_start)} to {formatDate(deterministicMemorySummary.train_end)}</strong>
                    <p>Only information available before each decision can be retrieved.</p>
                  </article>
                </div>
              ) : (
                <div className="empty-state compact-empty">Open or refresh a completed Budget Official run to see the official memory coverage.</div>
              )}
            </div>

            <div className="section-band">
              <div className="section-title">
                <div>
                  <h2>Recent cases supplied to the model</h2>
                  <p className="subtle">A readable sample from the selected run's actual prompt inputs.</p>
                </div>
              </div>
              <div className="memory-case-list">
                {runMemorySample.length === 0 && <div className="empty-state compact-empty">Load the selected run to inspect the historical cases used in its prompts.</div>}
                {runMemorySample.slice(0, 16).map((item, index) => (
                  <article key={`${item.id || item.symbol || "memory"}-${index}`} className="memory-case">
                    <div className="memory-case-head">
                      <span>{item.symbol || "Portfolio"}</span>
                      <strong>{formatDate(item.decision_timestamp || item.knowledge_timestamp)}</strong>
                      <em>{item.retrieval_score !== undefined ? `score ${Number(item.retrieval_score).toFixed(3)}` : item.memory_type || "case"}</em>
                    </div>
                    <p>{item.content || "No readable memory content stored for this item."}</p>
                  </article>
                ))}
              </div>
            </div>

            <div className="section-band legacy-memory-band">
              <div className="section-title">
                <div>
                  <h2>Legacy model-written lessons</h2>
                  <p className="subtle">Older rows kept for transparency. They are not used by the deterministic Budget Official benchmark.</p>
                </div>
              </div>
              <div className="memory-list">
                {legacyMemoryRows.length === 0 && <div className="empty-state compact-empty">No legacy memory rows found.</div>}
                {legacyMemoryRows.map((item) => (
                  <article key={item.id} className={`memory-row readable ${item.tone}`}>
                    <span>{item.label}</span>
                    <strong>{item.date}</strong>
                    <em>{item.horizon}</em>
                    <p>{item.detail}</p>
                  </article>
                ))}
              </div>
            </div>
          </section>
        )}

        {view === "results" && (
          <section className="screen results-grid">
            <div className="section-band run-picker-band">
              <div className="section-title">
                <div>
                  <h2>Select result run</h2>
                  <p className="subtle">Switch between completed, cancelled, failed, and pilot runs without leaving Results.</p>
                </div>
                <IconButton icon={RefreshCw} label="Refresh runs" onClick={refreshAll} disabled={loading === "run-refresh"} />
              </div>
              <div className="run-picker-control">
                <select value={activeRun?.id || ""} onChange={(e) => refreshRun(e.target.value)} disabled={!runs.length || loading === "run-refresh"}>
                  <option value="">Select a run</option>
                  {activeRun?.id && !runs.some((run) => run.id === activeRun.id) && <option value={activeRun.id}>{runLabel(activeRun)}</option>}
                  {runs.map((run) => (
                    <option key={run.id} value={run.id}>{runLabel(run)}</option>
                  ))}
                </select>
                <div className="run-picker-meta">
                  <span>{activeRun?.status || "No run selected"}</span>
                  <strong>{activeRun?.preset ? presetInfo[activeRun.preset]?.title || activeRun.preset : "Choose a run to inspect"}</strong>
                  <em>{activeRun?.id || "Results, decisions, and events will update after selection."}</em>
                </div>
              </div>
            </div>
            <div className="section-band chart-band">
              <div className="section-title"><h2>Equity curve</h2><IconButton icon={RefreshCw} label="Refresh run" onClick={() => refreshRun()} /></div>
              <LineSvg data={equityCurve} />
            </div>
            <div className="metrics-row">
              <Metric label="Final equity" value={formatMoney(metrics.final_equity)} />
              <Metric label="Total return" value={formatPct(metrics.total_return)} />
              <Metric label="Max drawdown" value={formatPct(metrics.max_drawdown)} />
              <Metric label="Sharpe-like" value={metrics.sharpe_like?.toFixed?.(2) ?? "-"} />
              <Metric label="Fees" value={formatMoney(metrics.fees)} />
              <Metric label="Slippage drag" value={formatPct(metrics.slippage_drag)} />
              <Metric label="Turnover" value={formatPct(metrics.total_turnover)} />
              <Metric label="Invalid allocations" value={metrics.invalid_allocation_count ?? metrics.model_failures ?? "-"} />
            </div>
            <DiagnosticsPanel report={diagnostics} onRefresh={() => loadDiagnostics(activeRun?.id)} loading={loading} />
            {buyHoldBenchmarks.length > 0 && (
              <div className="section-band comparison-band">
                <div className="section-title">
                  <div>
                    <h2>AI vs buy and hold</h2>
                    <p className="subtle">Same tested dates, adjusted prices, no model trading in the benchmark portfolios.</p>
                  </div>
                </div>
                <div className="comparison-grid">
                  <article className="comparison-card ai-card">
                    <span>Tested strategy</span>
                    <strong>AI portfolio</strong>
                    <div className="comparison-main">{formatPct(buyHold.ai_strategy?.total_return ?? metrics.total_return)}</div>
                    <dl>
                      <div><dt>Final value</dt><dd>{formatMoney(buyHold.ai_strategy?.final_equity ?? metrics.final_equity)}</dd></div>
                      <div><dt>Max drawdown</dt><dd>{formatPct(metrics.max_drawdown)}</dd></div>
                    </dl>
                  </article>
                  {buyHoldBenchmarks.map((item) => (
                    <article key={item.id} className="comparison-card">
                      <span>Buy and hold</span>
                      <strong>{item.label}</strong>
                      <div className="comparison-main">{formatPct(item.total_return)}</div>
                      <dl>
                        <div><dt>AI vs this</dt><dd className={toneForNumber(item.excess_return)}>{formatSignedPct(item.excess_return)}</dd></div>
                        <div><dt>Final value</dt><dd>{formatMoney(item.final_equity)}</dd></div>
                        <div><dt>Max drawdown</dt><dd>{formatPct(item.max_drawdown)}</dd></div>
                        <div><dt>Symbols</dt><dd>{item.symbols}</dd></div>
                      </dl>
                    </article>
                  ))}
                </div>
              </div>
            )}
            <div className="section-band">
              <div className="section-title"><h2>Decision inspector</h2></div>
              <pre className="json-view">{activeRun?.decisions ? JSON.stringify(activeRun.decisions.slice(-6), null, 2) : "Open a completed or running benchmark to inspect decisions."}</pre>
            </div>
            <div className="section-band">
              <div className="section-title"><h2>Run events</h2></div>
              <pre className="json-view">{activeRun?.events ? JSON.stringify(activeRun.events.slice(-50), null, 2) : "No events loaded."}</pre>
            </div>
          </section>
        )}

        {view === "live" && (
          <section className="screen">
            <div className="section-band hero-run">
              <div>
                <p className="eyebrow">Live paper benchmark</p>
                <h2>{live?.running ? "Hourly scheduler running" : "Scheduler stopped"}</h2>
                <p className="subtle">{live?.message || "Live mode runs while this local backend is open."}</p>
              </div>
              <InfoPopover item={help.live} />
              <div className="run-buttons">
                <IconButton icon={Zap} label="Start hourly" onClick={() => liveAction("start")} disabled={Boolean(loading) || !modelReady} variant="primary" />
                <IconButton icon={Square} label="Stop" onClick={() => liveAction("stop")} disabled={loading === "live-stop"} />
                <IconButton icon={Activity} label="Run snapshot now" onClick={() => liveAction("snapshot")} disabled={Boolean(loading) || !modelReady} />
              </div>
            </div>
            {!modelReady && (
              <div className="notice soft">
                <KeyRound size={16} /> Select a model and save an OpenAI API key before live model decisions. Dry pilots can still run without a key.
              </div>
            )}
            <div className={`market-status ${live?.market?.market_open ? "open" : "closed"}`}>
              <div>
                <span>{live?.market?.market_open ? "Market open" : "Market closed"}</span>
                <strong>{live?.market?.message || "Live mode follows regular US exchange hours."}</strong>
              </div>
              <div>
                <small>Exchange time</small>
                <b>{live?.market?.now ? new Date(live.market.now).toLocaleString() : "-"}</b>
              </div>
            </div>
            {live?.last_snapshot?.status === "skipped" && (
              <div className="notice soft">
                <Clock size={16} /> Last live snapshot was skipped because the market is closed. The scheduler keeps waiting locally.
              </div>
            )}
            <div className="section-band">
              <div className="section-title"><h2>Live status</h2></div>
              <pre className="json-view">{JSON.stringify(live || {}, null, 2)}</pre>
            </div>
          </section>
        )}

        {view === "warehouse" && (
          <section className="screen warehouse-grid">
            <div className="section-band">
              <div className="section-title"><h2>Warehouse tables</h2><IconButton icon={RefreshCw} label="Refresh" onClick={refreshAll} /></div>
              <div className="metrics-row wrap">
                {Object.entries(warehouse?.tables || {}).map(([name, count]) => (
                  <Metric key={name} label={name} value={Number(count).toLocaleString()} />
                ))}
              </div>
            </div>
            <div className="section-band">
              <div className="section-title"><h2>Latest logs</h2></div>
              <pre className="json-view">{warehouse ? JSON.stringify(warehouse.latest_logs || [], null, 2) : "Warehouse not initialized."}</pre>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

function PresetSelector({ active, onSelect, compact = false }) {
  return (
    <div className={`preset-list ${compact ? "compact" : ""}`}>
      {Object.entries(presetInfo).map(([key, preset]) => (
        <button key={key} className={active === key ? "selected" : ""} onClick={() => onSelect(key)}>
          <span>{preset.subtitle}</span>
          <strong>{preset.title}</strong>
          <em>{preset.body}</em>
        </button>
      ))}
    </div>
  );
}

function SourceControls({ benchmark, secrets, updateBenchmark, updateSecret }) {
  const selected = new Set(benchmark.data_sources.news_sources || []);
  function toggle(source) {
    const next = new Set(selected);
    if (next.has(source)) next.delete(source);
    else next.add(source);
    updateBenchmark(["data_sources", "news_sources"], Array.from(next));
  }
  return (
    <>
      <div className="source-list">
        {["gdelt", "marketaux", "finnhub", "newsapi", "rss"].map((source) => (
          <div key={source} className={`source-option ${selected.has(source) ? "selected" : ""}`}>
            <button onClick={() => toggle(source)} aria-pressed={selected.has(source)}>
              <Database size={16} />
              <span>{source.toUpperCase()}</span>
              <small>{sourceHelp[source].body}</small>
            </button>
            <InfoPopover item={sourceHelp[source]} />
          </div>
        ))}
      </div>
      <div className="form-grid">
        <ExplainedField label="Marketaux key" helpKey="dataSources">
          <input type="password" value={secrets.marketaux_key} placeholder="optional" onChange={(e) => updateSecret("marketaux_key", e.target.value)} />
        </ExplainedField>
        <ExplainedField label="Finnhub key" helpKey="dataSources">
          <input type="password" value={secrets.finnhub_key} placeholder="optional" onChange={(e) => updateSecret("finnhub_key", e.target.value)} />
        </ExplainedField>
        <ExplainedField label="NewsAPI key" helpKey="dataSources">
          <input type="password" value={secrets.newsapi_key} placeholder="optional" onChange={(e) => updateSecret("newsapi_key", e.target.value)} />
        </ExplainedField>
        <ExplainedField label="FRED key" helpKey="dataSources">
          <input type="password" value={secrets.fred_api_key} placeholder="optional" onChange={(e) => updateSecret("fred_api_key", e.target.value)} />
        </ExplainedField>
        <ExplainedField label="SEC user agent" helpKey="dataSources">
          <input value={secrets.sec_user_agent} placeholder="Name email@example.com" onChange={(e) => updateSecret("sec_user_agent", e.target.value)} />
        </ExplainedField>
        <ExplainedField label="News per symbol" helpKey="dataSources">
          <input type="number" value={benchmark.max_news_per_symbol} onChange={(e) => updateBenchmark(["max_news_per_symbol"], Number(e.target.value))} />
        </ExplainedField>
        <ExplainedField label="Macro policy" helpKey="macroPolicy">
          <select value={benchmark.macro_policy} onChange={(e) => updateBenchmark(["macro_policy"], e.target.value)}>
            <option value="omit_if_missing">Omit if missing</option>
            <option value="include_status_rows">Include status rows</option>
          </select>
        </ExplainedField>
        <ExplainedField label="News policy" helpKey="newsPolicy">
          <select value={benchmark.news_policy} onChange={(e) => updateBenchmark(["news_policy"], e.target.value)}>
            <option value="real_titles_or_aggregate_events">Real titles or aggregate events</option>
            <option value="raw_titles">Raw titles</option>
          </select>
        </ExplainedField>
      </div>
    </>
  );
}

export default App;
