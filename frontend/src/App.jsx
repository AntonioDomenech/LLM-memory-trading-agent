import React, { useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  BarChart3,
  CheckCircle2,
  Database,
  Eye,
  KeyRound,
  Play,
  RefreshCw,
  Save,
  Settings,
} from "lucide-react";

const API_BASE = "http://127.0.0.1:8000";

const emptyConfig = {
  benchmark: {
    symbol: "AAPL",
    company_name: "Apple",
    start_date: "2025-01-02",
    end_date: "2025-03-31",
    model: "",
    endpoint: "responses",
    initial_cash: 100000,
    max_days: 20,
    allow_short: false,
    max_leverage: 1,
    slippage_bps: 5,
    commission_per_trade: 0,
    commission_per_share: 0,
    temperature: 0,
    max_output_tokens: 900,
    use_cached_llm: true,
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
    marketaux_key: "",
    newsapi_key: "",
    finnhub_key: "",
    fred_api_key: "",
    sec_user_agent: "",
  },
  secret_status: {},
};

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || data.error || "Request failed");
  }
  return data;
}

function Field({ label, children }) {
  return (
    <label className="field">
      <span>{label}</span>
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

function StatusPill({ ok, label }) {
  return (
    <span className={`status-pill ${ok ? "ok" : "muted"}`}>
      {ok ? <CheckCircle2 size={14} /> : <AlertCircle size={14} />}
      {label}
    </span>
  );
}

function EquityChart({ data }) {
  if (!data?.length) {
    return <div className="empty-chart">No equity curve loaded.</div>;
  }
  const width = 760;
  const height = 280;
  const pad = 28;
  const values = data.map((point) => Number(point.equity || 0));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(1, max - min);
  const xStep = data.length > 1 ? (width - pad * 2) / (data.length - 1) : 0;
  const points = values
    .map((value, index) => {
      const x = pad + index * xStep;
      const y = height - pad - ((value - min) / span) * (height - pad * 2);
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
  return (
    <svg className="equity-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Equity curve">
      <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} />
      <line x1={pad} y1={pad} x2={pad} y2={height - pad} />
      <polyline points={points} />
      {data.map((point, index) => {
        if (index !== 0 && index !== data.length - 1 && index % Math.ceil(data.length / 4) !== 0) return null;
        const x = pad + index * xStep;
        return (
          <text key={point.date} x={x} y={height - 8} textAnchor="middle">
            {point.date.slice(5)}
          </text>
        );
      })}
      <text x={pad + 4} y={pad - 8}>{max.toLocaleString(undefined, { maximumFractionDigits: 0 })}</text>
      <text x={pad + 4} y={height - pad - 6}>{min.toLocaleString(undefined, { maximumFractionDigits: 0 })}</text>
    </svg>
  );
}

function App() {
  const [config, setConfig] = useState(emptyConfig);
  const [models, setModels] = useState([]);
  const [sourcePlan, setSourcePlan] = useState(null);
  const [preview, setPreview] = useState(null);
  const [run, setRun] = useState(null);
  const [runs, setRuns] = useState([]);
  const [warehouse, setWarehouse] = useState(null);
  const [activeTab, setActiveTab] = useState("config");
  const [loading, setLoading] = useState("");
  const [error, setError] = useState("");

  const benchmark = config.benchmark;
  const secrets = config.secrets;

  useEffect(() => {
    refreshAll();
  }, []);

  async function refreshAll() {
    setError("");
    try {
      const [cfg, plan, history, wh] = await Promise.all([
        api("/api/config"),
        api("/api/source-plan"),
        api("/api/runs"),
        api("/api/warehouse/status"),
      ]);
      setConfig({ ...emptyConfig, ...cfg });
      setSourcePlan(plan);
      setRuns(history.runs || []);
      setWarehouse(wh);
    } catch (err) {
      setError(err.message);
    }
  }

  async function refreshWarehouse() {
    setLoading("warehouse");
    setError("");
    try {
      setWarehouse(await api("/api/warehouse/status"));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading("");
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

  async function saveConfig() {
    setLoading("save");
    setError("");
    try {
      const saved = await api("/api/config", {
        method: "PUT",
        body: JSON.stringify({ benchmark, secrets }),
      });
      setConfig({ ...emptyConfig, ...saved });
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
      const data = await api("/api/preview", {
        method: "POST",
        body: JSON.stringify({ config: benchmark, as_of_date: benchmark.start_date }),
      });
      setPreview(data);
      setActiveTab("inputs");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading("");
    }
  }

  async function startRun(dryRun = false) {
    setLoading(dryRun ? "dry" : "run");
    setError("");
    try {
      await saveConfig();
      const data = await api("/api/runs", {
        method: "POST",
        body: JSON.stringify({ config: benchmark, dry_run: dryRun }),
      });
      setRun(data);
      const history = await api("/api/runs");
      setRuns(history.runs || []);
      setActiveTab("results");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading("");
    }
  }

  const selectedSources = useMemo(() => new Set(benchmark.data_sources.news_sources), [benchmark.data_sources.news_sources]);

  function toggleSource(source) {
    const next = new Set(selectedSources);
    if (next.has(source)) next.delete(source);
    else next.add(source);
    updateBenchmark(["data_sources", "news_sources"], Array.from(next));
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <BarChart3 size={22} />
          </div>
          <div>
            <strong>AI Market Benchmark</strong>
            <span>local research console</span>
          </div>
        </div>
        <nav>
          <button className={activeTab === "config" ? "active" : ""} onClick={() => setActiveTab("config")}>
            <Settings size={17} /> Configuration
          </button>
          <button className={activeTab === "inputs" ? "active" : ""} onClick={() => setActiveTab("inputs")}>
            <Database size={17} /> LLM Inputs
          </button>
          <button className={activeTab === "results" ? "active" : ""} onClick={() => setActiveTab("results")}>
            <BarChart3 size={17} /> Results
          </button>
          <button className={activeTab === "warehouse" ? "active" : ""} onClick={() => setActiveTab("warehouse")}>
            <Database size={17} /> Warehouse
          </button>
        </nav>
        <div className="side-status">
          <StatusPill ok={config.secret_status?.openai_api_key} label="OpenAI key" />
          <StatusPill ok={Boolean(benchmark.model)} label={benchmark.model || "No model"} />
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <h1>{benchmark.symbol || "Symbol"} benchmark</h1>
            <p>{benchmark.start_date} to {benchmark.end_date}</p>
          </div>
          <div className="actions">
            <IconButton icon={Eye} label="Preview" onClick={loadPreview} disabled={loading === "preview"} />
            <IconButton icon={Play} label="Dry run" onClick={() => startRun(true)} disabled={Boolean(loading)} />
            <IconButton icon={Play} label="Run model" onClick={() => startRun(false)} disabled={Boolean(loading) || !benchmark.model} variant="primary" />
          </div>
        </header>

        {error && <div className="error"><AlertCircle size={16} /> {error}</div>}

        {activeTab === "config" && (
          <section className="grid two">
            <div className="panel">
              <div className="panel-title">
                <h2>Model</h2>
                <IconButton icon={RefreshCw} label="Refresh models" onClick={refreshModels} disabled={loading === "models"} />
              </div>
              <div className="form-grid">
                <Field label="OpenAI API key">
                  <input type="password" value={secrets.openai_api_key} placeholder={config.secret_status?.openai_api_key ? "Configured" : "sk-..."} onChange={(e) => updateSecret("openai_api_key", e.target.value)} />
                </Field>
                <Field label="Base URL">
                  <input value={secrets.openai_base_url} onChange={(e) => updateSecret("openai_base_url", e.target.value)} />
                </Field>
                <Field label="Model">
                  <select value={benchmark.model} onChange={(e) => updateBenchmark(["model"], e.target.value)}>
                    <option value="">Select model</option>
                    {models.map((model) => <option key={model.id} value={model.id}>{model.id}</option>)}
                    {benchmark.model && !models.some((model) => model.id === benchmark.model) && <option value={benchmark.model}>{benchmark.model}</option>}
                  </select>
                </Field>
                <Field label="Endpoint">
                  <select value={benchmark.endpoint} onChange={(e) => updateBenchmark(["endpoint"], e.target.value)}>
                    <option value="responses">Responses</option>
                    <option value="chat_completions">Chat Completions</option>
                  </select>
                </Field>
                <Field label="Temperature">
                  <input type="number" step="0.1" value={benchmark.temperature} onChange={(e) => updateBenchmark(["temperature"], Number(e.target.value))} />
                </Field>
                <Field label="Output tokens">
                  <input type="number" value={benchmark.max_output_tokens} onChange={(e) => updateBenchmark(["max_output_tokens"], Number(e.target.value))} />
                </Field>
              </div>
              <div className="panel-actions">
                <IconButton icon={Save} label="Save" onClick={saveConfig} disabled={loading === "save"} variant="primary" />
              </div>
            </div>

            <div className="panel">
              <div className="panel-title">
                <h2>Benchmark</h2>
              </div>
              <div className="form-grid">
                <Field label="Symbol">
                  <input value={benchmark.symbol} onChange={(e) => updateBenchmark(["symbol"], e.target.value.toUpperCase())} />
                </Field>
                <Field label="Company">
                  <input value={benchmark.company_name} onChange={(e) => updateBenchmark(["company_name"], e.target.value)} />
                </Field>
                <Field label="Start">
                  <input type="date" value={benchmark.start_date} onChange={(e) => updateBenchmark(["start_date"], e.target.value)} />
                </Field>
                <Field label="End">
                  <input type="date" value={benchmark.end_date} onChange={(e) => updateBenchmark(["end_date"], e.target.value)} />
                </Field>
                <Field label="Max days">
                  <input type="number" value={benchmark.max_days} onChange={(e) => updateBenchmark(["max_days"], Number(e.target.value))} />
                </Field>
                <Field label="Initial cash">
                  <input type="number" value={benchmark.initial_cash} onChange={(e) => updateBenchmark(["initial_cash"], Number(e.target.value))} />
                </Field>
                <Field label="Max leverage">
                  <input type="number" step="0.1" value={benchmark.max_leverage} onChange={(e) => updateBenchmark(["max_leverage"], Number(e.target.value))} />
                </Field>
                <label className="check-row">
                  <input type="checkbox" checked={benchmark.allow_short} onChange={(e) => updateBenchmark(["allow_short"], e.target.checked)} />
                  <span>Shorting</span>
                </label>
              </div>
            </div>

            <div className="panel wide">
              <div className="panel-title">
                <h2>Data</h2>
              </div>
              <div className="source-grid">
                {["gdelt", "marketaux", "finnhub", "newsapi", "rss"].map((source) => (
                  <label key={source} className={`source-toggle ${selectedSources.has(source) ? "selected" : ""}`}>
                    <input type="checkbox" checked={selectedSources.has(source)} onChange={() => toggleSource(source)} />
                    <span>{source.toUpperCase()}</span>
                  </label>
                ))}
              </div>
              <div className="form-grid">
                <Field label="Marketaux key">
                  <input type="password" value={secrets.marketaux_key} placeholder={config.secret_status?.marketaux_key ? "Configured" : ""} onChange={(e) => updateSecret("marketaux_key", e.target.value)} />
                </Field>
                <Field label="Finnhub key">
                  <input type="password" value={secrets.finnhub_key} placeholder={config.secret_status?.finnhub_key ? "Configured" : ""} onChange={(e) => updateSecret("finnhub_key", e.target.value)} />
                </Field>
                <Field label="NewsAPI key">
                  <input type="password" value={secrets.newsapi_key} placeholder={config.secret_status?.newsapi_key ? "Configured" : ""} onChange={(e) => updateSecret("newsapi_key", e.target.value)} />
                </Field>
                <Field label="FRED key">
                  <input type="password" value={secrets.fred_api_key} placeholder={config.secret_status?.fred_api_key ? "Configured" : ""} onChange={(e) => updateSecret("fred_api_key", e.target.value)} />
                </Field>
                <Field label="SEC user agent">
                  <input value={secrets.sec_user_agent} placeholder={config.secret_status?.sec_user_agent ? "Configured" : "Name email@example.com"} onChange={(e) => updateSecret("sec_user_agent", e.target.value)} />
                </Field>
                <Field label="News/day">
                  <input type="number" value={benchmark.data_sources.max_news_per_day} onChange={(e) => updateBenchmark(["data_sources", "max_news_per_day"], Number(e.target.value))} />
                </Field>
              </div>
            </div>
          </section>
        )}

        {activeTab === "inputs" && (
          <section className="grid two">
            <div className="panel">
              <div className="panel-title">
                <h2>Information plan</h2>
              </div>
              <div className="source-list">
                {(sourcePlan?.sources || []).map((source) => (
                  <div className="source-row" key={source.name}>
                    <strong>{source.name}</strong>
                    <span>{source.provider}</span>
                    <em>{source.cost}</em>
                  </div>
                ))}
              </div>
            </div>
            <div className="panel">
              <div className="panel-title">
                <h2>Preview</h2>
                <IconButton icon={Eye} label="Refresh preview" onClick={loadPreview} disabled={loading === "preview"} />
              </div>
              <pre className="json-view">{preview ? JSON.stringify(preview, null, 2) : "No preview loaded."}</pre>
            </div>
          </section>
        )}

        {activeTab === "results" && (
          <section className="grid two">
            <div className="panel wide">
              <div className="panel-title">
                <h2>Equity</h2>
              </div>
              <div className="chart">
                <EquityChart data={run?.summary?.equity_curve || []} />
              </div>
            </div>
            <div className="panel">
              <div className="panel-title">
                <h2>Summary</h2>
              </div>
              <div className="metric-grid">
                <div><span>Final equity</span><strong>{run?.summary?.metrics?.final_equity?.toLocaleString?.() || "-"}</strong></div>
                <div><span>Total return</span><strong>{run?.summary?.metrics ? `${(run.summary.metrics.total_return * 100).toFixed(2)}%` : "-"}</strong></div>
                <div><span>Max drawdown</span><strong>{run?.summary?.metrics ? `${(run.summary.metrics.max_drawdown * 100).toFixed(2)}%` : "-"}</strong></div>
                <div><span>Constraint events</span><strong>{run?.summary?.event_count ?? "-"}</strong></div>
              </div>
            </div>
            <div className="panel">
              <div className="panel-title">
                <h2>History</h2>
                <IconButton icon={RefreshCw} label="Refresh" onClick={refreshAll} />
              </div>
              <div className="run-list">
                {runs.map((item) => (
                  <button key={item.id} onClick={() => setRun({ summary: item.summary, decisions: [] })}>
                    <strong>{item.summary.symbol} · {item.summary.model}</strong>
                    <span>{item.created_at}</span>
                  </button>
                ))}
              </div>
            </div>
            <div className="panel">
              <div className="panel-title">
                <h2>Decisions</h2>
              </div>
              <pre className="json-view">{run ? JSON.stringify(run.decisions?.slice(-3) || [], null, 2) : "No run loaded."}</pre>
            </div>
          </section>
        )}

        {activeTab === "warehouse" && (
          <section className="grid two">
            <div className="panel">
              <div className="panel-title">
                <h2>Warehouse</h2>
                <IconButton icon={RefreshCw} label="Refresh" onClick={refreshWarehouse} disabled={loading === "warehouse"} />
              </div>
              <div className="metric-grid">
                {Object.entries(warehouse?.tables || {}).map(([name, count]) => (
                  <div key={name}>
                    <span>{name}</span>
                    <strong>{Number(count).toLocaleString()}</strong>
                  </div>
                ))}
              </div>
            </div>
            <div className="panel">
              <div className="panel-title">
                <h2>Latest logs</h2>
              </div>
              <pre className="json-view">{warehouse ? JSON.stringify(warehouse.latest_logs || [], null, 2) : "Warehouse not initialized yet."}</pre>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
