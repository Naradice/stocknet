"use client";

import { useEffect, useState, useCallback } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import DatasourceFormModal from "@/app/components/DatasourceFormModal";
import { Datasource, CsvDatasource, SimulatorDatasource } from "@/lib/datasourceTypes";

type SimPhase = "running" | "completed" | "failed" | "stopped";

interface SimStatus {
  phase: SimPhase;
  simulationId: string;
  startedAt: string;
  endedAt?: string;
  pid?: number;
  length?: number | null;
}

// ── Type badge ─────────────────────────────────────────────────────────────────

function TypeBadge({ type }: { type: "csv" | "simulator" }) {
  return type === "csv" ? (
    <span className="text-xs bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 px-2 py-0.5 rounded-full">
      CSV
    </span>
  ) : (
    <span className="text-xs bg-purple-500/20 text-purple-300 border border-purple-500/30 px-2 py-0.5 rounded-full">
      Simulator
    </span>
  );
}

// ── Phase badge ────────────────────────────────────────────────────────────────

function PhaseBadge({ phase }: { phase: SimPhase }) {
  const map: Record<SimPhase, string> = {
    running:   "bg-green-500/20 text-green-400 border-green-500/30",
    completed: "bg-blue-500/20 text-blue-400 border-blue-500/30",
    failed:    "bg-red-500/20 text-red-400 border-red-500/30",
    stopped:   "bg-gray-500/20 text-gray-400 border-gray-500/30",
  };
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full border ${map[phase]}`}>
      {phase === "running" ? (
        <span className="flex items-center gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse inline-block" />
          running
        </span>
      ) : phase}
    </span>
  );
}

// ── API response type ─────────────────────────────────────────────────────────

interface SimApiResponse {
  running: boolean;
  phase: SimPhase | null;
  status: SimStatus | null;
  log: string;
}

// ── Data preview panel ────────────────────────────────────────────────────────

interface ChartRow {
  time: string;
  open?: number;
  high?: number;
  low?: number;
  close: number;
}

interface PreviewSummary {
  totalRows: number;
  fromDate: string;
  toDate: string;
  columns: string[];
  closeMin: number;
  closeMax: number;
  closeMean: number;
}

interface PreviewData {
  available: boolean;
  reason?: string;
  summary?: PreviewSummary;
  chart?: ChartRow[];
}

function DataPreviewPanel({ name }: { name: string }) {
  const [data, setData] = useState<PreviewData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetch(`/api/datasources/${encodeURIComponent(name)}/preview`, { cache: "no-store" })
      .then((r) => r.json())
      .then((d: PreviewData) => setData(d))
      .catch(() => setData({ available: false, reason: "Failed to fetch preview" }))
      .finally(() => setLoading(false));
  }, [name]);

  if (loading) {
    return (
      <div className="border-t border-gray-800 pt-3">
        <p className="text-xs text-gray-500">Loading preview…</p>
      </div>
    );
  }

  if (!data?.available) {
    return (
      <div className="border-t border-gray-800 pt-3">
        <p className="text-xs text-gray-500 italic">{data?.reason ?? "No data available"}</p>
      </div>
    );
  }

  const { summary, chart } = data;

  // Format time labels — show only time portion if dates are the same, otherwise date only
  const tickFormatter = (t: string) => {
    if (!t) return "";
    // Try to shorten: if it's a datetime, show date only
    const d = new Date(t);
    if (!isNaN(d.getTime())) {
      return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
    }
    return t.length > 10 ? t.slice(0, 10) : t;
  };

  return (
    <div className="border-t border-gray-800 pt-3 space-y-3">
      <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
        Data Preview
        {data.reason && (
          <span className="ml-2 text-yellow-500 font-normal normal-case">{data.reason}</span>
        )}
      </span>

      {/* Summary stats */}
      {summary && (
        <dl className="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-1 text-xs">
          <div>
            <dt className="text-gray-500">Rows</dt>
            <dd className="text-gray-300">{summary.totalRows.toLocaleString()}</dd>
          </div>
          <div>
            <dt className="text-gray-500">From</dt>
            <dd className="text-gray-300 truncate">{summary.fromDate}</dd>
          </div>
          <div>
            <dt className="text-gray-500">To</dt>
            <dd className="text-gray-300 truncate">{summary.toDate}</dd>
          </div>
          <div>
            <dt className="text-gray-500">Close Min</dt>
            <dd className="text-gray-300">{summary.closeMin.toFixed(4)}</dd>
          </div>
          <div>
            <dt className="text-gray-500">Close Max</dt>
            <dd className="text-gray-300">{summary.closeMax.toFixed(4)}</dd>
          </div>
          <div>
            <dt className="text-gray-500">Close Mean</dt>
            <dd className="text-gray-300">{summary.closeMean.toFixed(4)}</dd>
          </div>
        </dl>
      )}

      {/* Chart */}
      {chart && chart.length > 0 && (
        <div className="h-36">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chart} margin={{ top: 2, right: 4, left: 0, bottom: 2 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis
                dataKey="time"
                tick={{ fontSize: 9, fill: "#6b7280" }}
                tickFormatter={tickFormatter}
                interval="preserveStartEnd"
                minTickGap={60}
              />
              <YAxis
                tick={{ fontSize: 9, fill: "#6b7280" }}
                width={52}
                domain={["auto", "auto"]}
                tickFormatter={(v: number) => v.toFixed(2)}
              />
              <Tooltip
                contentStyle={{ background: "#111827", border: "1px solid #374151", fontSize: 10 }}
                labelStyle={{ color: "#9ca3af" }}
                itemStyle={{ color: "#60a5fa" }}
                formatter={(v: number) => [v.toFixed(4), "close"]}
              />
              <Line
                type="monotone"
                dataKey="close"
                dot={false}
                stroke="#60a5fa"
                strokeWidth={1}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

// ── Simulation panel ──────────────────────────────────────────────────────────

function SimulationPanel({
  name,
  onRegisterCsv,
}: {
  name: string;
  onRegisterCsv: (path: string) => void;
}) {
  const [data, setData] = useState<SimApiResponse | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [simulationId, setSimulationId] = useState("");
  const [length, setLength] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`/api/datasources/${encodeURIComponent(name)}/simulate`, {
        cache: "no-store",
      });
      if (res.ok) {
        setData(await res.json());
        setLoaded(true);
      }
    } catch {
      // transient network error — retain current state
    }
  }, [name]);

  // Fetch on mount
  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  // Poll while running
  useEffect(() => {
    if (!data?.running) return;
    const id = setInterval(fetchStatus, 3000);
    return () => clearInterval(id);
  }, [data?.running, fetchStatus]);

  async function handleStart() {
    setPending(true);
    setError("");
    const res = await fetch(`/api/datasources/${encodeURIComponent(name)}/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        action: "start",
        simulationId: simulationId.trim() || undefined,
        length: length ? parseInt(length, 10) : null,
      }),
    });
    const body = await res.json();
    if (!res.ok) setError(body.error ?? "Start failed");
    else await fetchStatus();
    setPending(false);
  }

  async function handleStop() {
    setPending(true);
    setError("");
    const res = await fetch(`/api/datasources/${encodeURIComponent(name)}/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "stop" }),
    });
    if (!res.ok) {
      const body = await res.json();
      setError(body.error ?? "Stop failed");
    }
    await fetchStatus();
    setPending(false);
  }

  async function handleCheckpoint() {
    setPending(true);
    setError("");
    const res = await fetch(`/api/datasources/${encodeURIComponent(name)}/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "checkpoint" }),
    });
    const body = await res.json();
    if (!res.ok) setError(body.error ?? "Checkpoint failed");
    setPending(false);
  }

  async function handleRegisterAsCsv() {
    setPending(true);
    setError("");
    const res = await fetch(`/api/datasources/${encodeURIComponent(name)}/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "export" }),
    });
    const body = await res.json();
    if (!res.ok) { setError(body.error ?? "Export failed"); setPending(false); return; }
    onRegisterCsv(body.filePath);
    setPending(false);
  }

  const running = data?.running ?? false;
  const phase = data?.phase;
  const status = data?.status;
  const log = data?.log ?? "";

  return (
    <div className="border-t border-gray-800 pt-3 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
            Simulation
          </span>
          {phase && <PhaseBadge phase={phase} />}
        </div>
        {status && (
          <span className="text-xs text-gray-600">
            {status.startedAt ? new Date(status.startedAt).toLocaleString() : ""}
          </span>
        )}
      </div>

      {/* Last simulation ID */}
      {status && phase !== "running" && (
        <div className="text-xs text-gray-500 font-mono truncate" title={status.simulationId}>
          {status.simulationId}
        </div>
      )}

      {/* Start controls — only render after first fetch; hide while running */}
      {!loaded && (
        <p className="text-xs text-gray-600">Checking status…</p>
      )}
      {loaded && !running && (
        <div className="space-y-2">
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block text-xs text-gray-500 mb-1">
                Simulation ID <span className="text-gray-600">(optional)</span>
              </label>
              <input
                className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs focus:outline-none focus:border-blue-500"
                value={simulationId}
                onChange={(e) => setSimulationId(e.target.value)}
                placeholder="auto-generated"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">
                Candles <span className="text-gray-600">(optional)</span>
              </label>
              <input
                type="number"
                min={1}
                className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs focus:outline-none focus:border-blue-500"
                value={length}
                onChange={(e) => setLength(e.target.value)}
                placeholder="1000"
              />
            </div>
          </div>
          <button
            onClick={handleStart}
            disabled={pending}
            className="text-xs px-3 py-1.5 bg-purple-700 hover:bg-purple-600 disabled:opacity-50 rounded-md transition-colors font-medium"
          >
            {pending ? "Starting…" : "▶ Run Simulation"}
          </button>
        </div>
      )}

      {running && (
        <div className="flex gap-2">
          <button
            onClick={handleCheckpoint}
            disabled={pending}
            className="text-xs px-3 py-1.5 bg-amber-700 hover:bg-amber-600 disabled:opacity-50 rounded-md transition-colors"
            title="Copy current output and register as a CSV datasource"
          >
            {pending ? "…" : "⬡ Checkpoint"}
          </button>
          <button
            onClick={handleStop}
            disabled={pending}
            className="text-xs px-3 py-1.5 bg-red-700 hover:bg-red-600 disabled:opacity-50 rounded-md transition-colors"
          >
            {pending ? "…" : "■ Stop"}
          </button>
        </div>
      )}

      {error && <p className="text-xs text-red-400">{error}</p>}

      {/* Log tail */}
      {log && (
        <pre className="text-xs text-gray-400 bg-gray-950 rounded p-2 overflow-x-auto max-h-32 overflow-y-auto whitespace-pre-wrap break-all">
          {log}
        </pre>
      )}

      {/* Register output — available after completed or stopped */}
      {(phase === "completed" || phase === "stopped") && status?.simulationId && (
        <button
          onClick={handleRegisterAsCsv}
          disabled={pending}
          className="text-xs px-3 py-1.5 bg-emerald-700 hover:bg-emerald-600 disabled:opacity-50 rounded-md text-white transition-colors"
        >
          {pending ? "Exporting…" : "+ Register output as CSV datasource"}
        </button>
      )}
    </div>
  );
}

// ── Simulator status summary (shown in card header without expanding panel) ──

function SimStatusDot({ name }: { name: string }) {
  const [phase, setPhase] = useState<SimPhase | null>(null);

  useEffect(() => {
    fetch(`/api/datasources/${encodeURIComponent(name)}/simulate`, { cache: "no-store" })
      .then((r) => r.json())
      .then((d: SimApiResponse) => setPhase(d.phase))
      .catch(() => {});
  }, [name]);

  if (!phase) return null;
  return <PhaseBadge phase={phase} />;
}

// ── Datasource card ────────────────────────────────────────────────────────────

function DatasourceCard({
  ds,
  onEdit,
  onDelete,
  onRegisterCsv,
}: {
  ds: Datasource;
  onEdit: () => void;
  onDelete: () => void;
  onRegisterCsv: (filePath: string, fromSim: string) => void;
}) {
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [previewExpanded, setPreviewExpanded] = useState(false);
  // For simulator cards: start expanded if there's a known simulation state
  const [simExpanded, setSimExpanded] = useState(false);
  const [simHasHistory, setSimHasHistory] = useState(false);

  // Check if this simulator has any history — if so, auto-expand
  useEffect(() => {
    if (ds.type !== "simulator") return;
    fetch(`/api/datasources/${encodeURIComponent(ds.name)}/simulate`, { cache: "no-store" })
      .then((r) => r.json())
      .then((d: SimApiResponse) => {
        if (d.running || d.phase !== null) {
          setSimHasHistory(true);
          setSimExpanded(true); // auto-expand when there's a known state
        }
      })
      .catch(() => {});
  }, [ds.name, ds.type]);

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-5 flex flex-col gap-3">
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-semibold text-white truncate">{ds.name}</span>
            <TypeBadge type={ds.type} />
            <span className="text-xs text-gray-500 font-mono">{ds.datasetKey}</span>
            {/* Show last simulation phase in the header without expanding */}
            {ds.type === "simulator" && !simExpanded && <SimStatusDot name={ds.name} />}
          </div>
          {ds.description && (
            <p className="text-sm text-gray-400 mt-0.5 truncate">{ds.description}</p>
          )}
        </div>
        <div className="flex gap-2 flex-shrink-0">
          <button
            onClick={() => setPreviewExpanded((v) => !v)}
            className={`text-xs px-2.5 py-1 rounded-md transition-colors ${
              previewExpanded
                ? "bg-blue-700 text-white"
                : "bg-gray-800 hover:bg-gray-700"
            }`}
          >
            {previewExpanded ? "Hide Preview" : "Preview"}
          </button>
          {ds.type === "simulator" && (
            <button
              onClick={() => setSimExpanded((v) => !v)}
              className={`text-xs px-2.5 py-1 rounded-md transition-colors ${
                simExpanded
                  ? "bg-purple-700 text-white"
                  : "bg-gray-800 hover:bg-gray-700"
              }`}
            >
              {simExpanded ? "Hide" : simHasHistory ? "Simulation ↕" : "Simulate"}
            </button>
          )}
          <button
            onClick={onEdit}
            className="text-xs px-2.5 py-1 bg-gray-800 hover:bg-gray-700 rounded-md transition-colors"
          >
            Edit
          </button>
          {confirmDelete ? (
            <div className="flex gap-1">
              <button
                onClick={onDelete}
                className="text-xs px-2.5 py-1 bg-red-700 hover:bg-red-600 rounded-md transition-colors"
              >
                Confirm
              </button>
              <button
                onClick={() => setConfirmDelete(false)}
                className="text-xs px-2.5 py-1 bg-gray-800 hover:bg-gray-700 rounded-md transition-colors"
              >
                Cancel
              </button>
            </div>
          ) : (
            <button
              onClick={() => setConfirmDelete(true)}
              className="text-xs px-2.5 py-1 bg-gray-800 hover:bg-red-900/50 text-gray-400 hover:text-red-300 rounded-md transition-colors"
            >
              Delete
            </button>
          )}
        </div>
      </div>

      {/* Details */}
      {ds.type === "csv" && <CsvDetails ds={ds as CsvDatasource} />}
      {ds.type === "simulator" && <SimDetails ds={ds as SimulatorDatasource} />}

      {/* Preview panel */}
      {previewExpanded && <DataPreviewPanel name={ds.name} />}

      {/* Simulation panel */}
      {ds.type === "simulator" && simExpanded && (
        <SimulationPanel
          name={ds.name}
          onRegisterCsv={(p) => onRegisterCsv(p, ds.name)}
        />
      )}
    </div>
  );
}

function CsvDetails({ ds }: { ds: CsvDatasource }) {
  return (
    <dl className="grid grid-cols-2 sm:grid-cols-4 gap-x-4 gap-y-1 text-xs">
      <Kv label="File" value={ds.filePath} span={2} mono />
      <Kv label="Columns" value={ds.columns.join(", ")} span={2} />
      <Kv label="Obs. Length" value={String(ds.observationLength)} />
      <Kv label="Pred. Length" value={String(ds.predictionLength)} />
    </dl>
  );
}

function SimDetails({ ds }: { ds: SimulatorDatasource }) {
  return (
    <dl className="grid grid-cols-2 sm:grid-cols-4 gap-x-4 gap-y-1 text-xs">
      <Kv label="Agents/Model" value={String(ds.agentPerModel)} />
      <Kv label="Models" value={String(ds.modelCount)} />
      <Kv label="Output Length" value={String(ds.outputLength)} />
      <Kv label="Sampler Rule" value={ds.samplerRule} />
      <Kv label="Initial Price" value={String(ds.modelConfig.initial_price)} />
      <Kv label="Spread" value={String(ds.modelConfig.spread)} />
      <Kv label="Max Volatility" value={String(ds.modelConfig.max_volatility)} />
      <Kv label="Min Volatility" value={String(ds.modelConfig.min_volatility)} />
    </dl>
  );
}

function Kv({
  label,
  value,
  span,
  mono,
}: {
  label: string;
  value: string;
  span?: number;
  mono?: boolean;
}) {
  return (
    <div className={span === 2 ? "col-span-2" : ""}>
      <dt className="text-gray-500">{label}</dt>
      <dd className={`text-gray-300 truncate ${mono ? "font-mono" : ""}`}>{value}</dd>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function DatasourcesPage() {
  const [datasources, setDatasources] = useState<Datasource[]>([]);
  const [loading, setLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);
  const [editing, setEditing] = useState<Datasource | null>(null);
  const [csvPrefill, setCsvPrefill] = useState<{ filePath: string; description: string } | null>(
    null
  );

  async function load() {
    try {
      const res = await fetch("/api/datasources", { cache: "no-store" });
      if (res.ok) setDatasources(await res.json());
    } catch {
      // transient network error — keep previous state
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleDelete(name: string) {
    await fetch(`/api/datasources/${encodeURIComponent(name)}`, { method: "DELETE" });
    load();
  }

  function handleSaved() {
    setShowModal(false);
    setEditing(null);
    setCsvPrefill(null);
    load();
  }

  function handleRegisterCsv(filePath: string, fromSim: string) {
    setCsvPrefill({ filePath, description: `Output from simulation: ${fromSim}` });
    setEditing(null);
    setShowModal(true);
  }

  const csvCount = datasources.filter((d) => d.type === "csv").length;
  const simCount = datasources.filter((d) => d.type === "simulator").length;

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Data Sources</h1>
          <p className="text-gray-500 text-sm mt-1">
            {loading ? "Loading…" : `${csvCount} CSV · ${simCount} Simulator`}
          </p>
        </div>
        <button
          onClick={() => {
            setEditing(null);
            setCsvPrefill(null);
            setShowModal(true);
          }}
          className="px-4 py-2 text-sm bg-blue-600 hover:bg-blue-500 rounded-md font-medium transition-colors"
        >
          + Register
        </button>
      </div>

      {!loading && datasources.length === 0 && (
        <div className="flex flex-col items-center justify-center py-24 text-gray-600 gap-3">
          <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 7h16M4 12h16M4 17h8" />
          </svg>
          <p className="text-sm">No data sources registered yet.</p>
          <button
            onClick={() => setShowModal(true)}
            className="text-blue-400 hover:text-blue-300 text-sm transition-colors"
          >
            Register your first data source →
          </button>
        </div>
      )}

      {datasources.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {datasources.map((ds) => (
            <DatasourceCard
              key={ds.name}
              ds={ds}
              onEdit={() => {
                setEditing(ds);
                setCsvPrefill(null);
                setShowModal(true);
              }}
              onDelete={() => handleDelete(ds.name)}
              onRegisterCsv={handleRegisterCsv}
            />
          ))}
        </div>
      )}

      {showModal && (
        <DatasourceFormModal
          onClose={() => {
            setShowModal(false);
            setEditing(null);
            setCsvPrefill(null);
          }}
          onSaved={handleSaved}
          initial={editing ?? undefined}
          csvPrefill={csvPrefill ?? undefined}
        />
      )}
    </div>
  );
}
