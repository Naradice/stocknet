"use client";

import { useEffect, useState, useCallback } from "react";
import {
  DbRun,
  EpochEntry,
  LogsMap,
  COLORS,
  pickColor,
  shortLabel,
  buildUnifiedData,
  buildVolData,
  LossCurvesChart,
  BestVolChart,
} from "@/components/ScaleCharts";

// ── Types ─────────────────────────────────────────────────────────────────────

interface DbModel {
  model_name: string;
  run_count: number;
}

function fmt(n: number | null | undefined): string {
  return n != null ? n.toExponential(4) : "—";
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function ComparePage() {
  const [dbModels, setDbModels] = useState<DbModel[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [runs, setRuns] = useState<DbRun[]>([]);
  const [checkedIds, setCheckedIds] = useState<Set<number>>(new Set());

  const [logs, setLogs] = useState<LogsMap>({});
  const [compared, setCompared] = useState<DbRun[]>([]);
  const [logScale, setLogScale] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/api/compare/models")
      .then((r) => r.json())
      .then((data: DbModel[]) => {
        if (!Array.isArray(data)) {
          setError("Cannot connect to database.");
          return;
        }
        setDbModels(data);
        if (data.length) setSelectedModel(data[0].model_name);
      })
      .catch(() => setError("Cannot connect to database."));
  }, []);

  useEffect(() => {
    if (!selectedModel) return;
    fetch(`/api/compare/runs?model=${encodeURIComponent(selectedModel)}`)
      .then((r) => r.json())
      .then((data: DbRun[]) => {
        setRuns(data);
        setCheckedIds(new Set());
      });
  }, [selectedModel]);

  const toggle = (id: number) =>
    setCheckedIds((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });

  const compare = useCallback(async () => {
    if (!checkedIds.size) return;
    setLoading(true);
    try {
      const ids = Array.from(checkedIds);
      const res = await fetch(`/api/compare/logs?run_ids=${ids.join(",")}`);
      const raw: Record<string, EpochEntry[]> = await res.json();
      setLogs(raw);
      setCompared(runs.filter((r) => checkedIds.has(r.id)));
    } finally {
      setLoading(false);
    }
  }, [checkedIds, runs]);

  const allRates = Array.from(
    new Set(
      compared
        .map((r) => r.metadata?.volume_rate)
        .filter((v): v is number => v != null)
    )
  ).sort((a, b) => a - b);

  const unifiedData = compared.length ? buildUnifiedData(compared, logs) : [];
  const volData = allRates.length >= 2 ? buildVolData(compared, allRates) : [];

  return (
    // sticky sidebar layout — no fixed height on outer container so content flows naturally
    <div className="flex gap-6 items-start">
      {/* ── Sidebar ── */}
      <aside className="w-72 flex-shrink-0 flex flex-col gap-4 sticky top-6">
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <label className="block text-xs text-gray-400 mb-1.5">Model</label>
          <select
            className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            {dbModels.map((m) => (
              <option key={m.model_name} value={m.model_name}>
                {m.model_name} ({m.run_count})
              </option>
            ))}
          </select>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-lg flex flex-col max-h-[calc(100vh-14rem)]">
          <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800 flex-shrink-0">
            <span className="text-xs text-gray-400">{checkedIds.size} selected</span>
            <div className="flex gap-3">
              <button
                onClick={() => setCheckedIds(new Set(runs.map((r) => r.id)))}
                className="text-xs text-blue-400 hover:text-blue-300"
              >
                All
              </button>
              <button
                onClick={() => setCheckedIds(new Set())}
                className="text-xs text-gray-500 hover:text-gray-300"
              >
                None
              </button>
            </div>
          </div>
          <div className="flex-1 overflow-y-auto p-2 space-y-1 min-h-0">
            {runs.map((run) => {
              const vr = run.metadata?.volume_rate;
              const checked = checkedIds.has(run.id);
              return (
                <label
                  key={run.id}
                  className={`flex items-start gap-2.5 px-3 py-2.5 rounded-md cursor-pointer transition-colors ${
                    checked
                      ? "bg-blue-900/40 border border-blue-700/40"
                      : "hover:bg-gray-800 border border-transparent"
                  }`}
                >
                  <input
                    type="checkbox"
                    className="mt-0.5 accent-blue-500"
                    checked={checked}
                    onChange={() => toggle(run.id)}
                  />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <span className="text-sm font-medium text-gray-200 truncate">
                        {run.version}
                      </span>
                      {vr != null && (
                        <span className="flex-shrink-0 text-xs bg-blue-500/20 text-blue-300 border border-blue-500/30 px-1.5 py-0.5 rounded-full">
                          vol {vr}
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-gray-500 mt-0.5">
                      {run.epoch_count} epochs
                      {run.best_val_loss != null &&
                        ` · val ${run.best_val_loss.toExponential(2)}`}
                    </div>
                  </div>
                </label>
              );
            })}
            {!runs.length && (
              <p className="text-sm text-gray-600 text-center py-6">
                No runs in database.
              </p>
            )}
          </div>
          <div className="p-3 border-t border-gray-800 flex-shrink-0">
            <button
              onClick={compare}
              disabled={!checkedIds.size || loading}
              className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-gray-800 disabled:text-gray-600 text-white text-sm font-medium py-2 rounded-md transition-colors"
            >
              {loading ? "Loading…" : "Compare"}
            </button>
          </div>
        </div>
      </aside>

      {/* ── Main ── */}
      <div className="flex-1 flex flex-col gap-6">
        {error && (
          <div className="bg-red-900/30 border border-red-700/40 text-red-400 rounded-lg px-4 py-3 text-sm">
            {error}
          </div>
        )}
        {!compared.length && !error && (
          <div className="flex items-center justify-center h-64 text-gray-600 text-sm">
            Select runs and click Compare.
          </div>
        )}

        {compared.length > 0 && (
          <>
            {/* ── Loss Curves ── */}
            <Card
              title="Loss Curves"
              right={
                <label className="flex items-center gap-2 cursor-pointer select-none">
                  <span className="text-xs text-gray-400">Log scale</span>
                  <div
                    onClick={() => setLogScale((v) => !v)}
                    className={`relative w-8 h-4 rounded-full transition-colors ${
                      logScale ? "bg-blue-600" : "bg-gray-700"
                    }`}
                  >
                    <span
                      className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-all ${
                        logScale ? "left-4" : "left-0.5"
                      }`}
                    />
                  </div>
                </label>
              }
            >
              <LossCurvesChart
                data={unifiedData}
                runs={compared}
                allRates={allRates}
                logScale={logScale}
              />
              <p className="text-xs text-gray-600 mt-2">
                Solid lines = train &nbsp;·&nbsp; Dashed lines = validation
              </p>
            </Card>

            {/* ── Best Val Loss by Volume ── */}
            {volData.length > 0 && (
              <Card title="Best Validation Loss by Data Volume">
                <BestVolChart data={volData} runs={compared} logScale={logScale} />
              </Card>
            )}

            {/* ── Summary table ── */}
            <Card title="Summary">
              <SummaryTable runs={compared} allRates={allRates} />
            </Card>
          </>
        )}
      </div>
    </div>
  );
}

// ── Card wrapper ───────────────────────────────────────────────────────────────

function Card({
  title,
  right,
  children,
}: {
  title: string;
  right?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <span className="text-sm font-medium text-gray-300">{title}</span>
        {right}
      </div>
      <div className="p-4">{children}</div>
    </div>
  );
}

// ── Summary table ─────────────────────────────────────────────────────────────

function SummaryTable({ runs, allRates }: { runs: DbRun[]; allRates: number[] }) {
  const sorted = [...runs].sort(
    (a, b) => (a.metadata?.volume_rate ?? 0) - (b.metadata?.volume_rate ?? 0)
  );
  return (
    <div className="overflow-x-auto -mx-4 -mb-4">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-800 text-gray-400 text-xs uppercase tracking-wide text-left">
            <th className="px-4 py-2">Version</th>
            <th className="px-4 py-2 text-right">Vol Rate</th>
            <th className="px-4 py-2 text-right">Epochs</th>
            <th className="px-4 py-2 text-right">Best Train</th>
            <th className="px-4 py-2 text-right">Best Val</th>
            <th className="px-4 py-2">Created</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((r, i) => {
            const vr = r.metadata?.volume_rate;
            const color = pickColor(vr, allRates, i);
            return (
              <tr
                key={r.id}
                className="border-b border-gray-800 last:border-0 hover:bg-gray-800/50"
              >
                <td className="px-4 py-2">
                  <div className="flex items-center gap-2">
                    <span
                      className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                      style={{ background: color }}
                    />
                    <span className="font-mono text-xs text-gray-200">
                      {r.version}
                    </span>
                  </div>
                </td>
                <td className="px-4 py-2 text-right">
                  {vr != null ? (
                    <span className="text-xs bg-blue-500/20 text-blue-300 border border-blue-500/30 px-1.5 py-0.5 rounded-full">
                      {vr}
                    </span>
                  ) : (
                    "—"
                  )}
                </td>
                <td className="px-4 py-2 text-right text-gray-300">
                  {r.epoch_count}
                </td>
                <td className="px-4 py-2 text-right font-mono text-gray-300">
                  {fmt(r.best_train_loss)}
                </td>
                <td className="px-4 py-2 text-right font-mono text-gray-300">
                  {fmt(r.best_val_loss)}
                </td>
                <td className="px-4 py-2 text-gray-500 text-xs">
                  {r.created_at ? new Date(r.created_at).toLocaleString() : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
