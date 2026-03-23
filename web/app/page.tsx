"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";

const ModelFormModal = dynamic(() => import("./components/ModelFormModal"), { ssr: false });
const DeleteModal = dynamic(() => import("./components/DeleteModal"), { ssr: false });

// ── Types ─────────────────────────────────────────────────────────────────────

interface ModelRow {
  modelName: string;
  hasConfig: boolean;
  hasLogs: boolean;
  versionCount: number;
  latestVersion: string | null;
  latestModified: string | null;
  latestLoss: { train: number | null; valid: number | null } | null;
  totalEpochs: number;
  status: "running" | "completed" | "no_logs" | "unknown";
  processInfo: { pid: number; startedAt: string } | null;
  stdoutTail: string;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: ModelRow["status"] }) {
  const map = {
    running: "bg-green-500/20 text-green-400 border-green-500/30",
    completed: "bg-blue-500/20 text-blue-400 border-blue-500/30",
    no_logs: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
    unknown: "bg-gray-500/20 text-gray-400 border-gray-500/30",
  };
  const label = { running: "running", completed: "completed", no_logs: "not trained", unknown: "unknown" };
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full border ${map[status]}`}>
      {label[status]}
    </span>
  );
}

function fmt(n: number | null | undefined): string {
  if (n == null) return "—";
  return n.toExponential(4);
}

function fmtDate(iso: string | null): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleString();
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function HomePage() {
  const [models, setModels] = useState<ModelRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  const [showCreate, setShowCreate] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [trainingAction, setTrainingAction] = useState<Record<string, boolean>>({});
  const [trainingError, setTrainingError] = useState<Record<string, string>>({});
  const [expandedLog, setExpandedLog] = useState<Record<string, boolean>>({});

  const fetchModels = useCallback(async () => {
    try {
      const res = await fetch("/api/models", { cache: "no-store" });
      const data = await res.json();
      setModels(data.models ?? []);
      setLastRefresh(new Date());
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchModels();
    const id = setInterval(fetchModels, 8_000);
    return () => clearInterval(id);
  }, [fetchModels]);

  async function handleTrain(modelName: string, action: "start" | "stop") {
    setTrainingAction((s) => ({ ...s, [modelName]: true }));
    setTrainingError((s) => ({ ...s, [modelName]: "" }));
    try {
      const res = await fetch(`/api/training/${encodeURIComponent(modelName)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action }),
      });
      const data = await res.json();
      if (!res.ok) {
        setTrainingError((s) => ({ ...s, [modelName]: data.error ?? "Error" }));
      } else {
        // Immediately refetch to update status
        await fetchModels();
      }
    } finally {
      setTrainingAction((s) => ({ ...s, [modelName]: false }));
    }
  }

  const anyRunning = models.some((m) => m.status === "running");

  return (
    <div>
      {/* Header row */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Models</h1>
          {anyRunning && (
            <p className="text-xs text-green-400 mt-0.5">
              {models.filter((m) => m.status === "running").length} training in progress
            </p>
          )}
        </div>
        <div className="flex items-center gap-3">
          {lastRefresh && (
            <span className="text-xs text-gray-500">
              {lastRefresh.toLocaleTimeString()}
            </span>
          )}
          <button
            onClick={fetchModels}
            className="text-sm px-3 py-1.5 bg-gray-800 hover:bg-gray-700 rounded-md transition-colors"
          >
            Refresh
          </button>
          <button
            onClick={() => setShowCreate(true)}
            className="text-sm px-4 py-1.5 bg-blue-600 hover:bg-blue-500 rounded-md transition-colors font-medium"
          >
            + New Model
          </button>
        </div>
      </div>

      {/* Table */}
      {loading ? (
        <p className="text-gray-500">Loading…</p>
      ) : models.length === 0 ? (
        <div className="rounded-lg border border-gray-800 p-8 text-center text-gray-500">
          <p className="text-lg mb-2">No models yet</p>
          <p className="text-sm mb-4">
            Click <span className="text-white">+ New Model</span> to create your first training config.
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-gray-800">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 text-gray-400 text-left">
                <th className="px-4 py-3 font-medium">Model</th>
                <th className="px-4 py-3 font-medium">Status</th>
                <th className="px-4 py-3 font-medium">Epochs</th>
                <th className="px-4 py-3 font-medium">Train Loss</th>
                <th className="px-4 py-3 font-medium">Valid Loss</th>
                <th className="px-4 py-3 font-medium">Last Updated</th>
                <th className="px-4 py-3 font-medium text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <>
                <tr
                  key={m.modelName}
                  className="border-b border-gray-800 last:border-0 hover:bg-gray-900/50 transition-colors"
                >
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      {m.hasLogs ? (
                        <Link
                          href={`/models/${encodeURIComponent(m.modelName)}`}
                          className="font-medium text-blue-400 hover:text-blue-300 hover:underline"
                        >
                          {m.modelName}
                        </Link>
                      ) : (
                        <span className="font-medium text-gray-300">{m.modelName}</span>
                      )}
                      {!m.hasConfig && (
                        <span className="text-xs text-gray-600 italic">(no config)</span>
                      )}
                    </div>
                    {trainingError[m.modelName] && (
                      <p className="text-xs text-red-400 mt-0.5">{trainingError[m.modelName]}</p>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2 flex-wrap">
                      <StatusBadge status={m.status} />
                      {/* Log toggle button — shown when there's a stdout log */}
                      {m.stdoutTail && (
                        <button
                          onClick={() => setExpandedLog((s) => ({ ...s, [m.modelName]: !s[m.modelName] }))}
                          className="text-xs text-gray-500 hover:text-yellow-400 transition-colors"
                          title="Show process output"
                        >
                          {expandedLog[m.modelName] ? "▲ log" : "▼ log"}
                        </button>
                      )}
                    </div>
                    {m.status === "running" && m.processInfo && (
                      <p className="text-xs text-gray-500 mt-0.5">PID {m.processInfo.pid}</p>
                    )}
                  </td>
                  <td className="px-4 py-3 text-gray-300">{m.totalEpochs || "—"}</td>
                  <td className="px-4 py-3 font-mono text-gray-300">{fmt(m.latestLoss?.train)}</td>
                  <td className="px-4 py-3 font-mono text-gray-300">{fmt(m.latestLoss?.valid)}</td>
                  <td className="px-4 py-3 text-gray-400 text-xs">{fmtDate(m.latestModified)}</td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2 justify-end">
                      {/* Train / Stop button */}
                      {m.hasConfig && (
                        m.status === "running" ? (
                          <button
                            onClick={() => handleTrain(m.modelName, "stop")}
                            disabled={trainingAction[m.modelName]}
                            className="text-xs px-3 py-1.5 bg-red-600/80 hover:bg-red-600 disabled:opacity-50 rounded-md transition-colors font-medium"
                          >
                            {trainingAction[m.modelName] ? "…" : "Stop"}
                          </button>
                        ) : (
                          <button
                            onClick={() => handleTrain(m.modelName, "start")}
                            disabled={trainingAction[m.modelName]}
                            className="text-xs px-3 py-1.5 bg-green-700/80 hover:bg-green-700 disabled:opacity-50 rounded-md transition-colors font-medium"
                          >
                            {trainingAction[m.modelName] ? "…" : m.hasLogs ? "Retrain" : "Train"}
                          </button>
                        )
                      )}

                      {/* Delete */}
                      <button
                        onClick={() => setDeleteTarget(m.modelName)}
                        className="text-xs px-3 py-1.5 bg-gray-800 hover:bg-gray-700 rounded-md transition-colors text-gray-400 hover:text-red-400"
                      >
                        Delete
                      </button>
                    </div>
                  </td>
                </tr>
                {/* Expandable stdout log row */}
                {expandedLog[m.modelName] && m.stdoutTail && (
                  <tr key={`${m.modelName}-log`} className="border-b border-gray-800 bg-gray-950">
                    <td colSpan={7} className="px-4 py-3">
                      <pre className="text-xs text-gray-400 whitespace-pre-wrap break-all max-h-48 overflow-y-auto font-mono">
                        {m.stdoutTail}
                      </pre>
                    </td>
                  </tr>
                )}
                </>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Modals */}
      {showCreate && (
        <ModelFormModal
          onClose={() => setShowCreate(false)}
          onCreated={(name) => {
            setShowCreate(false);
            fetchModels();
            // Optionally auto-navigate
            console.log("Created:", name);
          }}
        />
      )}

      {deleteTarget && (
        <DeleteModal
          modelName={deleteTarget}
          onClose={() => setDeleteTarget(null)}
          onDeleted={() => {
            setDeleteTarget(null);
            fetchModels();
          }}
        />
      )}
    </div>
  );
}
