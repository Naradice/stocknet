"use client";

import { useEffect, useState } from "react";
import DatasourceFormModal from "@/app/components/DatasourceFormModal";
import { Datasource, CsvDatasource, SimulatorDatasource } from "@/lib/datasourceTypes";

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

// ── Datasource card ────────────────────────────────────────────────────────────

function DatasourceCard({
  ds,
  onEdit,
  onDelete,
}: {
  ds: Datasource;
  onEdit: () => void;
  onDelete: () => void;
}) {
  const [confirmDelete, setConfirmDelete] = useState(false);

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-5 flex flex-col gap-3">
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-semibold text-white truncate">{ds.name}</span>
            <TypeBadge type={ds.type} />
            <span className="text-xs text-gray-500 font-mono">{ds.datasetKey}</span>
          </div>
          {ds.description && (
            <p className="text-sm text-gray-400 mt-0.5 truncate">{ds.description}</p>
          )}
        </div>
        <div className="flex gap-2 flex-shrink-0">
          <button onClick={onEdit}
            className="text-xs px-2.5 py-1 bg-gray-800 hover:bg-gray-700 rounded-md transition-colors">
            Edit
          </button>
          {confirmDelete ? (
            <div className="flex gap-1">
              <button onClick={onDelete}
                className="text-xs px-2.5 py-1 bg-red-700 hover:bg-red-600 rounded-md transition-colors">
                Confirm
              </button>
              <button onClick={() => setConfirmDelete(false)}
                className="text-xs px-2.5 py-1 bg-gray-800 hover:bg-gray-700 rounded-md transition-colors">
                Cancel
              </button>
            </div>
          ) : (
            <button onClick={() => setConfirmDelete(true)}
              className="text-xs px-2.5 py-1 bg-gray-800 hover:bg-red-900/50 text-gray-400 hover:text-red-300 rounded-md transition-colors">
              Delete
            </button>
          )}
        </div>
      </div>

      {/* Details */}
      {ds.type === "csv" && <CsvDetails ds={ds as CsvDatasource} />}
      {ds.type === "simulator" && <SimDetails ds={ds as SimulatorDatasource} />}
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

  async function load() {
    const res = await fetch("/api/datasources", { cache: "no-store" });
    if (res.ok) setDatasources(await res.json());
    setLoading(false);
  }

  useEffect(() => { load(); }, []);

  async function handleDelete(name: string) {
    await fetch(`/api/datasources/${encodeURIComponent(name)}`, { method: "DELETE" });
    load();
  }

  function handleSaved() {
    setShowModal(false);
    setEditing(null);
    load();
  }

  const csvCount = datasources.filter((d) => d.type === "csv").length;
  const simCount = datasources.filter((d) => d.type === "simulator").length;

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Data Sources</h1>
          <p className="text-gray-500 text-sm mt-1">
            {loading ? "Loading…" : `${csvCount} CSV · ${simCount} Simulator`}
          </p>
        </div>
        <button
          onClick={() => { setEditing(null); setShowModal(true); }}
          className="px-4 py-2 text-sm bg-blue-600 hover:bg-blue-500 rounded-md font-medium transition-colors"
        >
          + Register
        </button>
      </div>

      {/* Empty state */}
      {!loading && datasources.length === 0 && (
        <div className="flex flex-col items-center justify-center py-24 text-gray-600 gap-3">
          <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M4 7h16M4 12h16M4 17h8" />
          </svg>
          <p className="text-sm">No data sources registered yet.</p>
          <button onClick={() => setShowModal(true)}
            className="text-blue-400 hover:text-blue-300 text-sm transition-colors">
            Register your first data source →
          </button>
        </div>
      )}

      {/* Cards grid */}
      {datasources.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {datasources.map((ds) => (
            <DatasourceCard
              key={ds.name}
              ds={ds}
              onEdit={() => { setEditing(ds); setShowModal(true); }}
              onDelete={() => handleDelete(ds.name)}
            />
          ))}
        </div>
      )}

      {/* Modal */}
      {showModal && (
        <DatasourceFormModal
          onClose={() => { setShowModal(false); setEditing(null); }}
          onSaved={handleSaved}
          initial={editing ?? undefined}
        />
      )}
    </div>
  );
}
