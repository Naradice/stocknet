"use client";

import { useState } from "react";
import {
  Datasource,
  CsvDatasource,
  SimulatorDatasource,
  DatasetKey,
  SimKey,
  csvToDatasetConfig,
  simToDatasetConfig,
} from "@/lib/datasourceTypes";
import FilePicker from "./FilePicker";

interface Props {
  onClose: () => void;
  onSaved: (ds: Datasource) => void;
  initial?: Datasource;
}

const CSV_KEYS: DatasetKey[] = ["seq2seq", "seq2seq_time", "seq2seq_did"];
const SIM_KEYS: SimKey[] = ["seq2seq_sim", "seq2seq_sim_time"];
const SAMPLER_RULES = ["MIN", "5min", "15min", "30min", "H", "D"];

const defaultCsv: Omit<CsvDatasource, "name"> = {
  type: "csv",
  description: "",
  datasetKey: "seq2seq",
  filePath: "",
  columns: ["open", "high", "low", "close"],
  observationLength: 60,
  predictionLength: 10,
};

const defaultSim: Omit<SimulatorDatasource, "name"> = {
  type: "simulator",
  description: "",
  datasetKey: "seq2seq_sim",
  agentPerModel: 300,
  modelCount: 4,
  outputLength: 1000,
  samplerRule: "MIN",
  modelConfig: {
    max_volatility: 0.02,
    min_volatility: 0.01,
    trade_unit: 0.001,
    initial_price: 100,
    spread: 1,
  },
};

export default function DatasourceFormModal({ onClose, onSaved, initial }: Props) {
  const isEdit = initial != null;

  const [name, setName] = useState(initial?.name ?? "");
  const [dsType, setDsType] = useState<"csv" | "simulator">(initial?.type ?? "csv");
  const [csv, setCsv] = useState<Omit<CsvDatasource, "name">>(
    initial?.type === "csv" ? { ...initial } : { ...defaultCsv }
  );
  const [sim, setSim] = useState<Omit<SimulatorDatasource, "name">>(
    initial?.type === "simulator" ? { ...initial } : { ...defaultSim }
  );

  const [tab, setTab] = useState<"form" | "json">("form");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showFilePicker, setShowFilePicker] = useState(false);

  function setSimCfg(key: keyof SimulatorDatasource["modelConfig"], raw: string) {
    const val = parseFloat(raw);
    if (!isNaN(val))
      setSim((s) => ({ ...s, modelConfig: { ...s.modelConfig, [key]: val } }));
  }

  function columnsStr(cols: string[]) { return cols.join(","); }
  function parseCols(raw: string) { return raw.split(",").map((c) => c.trim()).filter(Boolean); }

  const builtDs: Datasource = dsType === "csv" ? { ...csv, name } : { ...sim, name };

  const previewJson =
    dsType === "csv"
      ? JSON.stringify(csvToDatasetConfig(builtDs as CsvDatasource), null, 2)
      : JSON.stringify(simToDatasetConfig(builtDs as SimulatorDatasource), null, 2);

  async function handleSave() {
    if (!name) { setError("Name is required"); return; }
    if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
      setError("Name: letters, numbers, underscores, hyphens only"); return;
    }
    if (dsType === "csv" && !csv.filePath) { setError("File path is required"); return; }

    setSaving(true);
    setError(null);
    try {
      const url = isEdit
        ? `/api/datasources/${encodeURIComponent(name)}`
        : "/api/datasources";
      const res = await fetch(url, {
        method: isEdit ? "PUT" : "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(builtDs),
      });
      const data = await res.json();
      if (!res.ok) { setError(data.error ?? "Save failed"); return; }
      onSaved(builtDs);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <div className="bg-gray-900 border border-gray-700 rounded-xl w-full max-w-2xl max-h-[90vh] flex flex-col">

        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-800">
          <h2 className="text-lg font-semibold">
            {isEdit ? "Edit Data Source" : "Register Data Source"}
          </h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white text-xl leading-none">×</button>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 px-6 pt-4 flex-shrink-0">
          {(["form", "json"] as const).map((t) => (
            <button key={t} onClick={() => setTab(t)}
              className={`text-sm px-4 py-1.5 rounded-md transition-colors ${
                tab === t ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-400 hover:text-white"
              }`}>
              {t === "form" ? "Form" : "Dataset Config Preview"}
            </button>
          ))}
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-5">
          {tab === "json" ? (
            <div>
              <p className="text-xs text-gray-500 mb-2">
                This is the <code className="text-green-400">dataset</code> section generated in training configs using this source. Preprocessing is configured per model at training time.
              </p>
              <pre className="text-xs text-green-400 bg-gray-950 rounded-lg p-4 overflow-x-auto">
                {previewJson}
              </pre>
            </div>
          ) : (
            <>
              {/* Identity */}
              <Section title="Identity">
                <Field label="Name *">
                  <input
                    className={inp}
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="usdjpy_30min"
                    disabled={isEdit}
                  />
                  {!isEdit && (
                    <p className="text-xs text-gray-500 mt-1">Letters, numbers, underscores, hyphens. Cannot be changed later.</p>
                  )}
                </Field>
                <Field label="Description">
                  <input
                    className={inp}
                    value={dsType === "csv" ? (csv.description ?? "") : (sim.description ?? "")}
                    onChange={(e) => {
                      if (dsType === "csv") setCsv((c) => ({ ...c, description: e.target.value }));
                      else setSim((s) => ({ ...s, description: e.target.value }));
                    }}
                    placeholder="USDJPY 30-min candles 2020–2024"
                  />
                </Field>
                <Field label="Source Type">
                  <div className="flex gap-4">
                    {(["csv", "simulator"] as const).map((t) => (
                      <label key={t} className="flex items-center gap-2 text-sm cursor-pointer">
                        <input
                          type="radio"
                          name="dsType"
                          value={t}
                          checked={dsType === t}
                          onChange={() => setDsType(t)}
                          disabled={isEdit}
                          className="accent-blue-500"
                        />
                        {t === "csv" ? "CSV File" : "Simulator"}
                      </label>
                    ))}
                  </div>
                </Field>
              </Section>

              {/* CSV config */}
              {dsType === "csv" && (
                <Section title="CSV Configuration">
                  <Field label="Dataset Key">
                    <select className={inp} value={csv.datasetKey}
                      onChange={(e) => setCsv((c) => ({ ...c, datasetKey: e.target.value as DatasetKey }))}>
                      {CSV_KEYS.map((k) => (
                        <option key={k} value={k}>{k}</option>
                      ))}
                    </select>
                    <p className="text-xs text-gray-500 mt-1">
                      seq2seq = standard · seq2seq_time = time-aware · seq2seq_did = diff+cluster-ID
                    </p>
                  </Field>
                  <Field label="File Path *">
                    <div className="flex gap-2">
                      <input
                        className={inp}
                        value={csv.filePath}
                        onChange={(e) => setCsv((c) => ({ ...c, filePath: e.target.value }))}
                        placeholder="C:/data/ohlcv.csv"
                      />
                      <button type="button" onClick={() => setShowFilePicker(true)}
                        className="shrink-0 px-3 py-1.5 text-sm bg-gray-700 hover:bg-gray-600 border border-gray-600 rounded-md">
                        Browse…
                      </button>
                    </div>
                  </Field>
                  <Field label="Columns (comma-separated)">
                    <input className={inp} value={columnsStr(csv.columns)}
                      onChange={(e) => setCsv((c) => ({ ...c, columns: parseCols(e.target.value) }))}
                      placeholder="open,high,low,close"
                    />
                  </Field>
                  <div className="grid grid-cols-2 gap-4">
                    <Field label="Observation Length">
                      <input type="number" className={inp} value={csv.observationLength} min={1}
                        onChange={(e) => setCsv((c) => ({ ...c, observationLength: +e.target.value }))}
                      />
                    </Field>
                    <Field label="Prediction Length">
                      <input type="number" className={inp} value={csv.predictionLength} min={1}
                        onChange={(e) => setCsv((c) => ({ ...c, predictionLength: +e.target.value }))}
                      />
                    </Field>
                  </div>
                </Section>
              )}

              {/* Simulator config */}
              {dsType === "simulator" && (
                <>
                  <Section title="Simulator Configuration">
                    <Field label="Dataset Key">
                      <select className={inp} value={sim.datasetKey}
                        onChange={(e) => setSim((s) => ({ ...s, datasetKey: e.target.value as SimKey }))}>
                        {SIM_KEYS.map((k) => (
                          <option key={k} value={k}>{k}</option>
                        ))}
                      </select>
                      <p className="text-xs text-gray-500 mt-1">
                        seq2seq_sim = standard · seq2seq_sim_time = weekly time-aware
                      </p>
                    </Field>
                    <div className="grid grid-cols-2 gap-4">
                      <Field label="Agents per Model">
                        <input type="number" className={inp} value={sim.agentPerModel} min={1}
                          onChange={(e) => setSim((s) => ({ ...s, agentPerModel: +e.target.value }))}
                        />
                      </Field>
                      <Field label="Model Count (parallel sims)">
                        <input type="number" className={inp} value={sim.modelCount} min={1}
                          onChange={(e) => setSim((s) => ({ ...s, modelCount: +e.target.value }))}
                        />
                      </Field>
                      <Field label="Output Length (candles/batch)">
                        <input type="number" className={inp} value={sim.outputLength} min={1}
                          onChange={(e) => setSim((s) => ({ ...s, outputLength: +e.target.value }))}
                        />
                      </Field>
                      <Field label="Sampler Rule">
                        <select className={inp} value={sim.samplerRule}
                          onChange={(e) => setSim((s) => ({ ...s, samplerRule: e.target.value }))}>
                          {SAMPLER_RULES.map((r) => (
                            <option key={r} value={r}>{r}</option>
                          ))}
                        </select>
                      </Field>
                    </div>
                  </Section>

                  <Section title="Market Model Config">
                    <div className="grid grid-cols-2 gap-4">
                      <Field label="Max Volatility">
                        <input type="number" className={inp} step={0.001} value={sim.modelConfig.max_volatility}
                          onChange={(e) => setSimCfg("max_volatility", e.target.value)}
                        />
                      </Field>
                      <Field label="Min Volatility">
                        <input type="number" className={inp} step={0.001} value={sim.modelConfig.min_volatility}
                          onChange={(e) => setSimCfg("min_volatility", e.target.value)}
                        />
                      </Field>
                      <Field label="Trade Unit">
                        <input type="number" className={inp} step={0.0001} value={sim.modelConfig.trade_unit}
                          onChange={(e) => setSimCfg("trade_unit", e.target.value)}
                        />
                      </Field>
                      <Field label="Initial Price">
                        <input type="number" className={inp} step={1} value={sim.modelConfig.initial_price}
                          onChange={(e) => setSimCfg("initial_price", e.target.value)}
                        />
                      </Field>
                      <Field label="Spread">
                        <input type="number" className={inp} step={0.1} value={sim.modelConfig.spread}
                          onChange={(e) => setSimCfg("spread", e.target.value)}
                        />
                      </Field>
                    </div>
                  </Section>
                </>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-800 flex items-center justify-between flex-shrink-0">
          {error ? <p className="text-red-400 text-sm">{error}</p> : <span />}
          <div className="flex gap-3">
            <button onClick={onClose}
              className="px-4 py-2 text-sm bg-gray-800 hover:bg-gray-700 rounded-md transition-colors">
              Cancel
            </button>
            <button onClick={handleSave} disabled={saving}
              className="px-4 py-2 text-sm bg-blue-600 hover:bg-blue-500 disabled:opacity-50 rounded-md transition-colors font-medium">
              {saving ? "Saving…" : isEdit ? "Update" : "Register"}
            </button>
          </div>
        </div>
      </div>

      {showFilePicker && (
        <FilePicker
          accept={[".csv"]}
          onSelect={(p) => setCsv((c) => ({ ...c, filePath: p }))}
          onClose={() => setShowFilePicker(false)}
        />
      )}
    </div>
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const inp =
  "w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-1.5 text-sm focus:outline-none focus:border-blue-500";

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-3">{title}</h3>
      <div className="space-y-3">{children}</div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-xs text-gray-400 mb-1">{label}</label>
      {children}
    </div>
  );
}
