"use client";

import { useState, useEffect } from "react";
import { buildConfig, defaultValues, FormValues, ScaleCombination } from "@/lib/configTemplate";
import FilePicker from "./FilePicker";
import { CsvDatasource } from "@/lib/datasourceTypes";

interface Props {
  onClose: () => void;
  onCreated: (name: string) => void;
  /** Pre-fill for edit mode */
  initialName?: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  initialConfig?: Record<string, any>;
}

const MODEL_TYPES: FormValues["modelType"][] = [
  "Seq2SeqTransformer",
  "LSTM",
  "MeanVarianceTransformer",
];
const OPTIMIZERS = ["AdamW", "Adam", "SGD", "RMSprop"];
const LOSSES = ["MSELoss", "L1Loss", "SmoothL1Loss", "GaussianNLLLoss"];
const DEVICES = ["auto", "cpu", "cuda"];

export default function ModelFormModal({
  onClose,
  onCreated,
  initialName,
}: Props) {
  const [values, setValues] = useState<FormValues>({
    ...defaultValues,
    modelName: initialName ?? "",
  });
  const [tab, setTab] = useState<"form" | "json">("form");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showFilePicker, setShowFilePicker] = useState(false);
  const [csvSources, setCsvSources] = useState<CsvDatasource[]>([]);
  const [selectedSource, setSelectedSource] = useState("");

  useEffect(() => {
    fetch("/api/datasources")
      .then((r) => r.json())
      .then((list) => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const csv = (list as any[]).filter((d) => d.type === "csv") as CsvDatasource[];
        setCsvSources(csv);
      })
      .catch(() => {});
  }, []);

  function applyDatasource(name: string) {
    const ds = csvSources.find((d) => d.name === name);
    if (!ds) return;
    setValues((v) => ({
      ...v,
      dataSourcePath: ds.filePath,
      columns: ds.columns.join(","),
      observationLength: ds.observationLength,
      predictionLength: ds.predictionLength,
    }));
    setSelectedSource(name);
  }

  function set<K extends keyof FormValues>(key: K, val: FormValues[K]) {
    setValues((v) => ({ ...v, [key]: val }));
  }

  function updateScale(idx: number, field: keyof ScaleCombination, raw: string) {
    const val = parseFloat(raw);
    if (isNaN(val)) return;
    setValues((v) => {
      const next = v.scaleCombinations.map((s, i) =>
        i === idx ? { ...s, [field]: field === "batchSize" ? Math.round(val) : val } : s
      );
      return { ...v, scaleCombinations: next };
    });
  }

  function addScale() {
    setValues((v) => ({
      ...v,
      scaleCombinations: [...v.scaleCombinations, { volumeRate: 1.0, batchSize: 32 }],
    }));
  }

  function removeScale(idx: number) {
    setValues((v) => ({
      ...v,
      scaleCombinations: v.scaleCombinations.filter((_, i) => i !== idx),
    }));
  }

  function setProcess(idx: number, val: string) {
    setValues((v) => {
      const next = [...v.processes];
      next[idx] = val;
      return { ...v, processes: next };
    });
  }

  function addProcess() {
    setValues((v) => ({ ...v, processes: [...v.processes, ""] }));
  }

  function removeProcess(idx: number) {
    setValues((v) => ({ ...v, processes: v.processes.filter((_, i) => i !== idx) }));
  }

  const generatedConfig = buildConfig(values);
  const configJson = JSON.stringify(generatedConfig, null, 2);

  async function handleSave() {
    if (!values.modelName) { setError("Model name is required"); return; }
    if (!values.dataSourcePath) { setError("Data source path is required"); return; }
    setSaving(true);
    setError(null);
    try {
      const res = await fetch("/api/configs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: values.modelName, config: generatedConfig }),
      });
      const data = await res.json();
      if (!res.ok) { setError(data.error ?? "Save failed"); return; }
      onCreated(values.modelName);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <div className="bg-gray-900 border border-gray-700 rounded-xl w-full max-w-3xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-800">
          <h2 className="text-lg font-semibold">New Model</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white text-xl leading-none">×</button>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 px-6 pt-4">
          {(["form", "json"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`text-sm px-4 py-1.5 rounded-md transition-colors ${
                tab === t ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-400 hover:text-white"
              }`}
            >
              {t === "form" ? "Form" : "Preview JSON"}
            </button>
          ))}
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {tab === "json" ? (
            <pre className="text-xs text-green-400 bg-gray-950 rounded-lg p-4 overflow-x-auto">
              {configJson}
            </pre>
          ) : (
            <div className="space-y-6">
              {/* Basic */}
              <Section title="Basic">
                <Field label="Model Name *">
                  <input
                    className={input}
                    value={values.modelName}
                    onChange={(e) => set("modelName", e.target.value)}
                    placeholder="my_transformer"
                    disabled={!!initialName}
                  />
                  <p className="text-xs text-gray-500 mt-1">Letters, numbers, underscores, hyphens only.</p>
                </Field>
                <Field label="Model Type">
                  <select className={input} value={values.modelType} onChange={(e) => set("modelType", e.target.value as FormValues["modelType"])}>
                    {MODEL_TYPES.map((t) => <option key={t}>{t}</option>)}
                  </select>
                </Field>
              </Section>

              {/* Data */}
              <Section title="Data">
                {csvSources.length > 0 && (
                  <Field label="Load from registered data source">
                    <div className="flex gap-2">
                      <select
                        className={input}
                        value={selectedSource}
                        onChange={(e) => applyDatasource(e.target.value)}
                      >
                        <option value="">— pick a registered source —</option>
                        {csvSources.map((ds) => (
                          <option key={ds.name} value={ds.name}>
                            {ds.name}{ds.description ? ` — ${ds.description}` : ""}
                          </option>
                        ))}
                      </select>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      Auto-fills path, columns, lengths, and preprocessing. You can still edit them below.
                    </p>
                  </Field>
                )}
                <Field label="Data Source File Path *">
                  <div className="flex gap-2">
                    <input
                      className={input}
                      value={values.dataSourcePath}
                      onChange={(e) => set("dataSourcePath", e.target.value)}
                      placeholder="C:/data/ohlcv.csv"
                    />
                    <button
                      type="button"
                      onClick={() => setShowFilePicker(true)}
                      className="shrink-0 px-3 py-1.5 text-sm bg-gray-700 hover:bg-gray-600 border border-gray-600 rounded-md transition-colors"
                      title="Browse files"
                    >
                      Browse…
                    </button>
                  </div>
                </Field>
                <Field label="Columns (comma-separated)">
                  <input
                    className={input}
                    value={values.columns}
                    onChange={(e) => set("columns", e.target.value)}
                  />
                </Field>
                <Field label="Preprocessing (applied at training time)">
                  <div className="space-y-1.5">
                    {values.processes.map((p, i) => (
                      <div key={i} className="flex gap-2 items-center">
                        <input
                          className={input}
                          value={p}
                          onChange={(e) => setProcess(i, e.target.value)}
                          placeholder='e.g. Diff, MiniMax, or {"wid":{"kinds":"wid","freq":30}}'
                        />
                        <button
                          type="button"
                          onClick={() => removeProcess(i)}
                          className="flex-shrink-0 text-gray-500 hover:text-red-400 transition-colors text-lg leading-none px-1"
                          title="Remove"
                        >
                          ×
                        </button>
                      </div>
                    ))}
                    <button
                      type="button"
                      onClick={addProcess}
                      className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
                    >
                      + Add process
                    </button>
                  </div>
                  <p className="text-xs text-gray-500 mt-1.5">
                    String names (e.g. <code className="text-gray-400">Diff</code>, <code className="text-gray-400">MiniMax</code>) or JSON for parameterized processes. Order matters.
                  </p>
                </Field>
                <div className="grid grid-cols-2 gap-4">
                  <Field label="Observation Length">
                    <input type="number" className={input} value={values.observationLength} onChange={(e) => set("observationLength", +e.target.value)} min={1} />
                  </Field>
                  <Field label="Prediction Length">
                    <input type="number" className={input} value={values.predictionLength} onChange={(e) => set("predictionLength", +e.target.value)} min={1} />
                  </Field>
                </div>

                {/* Scale combinations toggle */}
                <div className="pt-1">
                  <label className="flex items-center gap-2.5 cursor-pointer select-none">
                    <input
                      type="checkbox"
                      className="accent-blue-500 w-4 h-4"
                      checked={values.scaleEnabled}
                      onChange={(e) => set("scaleEnabled", e.target.checked)}
                    />
                    <span className="text-sm text-gray-300">Enable scale combinations <span className="text-gray-500">(train across multiple data volumes)</span></span>
                  </label>
                </div>

                {values.scaleEnabled && (
                  <div className="bg-gray-800/60 border border-gray-700 rounded-lg p-3 space-y-2">
                    <div className="grid grid-cols-[1fr_1fr_auto] gap-2 text-xs text-gray-500 px-1">
                      <span>Volume Rate (0–1)</span>
                      <span>Batch Size</span>
                      <span />
                    </div>
                    {values.scaleCombinations.map((s, i) => (
                      <div key={i} className="grid grid-cols-[1fr_1fr_auto] gap-2 items-center">
                        <input
                          type="number"
                          className={input}
                          value={s.volumeRate}
                          onChange={(e) => updateScale(i, "volumeRate", e.target.value)}
                          min={0.01} max={1} step={0.05}
                        />
                        <input
                          type="number"
                          className={input}
                          value={s.batchSize}
                          onChange={(e) => updateScale(i, "batchSize", e.target.value)}
                          min={1}
                        />
                        <button
                          type="button"
                          onClick={() => removeScale(i)}
                          disabled={values.scaleCombinations.length === 1}
                          className="text-gray-500 hover:text-red-400 disabled:opacity-30 transition-colors px-1 text-lg leading-none"
                          title="Remove"
                        >
                          ×
                        </button>
                      </div>
                    ))}
                    <button
                      type="button"
                      onClick={addScale}
                      className="text-xs text-blue-400 hover:text-blue-300 transition-colors mt-1"
                    >
                      + Add row
                    </button>
                  </div>
                )}
              </Section>

              {/* Model Architecture */}
              {values.modelType !== "LSTM" ? (
                <Section title="Transformer Architecture">
                  <div className="grid grid-cols-2 gap-4">
                    <Field label="d_model">
                      <input type="number" className={input} value={values.dModel} onChange={(e) => set("dModel", +e.target.value)} min={1} />
                    </Field>
                    <Field label="nhead">
                      <input type="number" className={input} value={values.nhead} onChange={(e) => set("nhead", +e.target.value)} min={1} />
                    </Field>
                    <Field label="Encoder Layers">
                      <input type="number" className={input} value={values.numEncoderLayers} onChange={(e) => set("numEncoderLayers", +e.target.value)} min={1} />
                    </Field>
                    <Field label="Decoder Layers">
                      <input type="number" className={input} value={values.numDecoderLayers} onChange={(e) => set("numDecoderLayers", +e.target.value)} min={1} />
                    </Field>
                    <Field label="dim_feedforward">
                      <input type="number" className={input} value={values.dimFeedforward} onChange={(e) => set("dimFeedforward", +e.target.value)} min={1} />
                    </Field>
                    <Field label="Dropout">
                      <input type="number" className={input} value={values.dropout} onChange={(e) => set("dropout", +e.target.value)} min={0} max={1} step={0.01} />
                    </Field>
                  </div>
                </Section>
              ) : (
                <Section title="LSTM Architecture">
                  <Field label="Hidden Dim">
                    <input type="number" className={input} value={values.hiddenDim} onChange={(e) => set("hiddenDim", +e.target.value)} min={1} />
                  </Field>
                </Section>
              )}

              {/* Training */}
              <Section title="Training">
                <div className="grid grid-cols-2 gap-4">
                  <Field label="Optimizer">
                    <select className={input} value={values.optimizer} onChange={(e) => set("optimizer", e.target.value)}>
                      {OPTIMIZERS.map((o) => <option key={o}>{o}</option>)}
                    </select>
                  </Field>
                  <Field label="Learning Rate">
                    <input type="number" className={input} value={values.learningRate} onChange={(e) => set("learningRate", +e.target.value)} step={0.0001} min={0} />
                  </Field>
                  <Field label="LR Scheduler Step Size">
                    <input type="number" className={input} value={values.schedulerStepSize} onChange={(e) => set("schedulerStepSize", +e.target.value)} min={1} />
                  </Field>
                  <Field label="LR Scheduler Gamma">
                    <input type="number" className={input} value={values.schedulerGamma} onChange={(e) => set("schedulerGamma", +e.target.value)} step={0.01} min={0} max={1} />
                  </Field>
                  <Field label="Loss Function">
                    <select className={input} value={values.loss} onChange={(e) => set("loss", e.target.value)}>
                      {LOSSES.map((l) => <option key={l}>{l}</option>)}
                    </select>
                  </Field>
                  <Field label="Device">
                    <select className={input} value={values.device} onChange={(e) => set("device", e.target.value)}>
                      {DEVICES.map((d) => <option key={d}>{d}</option>)}
                    </select>
                  </Field>
                  {!values.scaleEnabled && (
                    <Field label="Batch Size (comma-sep for multiple)">
                      <input className={input} value={values.batchSize} onChange={(e) => set("batchSize", e.target.value)} />
                    </Field>
                  )}
                  <Field label="Max Epochs">
                    <input type="number" className={input} value={values.epochs} onChange={(e) => set("epochs", +e.target.value)} min={1} />
                  </Field>
                  <Field label="Patience (early stop)">
                    <input type="number" className={input} value={values.patience} onChange={(e) => set("patience", +e.target.value)} min={1} />
                  </Field>
                  <Field label="Log Path">
                    <input className={input} value={values.logPath} onChange={(e) => set("logPath", e.target.value)} />
                  </Field>
                </div>
              </Section>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-800 flex items-center justify-between">
          {error && <p className="text-red-400 text-sm">{error}</p>}
          {!error && <span />}
          <div className="flex gap-3">
            <button onClick={onClose} className="px-4 py-2 text-sm bg-gray-800 hover:bg-gray-700 rounded-md transition-colors">
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={saving}
              className="px-4 py-2 text-sm bg-blue-600 hover:bg-blue-500 disabled:opacity-50 rounded-md transition-colors font-medium"
            >
              {saving ? "Saving…" : "Save Config"}
            </button>
          </div>
        </div>
      </div>

      {/* File picker — rendered outside the modal scroll area so z-index stacks above it */}
      {showFilePicker && (
        <FilePicker
          accept={[".csv"]}
          onSelect={(p) => set("dataSourcePath", p)}
          onClose={() => setShowFilePicker(false)}
        />
      )}
    </div>
  );
}

// ── Small helpers ─────────────────────────────────────────────────────────────

const input =
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
