"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface DbRun {
  id: number;
  model_name: string;
  version: string;
  created_at: string | null;
  metadata: { volume_rate?: number } | null;
  epoch_count: number;
  best_train_loss: number | null;
  best_val_loss: number | null;
}

export interface EpochEntry {
  epoch: number;
  trainLoss: number | null;
  validLoss: number | null;
}

export type LogsMap = Record<string, EpochEntry[]>;

// ── Palette ───────────────────────────────────────────────────────────────────

export const COLORS = [
  "#3b82f6", "#ef4444", "#22c55e", "#f97316",
  "#a855f7", "#14b8a6", "#ec4899", "#eab308",
];

export function pickColor(
  volumeRate: number | undefined,
  allRates: number[],
  idx: number
): string {
  if (volumeRate != null && allRates.length) {
    const sorted = Array.from(new Set(allRates)).sort((a, b) => a - b);
    const i = sorted.indexOf(volumeRate);
    if (i >= 0) return COLORS[i % COLORS.length];
  }
  return COLORS[idx % COLORS.length];
}

export function shortLabel(run: DbRun): string {
  const vr = run.metadata?.volume_rate;
  return vr != null ? `vol ${vr}` : run.version;
}

// ── Data builders ─────────────────────────────────────────────────────────────

export function buildUnifiedData(
  selected: DbRun[],
  logs: LogsMap
): Record<string, unknown>[] {
  const maxEpochs = Math.max(
    ...selected.map((r) => (logs[r.id] ?? []).length),
    0
  );
  return Array.from({ length: maxEpochs }, (_, i) => {
    const ep = i + 1;
    const row: Record<string, unknown> = { epoch: ep };
    for (const run of selected) {
      const entry = (logs[run.id] ?? []).find((e) => e.epoch === ep);
      const base = shortLabel(run);
      row[`${base}:train`] = entry?.trainLoss ?? null;
      row[`${base}:val`] = entry?.validLoss ?? null;
    }
    return row;
  });
}

export function buildVolData(
  selected: DbRun[],
  allRates: number[]
): Record<string, unknown>[] {
  const models = Array.from(new Set(selected.map((r) => r.model_name)));
  return allRates.map((vr) => {
    const row: Record<string, unknown> = { volume_rate: vr };
    for (const model of models) {
      const run = selected.find(
        (r) => r.model_name === model && r.metadata?.volume_rate === vr
      );
      row[model] = run?.best_val_loss ?? null;
    }
    return row;
  });
}

// ── Shared chart style ─────────────────────────────────────────────────────────

const GRID = { strokeDasharray: "3 3", stroke: "#374151" } as const;
const AXIS = { stroke: "#6b7280", tick: { fontSize: 11 } } as const;
export const TOOLTIP_STYLE = {
  contentStyle: {
    background: "#1f2937",
    border: "1px solid #374151",
    borderRadius: 6,
  },
  labelStyle: { color: "#9ca3af", fontSize: 11 },
  itemStyle: { fontSize: 11 },
} as const;

export function yAxisProps(logScale: boolean) {
  return {
    ...AXIS,
    width: 72,
    scale: logScale ? ("log" as const) : ("auto" as const),
    domain: ["auto", "auto"] as [string, string],
    allowDataOverflow: logScale,
    tickFormatter: (v: number) => v.toExponential(1),
  };
}

// ── Loss Curves (train + val on same chart) ───────────────────────────────────

export function LossCurvesChart({
  data,
  runs,
  allRates,
  logScale,
}: {
  data: Record<string, unknown>[];
  runs: DbRun[];
  allRates: number[];
  logScale: boolean;
}) {
  if (!data.length)
    return (
      <div className="flex items-center justify-center h-48 text-gray-600 text-sm">
        No data
      </div>
    );

  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={data} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid {...GRID} />
        <XAxis
          dataKey="epoch"
          {...AXIS}
          label={{
            value: "Epoch",
            position: "insideBottomRight",
            offset: -4,
            fontSize: 11,
            fill: "#6b7280",
          }}
        />
        <YAxis {...yAxisProps(logScale)} />
        <Tooltip
          {...TOOLTIP_STYLE}
          formatter={(v: number, name: string) => [v.toExponential(4), name]}
        />
        <Legend wrapperStyle={{ fontSize: 11, paddingTop: 8 }} />
        {runs.map((run, i) => {
          const color = pickColor(run.metadata?.volume_rate, allRates, i);
          const base = shortLabel(run);
          return [
            <Line
              key={`${run.id}-train`}
              type="monotone"
              dataKey={`${base}:train`}
              name={`${base} train`}
              stroke={color}
              strokeWidth={2}
              dot={false}
              connectNulls
            />,
            <Line
              key={`${run.id}-val`}
              type="monotone"
              dataKey={`${base}:val`}
              name={`${base} val`}
              stroke={color}
              strokeWidth={2}
              strokeDasharray="6 3"
              dot={false}
              connectNulls
            />,
          ];
        })}
      </LineChart>
    </ResponsiveContainer>
  );
}

// ── Best Val Loss by Volume ────────────────────────────────────────────────────

export function BestVolChart({
  data,
  runs,
  logScale,
}: {
  data: Record<string, unknown>[];
  runs: DbRun[];
  logScale: boolean;
}) {
  const models = Array.from(new Set(runs.map((r) => r.model_name)));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data} margin={{ top: 4, right: 16, left: 0, bottom: 8 }}>
        <CartesianGrid {...GRID} />
        <XAxis
          dataKey="volume_rate"
          {...AXIS}
          type="number"
          domain={[0, 1]}
          tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
          label={{
            value: "Data Volume",
            position: "insideBottomRight",
            offset: -4,
            fontSize: 11,
            fill: "#6b7280",
          }}
        />
        <YAxis
          {...yAxisProps(logScale)}
          label={{
            value: "Best Val Loss",
            angle: -90,
            position: "insideLeft",
            offset: 8,
            fontSize: 11,
            fill: "#6b7280",
          }}
        />
        <Tooltip
          {...TOOLTIP_STYLE}
          labelFormatter={(v: number) => `Volume: ${Math.round(v * 100)}%`}
          formatter={(v: number, name: string) => [v.toExponential(4), name]}
        />
        <Legend wrapperStyle={{ fontSize: 11, paddingTop: 8 }} />
        {models.map((model, i) => (
          <Line
            key={model}
            type="monotone"
            dataKey={model}
            name={model}
            stroke={COLORS[i % COLORS.length]}
            strokeWidth={2}
            dot={{ r: 5, strokeWidth: 0 }}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
