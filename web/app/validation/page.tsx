"use client";

import { useState, useEffect } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

// ── Palette ───────────────────────────────────────────────────────────────────

const COLORS = [
  "#60a5fa", // blue-400
  "#34d399", // emerald-400
  "#f59e0b", // amber-400
  "#f87171", // red-400
  "#a78bfa", // violet-400
  "#fb923c", // orange-400
  "#22d3ee", // cyan-400
  "#4ade80", // green-400
];

// ── Types ─────────────────────────────────────────────────────────────────────

interface DsItem {
  name: string;
  type: "csv" | "simulator";
  description?: string;
}

interface SeasonalityWeeklyDay { label: string; slots: number[]; volume: (number | null)[]; }
interface SeasonalityWeekly {
  days: SeasonalityWeeklyDay[];
  counts_per_day: number; time_unit_min: number;
  time_labels: string[]; day_boundaries: number[];
}
interface SeasonalityMonthlyWeek { label: string; days: number[]; volume: (number | null)[]; }
interface SeasonalityMonthly { weeks: SeasonalityMonthlyWeek[]; day_labels: string[]; }
interface SeasonalityYearlySeries { label: string; months: number[]; volume: (number | null)[]; }
interface SeasonalityYearly { series: SeasonalityYearlySeries[]; month_labels: string[]; }
interface Seasonality { weekly: SeasonalityWeekly; monthly: SeasonalityMonthly; yearly: SeasonalityYearly; }

interface Exogenous {
  seasonality: Seasonality | null;
  intraday_seasonality: { hours: number[]; mean: number[]; std: number[] } | null;
  jump_tail: { jump_rate: number; threshold_3sigma: number; q99: number; q999: number; q001: number; q01: number } | null;
  cdf: { x: number[]; y: number[] } | null;
  rolling_mean: { index: number[]; values: number[]; window: number } | null;
  long_lag_acf: { lags: number[]; values: number[]; highlights: Record<string, number> } | null;
}

interface ValidationResult {
  name: string;
  exogenous: Exogenous | null;
  stats: { n: number; mean: number; std: number; skewness: number; kurtosis: number; hurst: number };
  ccdf: { x: number[]; y: number[] };
  acf_returns: number[];
  acf_abs_returns: number[];
  diffusion: { lags: number[]; vars: number[] };
  qq: { t: number; s: number }[];
  qq_line: { slope: number; intercept: number };
  volatility_clustering: number[];
  return_dist: {
    centers: number[];
    hist: (number | null)[];
    normal_pdf: number[];
    laplace_pdf: number[];
  };
  ccdf_hist: { x: number[]; y: (number | null)[] };
}

// ── Shared chart helpers ──────────────────────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const NoShape = (_: any): React.ReactElement => <g />;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const SmallDot = ({ cx, cy, fill }: any): React.ReactElement => (
  <circle cx={cx} cy={cy} r={2.5} fill={fill} opacity={0.85} />
);

const CHART_STYLE = {
  contentStyle: {
    background: "#111827",
    border: "1px solid #374151",
    fontSize: 10,
  },
  labelStyle: { color: "#9ca3af" },
};

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-5 space-y-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">{title}</h3>
      {children}
    </div>
  );
}

// ── Statistics section — table + return distribution chart (No.5) ─────────────

function StatsSection({ results }: { results: ValidationResult[] }) {
  return (
    <Section title="Statistics — No.5 Return Distribution">
      {/* Stats table */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="border-b border-gray-800">
              {["Datasource", "N", "Mean", "Std", "Skewness", "Kurtosis", "Hurst"].map((h, i) => (
                <th
                  key={h}
                  className={`py-2 text-gray-500 font-medium ${i === 0 ? "text-left pr-4" : "text-right px-3"}`}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {results.map((r, i) => (
              <tr key={r.name} className="border-b border-gray-900 last:border-0">
                <td className="py-2 pr-4 font-semibold" style={{ color: COLORS[i % COLORS.length] }}>
                  {r.name}
                </td>
                <td className="text-right py-2 px-3 text-gray-300 font-mono">{r.stats.n.toLocaleString()}</td>
                <td className="text-right py-2 px-3 text-gray-300 font-mono">{r.stats.mean.toExponential(2)}</td>
                <td className="text-right py-2 px-3 text-gray-300 font-mono">{r.stats.std.toExponential(2)}</td>
                <td className="text-right py-2 px-3 text-gray-300 font-mono">{r.stats.skewness.toFixed(3)}</td>
                <td className="text-right py-2 px-3 text-gray-300 font-mono">{r.stats.kurtosis.toFixed(2)}</td>
                <td className="text-right py-2 pl-3 text-gray-300 font-mono">
                  {isNaN(r.stats.hurst) ? "—" : r.stats.hurst.toFixed(3)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-gray-600 italic">Normal: skew=0, kurtosis=3, Hurst≈0.5</p>

      {/* Return distribution chart — Y pre-transformed to log10 for reliable rendering */}
      <p className="text-xs text-gray-600 mt-2">
        Dots: empirical · <span className="opacity-70">— dashed: Normal fit · ··· dotted: Laplace fit</span>
      </p>
      <ResponsiveContainer width="100%" height={280}>
        <ScatterChart margin={{ top: 4, right: 16, bottom: 24, left: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis
            type="number"
            dataKey="x"
            name="log return"
            tick={{ fontSize: 10, fill: "#6b7280" }}
            label={{ value: "log return", position: "insideBottom", offset: -12, fill: "#6b7280", fontSize: 11 }}
          />
          <YAxis
            type="number"
            dataKey="y"
            name="Density"
            tick={{ fontSize: 10, fill: "#6b7280" }}
            tickFormatter={(v: number) => `10^${v.toFixed(0)}`}
            width={56}
            label={{ value: "log Density", angle: -90, position: "insideLeft", fill: "#6b7280", fontSize: 11 }}
          />
          <Tooltip
            {...CHART_STYLE}
            formatter={(v: number) => (Math.pow(10, v)).toExponential(3)}
          />
          <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
          {results.map((r, i) => {
            const color = COLORS[i % COLORS.length];
            const lg = (v: number | null) => (v !== null && v > 0) ? Math.log10(v) : null;
            const histData = r.return_dist.centers
              .map((x, j) => { const y = lg(r.return_dist.hist[j]); return y !== null ? { x, y } : null; })
              .filter((d): d is { x: number; y: number } => d !== null);
            const normalData = r.return_dist.centers
              .map((x, j) => { const y = lg(r.return_dist.normal_pdf[j]); return y !== null ? { x, y } : null; })
              .filter((d): d is { x: number; y: number } => d !== null);
            const laplaceData = r.return_dist.centers
              .map((x, j) => { const y = lg(r.return_dist.laplace_pdf[j]); return y !== null ? { x, y } : null; })
              .filter((d): d is { x: number; y: number } => d !== null);
            return [
              <Scatter key={`${r.name}_hist`} name={r.name} data={histData} fill={color} shape={SmallDot} />,
              <Scatter key={`${r.name}_norm`} name={`${r.name} Normal`} data={normalData} fill={color}
                line={{ stroke: color, strokeWidth: 1.2, strokeDasharray: "5 3" }} shape={NoShape} legendType="none" />,
              <Scatter key={`${r.name}_lap`} name={`${r.name} Laplace`} data={laplaceData} fill={color}
                line={{ stroke: color, strokeWidth: 1.2, strokeDasharray: "2 2" }} shape={NoShape} legendType="none" />,
            ];
          })}
        </ScatterChart>
      </ResponsiveContainer>
    </Section>
  );
}


// ── CCDF chart (log-log) ──────────────────────────────────────────────────────

function CcdfChart({ results }: { results: ValidationResult[] }) {
  return (
    <Section title="Fat Tail — CCDF (log-log)">
      <p className="text-xs text-gray-600">
        P(|r| &gt; x) vs |r|. Heavy tails deviate above the Normal reference.
      </p>
      <ResponsiveContainer width="100%" height={270}>
        <ScatterChart margin={{ top: 4, right: 16, bottom: 24, left: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis
            type="number"
            dataKey="x"
            name="|log return|"
            tick={{ fontSize: 10, fill: "#6b7280" }}
            tickFormatter={(v: number) => `10^${v.toFixed(1)}`}
            label={{ value: "|log return|", position: "insideBottom", offset: -12, fill: "#6b7280", fontSize: 11 }}
          />
          <YAxis
            type="number"
            dataKey="y"
            name="P(|r|>x)"
            tick={{ fontSize: 10, fill: "#6b7280" }}
            tickFormatter={(v: number) => `10^${v.toFixed(0)}`}
            width={56}
            label={{ value: "P(|r|>x)", angle: -90, position: "insideLeft", fill: "#6b7280", fontSize: 11 }}
          />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            {...CHART_STYLE}
            formatter={(v: number) => (Math.pow(10, v)).toExponential(3)}
          />
          <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
          {results.map((r, i) => {
            const color = COLORS[i % COLORS.length];
            const data = r.ccdf_hist.x
              .map((x, j) => {
                const y = r.ccdf_hist.y[j];
                return (y !== null && y > 0 && x > 0) ? { x: Math.log10(x), y: Math.log10(y) } : null;
              })
              .filter((d): d is { x: number; y: number } => d !== null);
            return (
              <Scatter
                key={r.name}
                name={r.name}
                data={data}
                fill={color}
                line={{ stroke: color, strokeWidth: 1.5 }}
                shape={SmallDot}
              />
            );
          })}
        </ScatterChart>
      </ResponsiveContainer>
    </Section>
  );
}

// ── Autocorrelation chart ─────────────────────────────────────────────────────

function AcfChart({ results }: { results: ValidationResult[] }) {
  const series = results.flatMap((r, i) => [
    {
      key: `${r.name}_r`,
      name: `${r.name} (r)`,
      data: r.acf_returns.map((v, lag) => ({ lag, acf: v })),
      color: COLORS[i % COLORS.length],
      dash: "5 3",
    },
    {
      key: `${r.name}_abs`,
      name: `${r.name} (|r|)`,
      data: r.acf_abs_returns.map((v, lag) => ({ lag, acf: v })),
      color: COLORS[i % COLORS.length],
      dash: undefined,
    },
  ]);

  return (
    <Section title="Autocorrelation — returns r (dashed) and |r| (solid)">
      <p className="text-xs text-gray-600">
        Significant positive ACF of |r| at many lags signals volatility clustering.
      </p>
      <ResponsiveContainer width="100%" height={270}>
        <ScatterChart margin={{ top: 4, right: 16, bottom: 24, left: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis
            type="number"
            dataKey="lag"
            name="Lag"
            tick={{ fontSize: 10, fill: "#6b7280" }}
            label={{ value: "Lag", position: "insideBottom", offset: -12, fill: "#6b7280", fontSize: 11 }}
          />
          <YAxis
            type="number"
            dataKey="acf"
            name="ACF"
            tick={{ fontSize: 10, fill: "#6b7280" }}
            width={44}
          />
          <ReferenceLine y={0} stroke="#374151" strokeDasharray="2 2" />
          <Tooltip {...CHART_STYLE} formatter={(v: number) => v.toFixed(4)} />
          <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
          {series.map((s) => (
            <Scatter
              key={s.key}
              name={s.name}
              data={s.data}
              fill={s.color}
              line={{
                stroke: s.color,
                strokeWidth: 1.5,
                ...(s.dash ? { strokeDasharray: s.dash } : {}),
              }}
              shape={NoShape}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </Section>
  );
}

// ── Diffusion chart (log-log) ─────────────────────────────────────────────────

function DiffusionChart({ results }: { results: ValidationResult[] }) {
  return (
    <Section title="Diffusion Scaling — Var(lag) / Var(1) (log-log)">
      <p className="text-xs text-gray-600">
        Slope ≈ 1 indicates random-walk diffusion; slope &gt; 1 suggests super-diffusion.
      </p>
      <ResponsiveContainer width="100%" height={270}>
        <ScatterChart margin={{ top: 4, right: 16, bottom: 24, left: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis
            type="number"
            dataKey="lag"
            name="Lag"
            tick={{ fontSize: 10, fill: "#6b7280" }}
            tickFormatter={(v: number) => `10^${v.toFixed(1)}`}
            label={{ value: "Lag", position: "insideBottom", offset: -12, fill: "#6b7280", fontSize: 11 }}
          />
          <YAxis
            type="number"
            dataKey="v"
            name="Var ratio"
            tick={{ fontSize: 10, fill: "#6b7280" }}
            width={52}
            tickFormatter={(v: number) => `10^${v.toFixed(1)}`}
          />
          <Tooltip
            {...CHART_STYLE}
            formatter={(v: number) => (Math.pow(10, v)).toFixed(4)}
          />
          <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
          {results.map((r, i) => {
            const data = r.diffusion.lags
              .map((lag, j) => {
                const v = r.diffusion.vars[j];
                return (lag > 0 && v > 0) ? { lag: Math.log10(lag), v: Math.log10(v) } : null;
              })
              .filter((d): d is { lag: number; v: number } => d !== null);
            return (
              <Scatter
                key={r.name}
                name={r.name}
                data={data}
                fill={COLORS[i % COLORS.length]}
                line={{ stroke: COLORS[i % COLORS.length], strokeWidth: 1.5 }}
                shape={NoShape}
              />
            );
          })}
        </ScatterChart>
      </ResponsiveContainer>
    </Section>
  );
}

// ── Volatility clustering chart ───────────────────────────────────────────────

function VolClusteringChart({ results }: { results: ValidationResult[] }) {
  return (
    <Section title="Volatility Clustering — ACF of |r| (lags 0–100)">
      <p className="text-xs text-gray-600">
        Real markets show slowly-decaying positive autocorrelation in absolute returns.
      </p>
      <ResponsiveContainer width="100%" height={270}>
        <ScatterChart margin={{ top: 4, right: 16, bottom: 24, left: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis
            type="number"
            dataKey="lag"
            name="Lag"
            tick={{ fontSize: 10, fill: "#6b7280" }}
            label={{ value: "Lag", position: "insideBottom", offset: -12, fill: "#6b7280", fontSize: 11 }}
          />
          <YAxis
            type="number"
            dataKey="acf"
            name="ACF"
            tick={{ fontSize: 10, fill: "#6b7280" }}
            width={44}
          />
          <ReferenceLine y={0} stroke="#374151" strokeDasharray="2 2" />
          <Tooltip {...CHART_STYLE} formatter={(v: number) => v.toFixed(4)} />
          <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
          {results.map((r, i) => (
            <Scatter
              key={r.name}
              name={r.name}
              data={r.volatility_clustering.map((v, lag) => ({ lag, acf: v }))}
              fill={COLORS[i % COLORS.length]}
              line={{ stroke: COLORS[i % COLORS.length], strokeWidth: 1.5 }}
              shape={NoShape}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </Section>
  );
}

// ── QQ plots ──────────────────────────────────────────────────────────────────

function QqPlots({ results }: { results: ValidationResult[] }) {
  return (
    <Section title="QQ Plots vs Normal Distribution">
      <p className="text-xs text-gray-600">
        Points along the diagonal indicate normality; heavy tails bow outward.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
        {results.map((r, i) => {
          const color = COLORS[i % COLORS.length];
          const tVals = r.qq.map((p) => p.t);
          const mn = Math.min(...tVals);
          const mx = Math.max(...tVals);
          const { slope, intercept } = r.qq_line;
          const fitLine = [
            { t: mn, s: slope * mn + intercept },
            { t: mx, s: slope * mx + intercept },
          ];
          return (
            <div key={r.name}>
              <p className="text-xs font-medium mb-1" style={{ color }}>
                {r.name}
              </p>
              <ResponsiveContainer width="100%" height={230}>
                <ScatterChart margin={{ top: 4, right: 16, bottom: 24, left: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis
                    type="number"
                    dataKey="t"
                    name="Theoretical"
                    tick={{ fontSize: 10, fill: "#6b7280" }}
                    label={{
                      value: "Theoretical quantile",
                      position: "insideBottom",
                      offset: -12,
                      fill: "#6b7280",
                      fontSize: 11,
                    }}
                  />
                  <YAxis
                    type="number"
                    dataKey="s"
                    name="Sample"
                    tick={{ fontSize: 10, fill: "#6b7280" }}
                    width={56}
                    tickFormatter={(v: number) => v.toExponential(1)}
                    label={{
                      value: "Sample quantile",
                      angle: -90,
                      position: "insideLeft",
                      fill: "#6b7280",
                      fontSize: 11,
                    }}
                  />
                  <Tooltip {...CHART_STYLE} formatter={(v: number) => v.toExponential(4)} />
                  {/* Fitted reference line: s = slope·t + intercept */}
                  <Scatter
                    name="Normal fit"
                    data={fitLine}
                    fill="#4b5563"
                    line={{ stroke: "#4b5563", strokeDasharray: "4 2" }}
                    shape={NoShape}
                    legendType="none"
                  />
                  <Scatter
                    name={r.name}
                    data={r.qq}
                    fill={color}
                    line={{ stroke: color, strokeWidth: 1.5 }}
                    shape={NoShape}
                  />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          );
        })}
      </div>
    </Section>
  );
}

// Day-of-week colors (fixed palette so Mon/Tue/… are always the same color across datasources)
const DAY_COLORS = ["#60a5fa", "#34d399", "#f59e0b", "#f87171", "#a78bfa", "#fb923c", "#22d3ee"];

// ── Intraday seasonality ───────────────────────────────────────────────────

function IntradaySeasonalityChart({ results }: { results: ValidationResult[] }) {
  const available = results.filter((r) => r.exogenous?.intraday_seasonality);
  if (available.length === 0) return null;
  return (
    <Section title="Intraday Seasonality — mean & std of returns by hour">
      <p className="text-xs text-gray-600">Average log-return and its std per hour of day across all data.</p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
        {(["mean", "std"] as const).map((field) => (
          <div key={field}>
            <p className="text-xs text-gray-500 mb-1 capitalize">{field}</p>
            <ResponsiveContainer width="100%" height={200}>
              <ScatterChart margin={{ top: 4, right: 8, bottom: 20, left: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis type="number" dataKey="h" name="Hour" domain={[0, 23]}
                  ticks={[0, 6, 12, 18, 23]} tick={{ fontSize: 10, fill: "#6b7280" }}
                  label={{ value: "Hour (UTC)", position: "insideBottom", offset: -12, fill: "#6b7280", fontSize: 11 }} />
                <YAxis type="number" dataKey="v" tick={{ fontSize: 10, fill: "#6b7280" }} width={52}
                  tickFormatter={(v: number) => v.toExponential(1)} />
                <ReferenceLine y={0} stroke="#374151" strokeDasharray="2 2" />
                <Tooltip {...CHART_STYLE} formatter={(v: number) => v.toExponential(4)} />
                <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
                {available.map((r, i) => {
                  const s = r.exogenous!.intraday_seasonality!;
                  const data = s.hours.map((h, j) => ({ h, v: s[field][j] }));
                  return (
                    <Scatter key={r.name} name={r.name} data={data}
                      fill={COLORS[i % COLORS.length]}
                      line={{ stroke: COLORS[i % COLORS.length], strokeWidth: 1.5 }}
                      shape={NoShape} />
                  );
                })}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        ))}
      </div>
    </Section>
  );
}

// ── Jump / Tail statistics ─────────────────────────────────────────────────

function JumpTailTable({ results }: { results: ValidationResult[] }) {
  const available = results.filter((r) => r.exogenous?.jump_tail);
  if (available.length === 0) return null;
  return (
    <Section title="Jump / Tail Statistics">
      <p className="text-xs text-gray-600">
        Jump rate = fraction of returns exceeding ±3σ. Quantiles show tail heaviness.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="border-b border-gray-800">
              {["Datasource", "Jump rate (>|3σ|)", "3σ threshold", "q0.1%", "q1%", "q99%", "q99.9%"].map((h, i) => (
                <th key={h} className={`py-2 text-gray-500 font-medium ${i === 0 ? "text-left pr-4" : "text-right px-3"}`}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {available.map((r, i) => {
              const jt = r.exogenous!.jump_tail!;
              return (
                <tr key={r.name} className="border-b border-gray-900 last:border-0">
                  <td className="py-2 pr-4 font-semibold" style={{ color: COLORS[i % COLORS.length] }}>{r.name}</td>
                  <td className="text-right py-2 px-3 text-gray-300 font-mono">{(jt.jump_rate * 100).toFixed(3)}%</td>
                  <td className="text-right py-2 px-3 text-gray-300 font-mono">{jt.threshold_3sigma.toExponential(3)}</td>
                  <td className="text-right py-2 px-3 text-gray-300 font-mono">{jt.q001.toExponential(3)}</td>
                  <td className="text-right py-2 px-3 text-gray-300 font-mono">{jt.q01.toExponential(3)}</td>
                  <td className="text-right py-2 px-3 text-gray-300 font-mono">{jt.q99.toExponential(3)}</td>
                  <td className="text-right py-2 px-3 text-gray-300 font-mono">{jt.q999.toExponential(3)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Section>
  );
}

// ── CDF ───────────────────────────────────────────────────────────────────

function CdfChart({ results }: { results: ValidationResult[] }) {
  const available = results.filter((r) => r.exogenous?.cdf);
  if (available.length === 0) return null;
  return (
    <Section title="CDF — Cumulative Distribution of Returns">
      <p className="text-xs text-gray-600">Empirical CDF of log-returns. S-curve shape; fat tails bow wider than Normal.</p>
      <ResponsiveContainer width="100%" height={270}>
        <ScatterChart margin={{ top: 4, right: 16, bottom: 24, left: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis type="number" dataKey="x" name="log return" tick={{ fontSize: 10, fill: "#6b7280" }}
            label={{ value: "log return", position: "insideBottom", offset: -12, fill: "#6b7280", fontSize: 11 }} />
          <YAxis type="number" dataKey="y" name="CDF" domain={[0, 1]}
            tick={{ fontSize: 10, fill: "#6b7280" }} width={44}
            label={{ value: "CDF", angle: -90, position: "insideLeft", fill: "#6b7280", fontSize: 11 }} />
          <Tooltip {...CHART_STYLE} formatter={(v: number) => v.toFixed(4)} />
          <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
          {available.map((r, i) => {
            const c = r.exogenous!.cdf!;
            const data = c.x.map((x, j) => ({ x, y: c.y[j] }));
            return (
              <Scatter key={r.name} name={r.name} data={data}
                fill={COLORS[i % COLORS.length]}
                line={{ stroke: COLORS[i % COLORS.length], strokeWidth: 1.5 }}
                shape={NoShape} />
            );
          })}
        </ScatterChart>
      </ResponsiveContainer>
    </Section>
  );
}

// ── Drift / Rolling Mean ───────────────────────────────────────────────────

function RollingMeanChart({ results }: { results: ValidationResult[] }) {
  const available = results.filter((r) => r.exogenous?.rolling_mean);
  if (available.length === 0) return null;
  return (
    <Section title="Drift / Rolling Mean of Returns">
      <p className="text-xs text-gray-600">
        Rolling mean of log-returns (window ≈ 5% of series length). Should hover near zero for a drift-free market.
      </p>
      <ResponsiveContainer width="100%" height={270}>
        <ScatterChart margin={{ top: 4, right: 16, bottom: 24, left: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis type="number" dataKey="x" name="Position"
            tick={{ fontSize: 10, fill: "#6b7280" }}
            label={{ value: "Position (candle)", position: "insideBottom", offset: -12, fill: "#6b7280", fontSize: 11 }} />
          <YAxis type="number" dataKey="y" name="Rolling mean"
            tick={{ fontSize: 10, fill: "#6b7280" }} width={56}
            tickFormatter={(v: number) => v.toExponential(1)} />
          <ReferenceLine y={0} stroke="#374151" strokeDasharray="2 2" />
          <Tooltip {...CHART_STYLE} formatter={(v: number) => v.toExponential(4)} />
          <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
          {available.map((r, i) => {
            const rm = r.exogenous!.rolling_mean!;
            const data = rm.index.map((x, j) => ({ x, y: rm.values[j] }));
            return (
              <Scatter key={r.name} name={`${r.name} (w=${rm.window})`} data={data}
                fill={COLORS[i % COLORS.length]}
                line={{ stroke: COLORS[i % COLORS.length], strokeWidth: 1.2 }}
                shape={NoShape} />
            );
          })}
        </ScatterChart>
      </ResponsiveContainer>
    </Section>
  );
}

// ── Long-lag ACF ───────────────────────────────────────────────────────────

function LongLagAcfChart({ results }: { results: ValidationResult[] }) {
  const available = results.filter((r) => r.exogenous?.long_lag_acf);
  if (available.length === 0) return null;
  return (
    <Section title="Long-lag ACF of Returns">
      <p className="text-xs text-gray-600">
        ACF of log-returns up to lag 200. Values near zero indicate no long-range linear autocorrelation.
      </p>
      <div className="overflow-x-auto mb-3">
        <table className="text-xs border-collapse">
          <thead>
            <tr className="border-b border-gray-800">
              <th className="py-1 pr-4 text-left text-gray-500">Datasource</th>
              {[10, 20, 30, 40, 50].map((lg) => (
                <th key={lg} className="py-1 px-3 text-right text-gray-500">lag {lg}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {available.map((r, i) => {
              const hl = r.exogenous!.long_lag_acf!.highlights;
              return (
                <tr key={r.name} className="border-b border-gray-900 last:border-0">
                  <td className="py-1 pr-4 font-semibold" style={{ color: COLORS[i % COLORS.length] }}>{r.name}</td>
                  {[10, 20, 30, 40, 50].map((lg) => (
                    <td key={lg} className="py-1 px-3 text-right text-gray-300 font-mono">
                      {hl[String(lg)] != null ? hl[String(lg)].toFixed(4) : "—"}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <ScatterChart margin={{ top: 4, right: 16, bottom: 24, left: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis type="number" dataKey="lag" name="Lag" tick={{ fontSize: 10, fill: "#6b7280" }}
            label={{ value: "Lag", position: "insideBottom", offset: -12, fill: "#6b7280", fontSize: 11 }} />
          <YAxis type="number" dataKey="acf" name="ACF" tick={{ fontSize: 10, fill: "#6b7280" }} width={44} />
          <ReferenceLine y={0} stroke="#374151" strokeDasharray="2 2" />
          <Tooltip {...CHART_STYLE} formatter={(v: number) => v.toFixed(4)} />
          <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
          {available.map((r, i) => {
            const la = r.exogenous!.long_lag_acf!;
            const data = la.lags.map((lag, j) => ({ lag, acf: la.values[j] }));
            return (
              <Scatter key={r.name} name={r.name} data={data}
                fill={COLORS[i % COLORS.length]}
                line={{ stroke: COLORS[i % COLORS.length], strokeWidth: 1.2 }}
                shape={NoShape} />
            );
          })}
        </ScatterChart>
      </ResponsiveContainer>
    </Section>
  );
}

// ── Seasonality (weekly / monthly / yearly) ────────────────────────────────

type SeasonalityTab = "weekly" | "monthly" | "yearly";

function SeasonalityWeeklyView({ ws }: { ws: SeasonalityWeekly }) {
  return (
    <ResponsiveContainer width="100%" height={250}>
      <ScatterChart margin={{ top: 4, right: 16, bottom: 28, left: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
        <XAxis type="number" dataKey="x" name="Time slot"
          domain={[0, ws.counts_per_day * 7 - 1]}
          ticks={ws.day_boundaries.slice(0, 7)}
          tickFormatter={(v: number) => { const i = ws.day_boundaries.indexOf(v); return i >= 0 ? ws.days[i]?.label ?? "" : ""; }}
          tick={{ fontSize: 10, fill: "#6b7280" }}
          label={{ value: "Day of week (time →)", position: "insideBottom", offset: -16, fill: "#6b7280", fontSize: 11 }} />
        <YAxis type="number" dataKey="y" name="volume" domain={["auto", "auto"]}
          tick={{ fontSize: 10, fill: "#6b7280" }} width={52}
          label={{ value: "volume", angle: -90, position: "insideLeft", fill: "#6b7280", fontSize: 11 }} />
        <Tooltip {...CHART_STYLE} formatter={(v: number) => v.toFixed(2)}
          labelFormatter={(slot: number) => {
            const di = Math.floor(slot / ws.counts_per_day);
            return `${ws.days[di]?.label ?? ""} ${ws.time_labels[slot % ws.counts_per_day] ?? ""}`;
          }} />
        <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
        {ws.day_boundaries.slice(1, 7).map((b) => (
          <ReferenceLine key={b} x={b} stroke="#374151" strokeDasharray="2 2" />
        ))}
        {ws.days.map((day, d) => {
          const data = day.slots.map((x, j) => { const y = day.volume[j]; return y !== null ? { x, y } : null; })
            .filter((pt): pt is { x: number; y: number } => pt !== null);
          return (
            <Scatter key={day.label} name={day.label} data={data}
              fill={DAY_COLORS[d % DAY_COLORS.length]}
              line={{ stroke: DAY_COLORS[d % DAY_COLORS.length], strokeWidth: 1.5 }} shape={NoShape} />
          );
        })}
      </ScatterChart>
    </ResponsiveContainer>
  );
}

function SeasonalityMonthlyView({ monthly }: { monthly: SeasonalityMonthly }) {
  return (
    <ResponsiveContainer width="100%" height={250}>
      <ScatterChart margin={{ top: 4, right: 16, bottom: 28, left: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
        <XAxis type="number" dataKey="x" name="Day of week" domain={[0, 6]}
          ticks={[0, 1, 2, 3, 4, 5, 6]}
          tickFormatter={(v: number) => monthly.day_labels[v] ?? ""}
          tick={{ fontSize: 10, fill: "#6b7280" }}
          label={{ value: "Day of week", position: "insideBottom", offset: -16, fill: "#6b7280", fontSize: 11 }} />
        <YAxis type="number" dataKey="y" name="volume" domain={["auto", "auto"]}
          tick={{ fontSize: 10, fill: "#6b7280" }} width={52}
          label={{ value: "volume", angle: -90, position: "insideLeft", fill: "#6b7280", fontSize: 11 }} />
        <Tooltip {...CHART_STYLE} formatter={(v: number) => v.toFixed(2)}
          labelFormatter={(v: number) => monthly.day_labels[v] ?? ""} />
        <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
        {monthly.weeks.map((wk, i) => {
          const data = wk.days.map((x, j) => { const y = wk.volume[j]; return y !== null ? { x, y } : null; })
            .filter((pt): pt is { x: number; y: number } => pt !== null);
          return (
            <Scatter key={wk.label} name={wk.label} data={data}
              fill={DAY_COLORS[i % DAY_COLORS.length]}
              line={{ stroke: DAY_COLORS[i % DAY_COLORS.length], strokeWidth: 1.5 }} shape={NoShape} />
          );
        })}
      </ScatterChart>
    </ResponsiveContainer>
  );
}

function SeasonalityYearlyView({ yearly }: { yearly: SeasonalityYearly }) {
  return (
    <ResponsiveContainer width="100%" height={250}>
      <ScatterChart margin={{ top: 4, right: 16, bottom: 28, left: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
        <XAxis type="number" dataKey="x" name="Month" domain={[1, 12]}
          ticks={[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}
          tickFormatter={(v: number) => yearly.month_labels[v - 1] ?? ""}
          tick={{ fontSize: 10, fill: "#6b7280" }}
          label={{ value: "Month", position: "insideBottom", offset: -16, fill: "#6b7280", fontSize: 11 }} />
        <YAxis type="number" dataKey="y" name="volume" domain={["auto", "auto"]}
          tick={{ fontSize: 10, fill: "#6b7280" }} width={52}
          label={{ value: "volume", angle: -90, position: "insideLeft", fill: "#6b7280", fontSize: 11 }} />
        <Tooltip {...CHART_STYLE} formatter={(v: number) => v.toFixed(2)}
          labelFormatter={(v: number) => yearly.month_labels[v - 1] ?? ""} />
        <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
        {yearly.series.map((s, i) => {
          const data = s.months.map((x, j) => { const y = s.volume[j]; return y !== null ? { x, y } : null; })
            .filter((pt): pt is { x: number; y: number } => pt !== null);
          return (
            <Scatter key={s.label} name={s.label} data={data}
              fill={COLORS[i % COLORS.length]}
              line={{ stroke: COLORS[i % COLORS.length], strokeWidth: 1.5 }} shape={NoShape} />
          );
        })}
      </ScatterChart>
    </ResponsiveContainer>
  );
}

function SeasonalityPanel({ result, colorIdx }: { result: ValidationResult; colorIdx: number }) {
  const [tab, setTab] = useState<SeasonalityTab>("weekly");
  const s = result.exogenous!.seasonality!;
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs font-semibold" style={{ color: COLORS[colorIdx % COLORS.length] }}>
          {result.name}
          <span className="text-gray-500 font-normal ml-2">({s.weekly.time_unit_min}-min bars)</span>
        </p>
        <div className="flex gap-1">
          {(["weekly", "monthly", "yearly"] as SeasonalityTab[]).map((t) => (
            <button key={t} onClick={() => setTab(t)}
              className={`text-xs px-2.5 py-1 rounded transition-colors capitalize ${
                tab === t ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-400 hover:text-white"
              }`}>
              {t}
            </button>
          ))}
        </div>
      </div>
      {tab === "weekly"  && <SeasonalityWeeklyView ws={s.weekly} />}
      {tab === "monthly" && <SeasonalityMonthlyView monthly={s.monthly} />}
      {tab === "yearly"  && <SeasonalityYearlyView yearly={s.yearly} />}
    </div>
  );
}

function SeasonalityChart({ results }: { results: ValidationResult[] }) {
  const available = results.filter((r) => r.exogenous?.seasonality);
  const unavailable = results.filter((r) => r.exogenous && !r.exogenous.seasonality);
  if (available.length === 0 && unavailable.length === 0) return null;
  return (
    <Section title="Volume Seasonality">
      <p className="text-xs text-gray-600">
        Switch between weekly (intraday by day), monthly (by week-of-month), and yearly (by month) views per datasource.
      </p>
      {unavailable.length > 0 && (
        <p className="text-xs text-amber-500/80 bg-amber-500/10 border border-amber-500/20 rounded px-3 py-2">
          No volume data for: {unavailable.map((r) => r.name).join(", ")}. CSV datasources need a &ldquo;volume&rdquo; or &ldquo;tick_volume&rdquo; column; simulator exports include tick_volume automatically.
        </p>
      )}
      <div className="space-y-6 pt-1">
        {available.map((r, i) => <SeasonalityPanel key={r.name} result={r} colorIdx={i} />)}
      </div>
    </Section>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

type TabId = "endogenous" | "exogenous";

export default function ValidationPage() {
  const [datasources, setDatasources] = useState<DsItem[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [results, setResults] = useState<ValidationResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState<TabId>("endogenous");

  useEffect(() => {
    fetch("/api/datasources", { cache: "no-store" })
      .then((r) => r.json())
      .then((ds: DsItem[]) => setDatasources(ds))
      .catch(() => {});
  }, []);

  function toggle(name: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  async function handleCompare() {
    if (selected.size === 0) return;
    setLoading(true);
    setError("");
    setResults(null);
    try {
      const names = Array.from(selected).join(",");
      const res = await fetch(`/api/validation?names=${encodeURIComponent(names)}`);
      const data = await res.json();
      if (!res.ok) {
        setError(data.error ?? "Comparison failed");
        return;
      }
      setResults(data);
    } catch {
      setError("Network error — is the dev server running?");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      {/* Page header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold">Simulation Validation</h1>
        <p className="text-gray-500 text-sm mt-1">
          Compare stylised statistical facts between market data and simulated price series.
        </p>
      </div>

      {/* Datasource selector */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-5 mb-6">
        <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Select CSV Datasources to Compare
        </h2>

        {datasources.length === 0 ? (
          <p className="text-sm text-gray-600 mb-4">
            No datasources registered yet. Go to{" "}
            <a href="/datasources" className="text-blue-400 hover:text-blue-300">
              Data Sources
            </a>{" "}
            to register one.
          </p>
        ) : (
          <>
            <div className="flex flex-wrap gap-2 mb-4">
              {datasources.map((ds) => {
                const checked = selected.has(ds.name);
                const color = COLORS[Array.from(selected).indexOf(ds.name) % COLORS.length];
                return (
                  <label
                    key={ds.name}
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg border cursor-pointer text-sm transition-colors select-none ${
                      checked
                        ? "border-blue-500/60 bg-blue-500/10 text-white"
                        : "border-gray-700 bg-gray-800 text-gray-400 hover:border-gray-600 hover:text-gray-300"
                    }`}
                  >
                    <input
                      type="checkbox"
                      className="accent-blue-500"
                      checked={checked}
                      onChange={() => toggle(ds.name)}
                    />
                    <span className="font-medium" style={checked ? { color } : undefined}>
                      {ds.name}
                    </span>
                    {ds.type === "csv" ? (
                      <span className="text-xs bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 px-1.5 py-0.5 rounded-full">
                        CSV
                      </span>
                    ) : (
                      <span className="text-xs bg-purple-500/20 text-purple-400 border border-purple-500/30 px-1.5 py-0.5 rounded-full">
                        Sim
                      </span>
                    )}
                    {ds.description && (
                      <span className="text-xs text-gray-500 hidden sm:inline truncate max-w-[160px]">
                        {ds.description}
                      </span>
                    )}
                  </label>
                );
              })}
            </div>
            <p className="text-xs text-gray-600 mb-3">
              Simulator datasources use the current simulation output (running, completed, or stopped).
            </p>
          </>
        )}

        <div className="flex items-center gap-3">
          <button
            onClick={handleCompare}
            disabled={selected.size === 0 || loading}
            className="px-4 py-2 text-sm bg-blue-600 hover:bg-blue-500 disabled:opacity-50 rounded-md font-medium transition-colors"
          >
            {loading
              ? "Computing…"
              : selected.size > 0
              ? `Compare (${selected.size})`
              : "Compare"}
          </button>
          {selected.size > 0 && !loading && (
            <button
              onClick={() => setSelected(new Set())}
              className="text-xs text-gray-600 hover:text-gray-400 transition-colors"
            >
              Clear selection
            </button>
          )}
        </div>

        {error && (
          <p className="mt-3 text-sm text-red-400 bg-red-400/10 border border-red-400/20 rounded px-3 py-2">
            {error}
          </p>
        )}
      </div>

      {/* Results */}
      {results && (
        <div className="space-y-4">
          {/* Tabs */}
          <div className="flex gap-1 border-b border-gray-800">
            {(["endogenous", "exogenous"] as TabId[]).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 text-sm font-medium capitalize transition-colors border-b-2 -mb-px ${
                  activeTab === tab
                    ? "border-blue-500 text-white"
                    : "border-transparent text-gray-500 hover:text-gray-300"
                }`}
              >
                {tab}
              </button>
            ))}
          </div>

          {activeTab === "endogenous" && (
            <>
              <StatsSection results={results} />
              <CcdfChart results={results} />
              <AcfChart results={results} />
              <DiffusionChart results={results} />
              <VolClusteringChart results={results} />
              <QqPlots results={results} />
            </>
          )}

          {activeTab === "exogenous" && (
            <>
              <JumpTailTable results={results} />
              <CdfChart results={results} />
              <IntradaySeasonalityChart results={results} />
              <RollingMeanChart results={results} />
              <LongLagAcfChart results={results} />
              <SeasonalityChart results={results} />
            </>
          )}
        </div>
      )}
    </div>
  );
}
