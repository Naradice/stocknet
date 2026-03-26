import { NextResponse } from "next/server";
import fs from "fs";
import readline from "readline";
import { getDatasource } from "@/lib/datasources";
import { readStatus } from "@/lib/simulations";
import pool from "@/lib/db";
import type { SimulatorDatasource } from "@/lib/datasourceTypes";

export const dynamic = "force-dynamic";

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

interface PreviewResponse {
  available: boolean;
  reason?: string;
  summary?: PreviewSummary;
  chart?: ChartRow[];
}

const CHART_SAMPLES = 500;

const RULE_TO_INTERVAL: Record<string, string> = {
  min: "1 minute",   T: "1 minute",
  "5min": "5 minutes",   "5T": "5 minutes",
  "15min": "15 minutes", "15T": "15 minutes",
  "30min": "30 minutes", "30T": "30 minutes",
  H: "1 hour",
  D: "1 day",
};

// ── CSV preview (for CSV datasources) ────────────────────────────────────────

async function buildPreviewFromCsv(filePath: string): Promise<PreviewResponse> {
  if (!fs.existsSync(filePath)) {
    return { available: false, reason: "File not found" };
  }

  return new Promise((resolve) => {
    let fileStream: fs.ReadStream;
    try {
      fileStream = fs.createReadStream(filePath, { encoding: "utf-8" });
    } catch {
      resolve({ available: false, reason: "Could not read file" });
      return;
    }

    const rl = readline.createInterface({ input: fileStream, crlfDelay: Infinity });

    let headers: string[] = [];
    let timeIdx = 0;
    let closeIdx = -1;
    let openIdx = -1;
    let highIdx = -1;
    let lowIdx = -1;

    let rowCount = 0;
    let fromDate = "";
    let toDate = "";
    let closeMin = Infinity;
    let closeMax = -Infinity;
    let closeSum = 0;
    let closeCount = 0;

    const reservoir: { seq: number; cols: string[] }[] = [];

    rl.on("line", (line) => {
      const trimmed = line.trim();
      if (!trimmed) return;

      const cols = trimmed.split(",").map((c) => c.trim());

      if (headers.length === 0) {
        headers = cols;
        closeIdx = headers.findIndex((h) => h.toLowerCase() === "close");
        if (closeIdx < 0) closeIdx = headers.length - 1;
        openIdx = headers.findIndex((h) => h.toLowerCase() === "open");
        highIdx = headers.findIndex((h) => h.toLowerCase() === "high");
        lowIdx = headers.findIndex((h) => h.toLowerCase() === "low");
        return;
      }

      if (cols.length < headers.length) return;

      rowCount++;
      if (rowCount === 1) fromDate = cols[timeIdx];
      toDate = cols[timeIdx];

      const v = parseFloat(cols[closeIdx]);
      if (!isNaN(v)) {
        if (v < closeMin) closeMin = v;
        if (v > closeMax) closeMax = v;
        closeSum += v;
        closeCount++;
      }

      if (reservoir.length < CHART_SAMPLES) {
        reservoir.push({ seq: rowCount, cols });
      } else {
        const j = Math.floor(Math.random() * rowCount);
        if (j < CHART_SAMPLES) {
          reservoir[j] = { seq: rowCount, cols };
        }
      }
    });

    const onDone = () => {
      if (headers.length === 0 || rowCount === 0) {
        resolve({ available: false, reason: "No data rows" });
        return;
      }
      if (closeCount === 0) {
        resolve({ available: false, reason: "No numeric close values" });
        return;
      }

      const summary: PreviewSummary = {
        totalRows: rowCount,
        fromDate,
        toDate,
        columns: headers.slice(1),
        closeMin,
        closeMax,
        closeMean: Math.round((closeSum / closeCount) * 10000) / 10000,
      };

      reservoir.sort((a, b) => a.seq - b.seq);

      const chart: ChartRow[] = reservoir.map(({ cols: r }) => {
        const row: ChartRow = { time: r[timeIdx], close: parseFloat(r[closeIdx]) };
        if (openIdx >= 0) row.open = parseFloat(r[openIdx]);
        if (highIdx >= 0) row.high = parseFloat(r[highIdx]);
        if (lowIdx >= 0) row.low = parseFloat(r[lowIdx]);
        return row;
      });

      resolve({ available: true, summary, chart });
    };

    rl.on("close", onDone);
    rl.on("error", () => resolve({ available: false, reason: "Could not read file" }));
    fileStream.on("error", () => resolve({ available: false, reason: "Could not read file" }));
  });
}

// ── PostgreSQL preview (for simulator datasources) ────────────────────────────

async function buildPreviewFromDb(
  datasource: string,
  simulationId: string,
  samplerRule: string,
): Promise<PreviewResponse> {
  const interval = RULE_TO_INTERVAL[samplerRule] ?? "1 minute";

  try {
    // Summary stats from raw ticks
    const statsRes = await pool.query<{
      total_ticks: string;
      from_ts: Date;
      to_ts: Date;
      price_min: string;
      price_max: string;
      price_mean: string;
    }>(
      `SELECT
         count(*)   AS total_ticks,
         min(ts)    AS from_ts,
         max(ts)    AS to_ts,
         min(price) AS price_min,
         max(price) AS price_max,
         avg(price) AS price_mean
       FROM simulation_ticks
       WHERE datasource = $1 AND simulation_id = $2`,
      [datasource, simulationId]
    );

    const stats = statsRes.rows[0];
    if (!stats || parseInt(stats.total_ticks) === 0) {
      return { available: false, reason: "No tick data found in database" };
    }

    // OHLC aggregation via date_bin (requires PostgreSQL 14+)
    const ohlcRes = await pool.query<{
      period: Date;
      open: string;
      high: string;
      low: string;
      close: string;
    }>(
      `SELECT
         date_bin($3::interval, ts, '2020-01-01'::timestamptz) AS period,
         (array_agg(price ORDER BY ts))[1]       AS open,
         max(price)                               AS high,
         min(price)                               AS low,
         (array_agg(price ORDER BY ts DESC))[1]  AS close
       FROM simulation_ticks
       WHERE datasource = $1 AND simulation_id = $2
       GROUP BY date_bin($3::interval, ts, '2020-01-01'::timestamptz)
       ORDER BY period`,
      [datasource, simulationId, interval]
    );

    const rows = ohlcRes.rows;
    const totalCandles = rows.length;

    if (totalCandles === 0) {
      return { available: false, reason: "No candles after aggregation" };
    }

    // Reservoir-sample down to CHART_SAMPLES for the chart
    const reservoir: typeof rows = [];
    for (let i = 0; i < rows.length; i++) {
      if (reservoir.length < CHART_SAMPLES) {
        reservoir.push(rows[i]);
      } else {
        const j = Math.floor(Math.random() * (i + 1));
        if (j < CHART_SAMPLES) reservoir[j] = rows[i];
      }
    }
    reservoir.sort((a, b) => a.period.getTime() - b.period.getTime());

    const summary: PreviewSummary = {
      totalRows: totalCandles,
      fromDate: stats.from_ts.toISOString(),
      toDate: stats.to_ts.toISOString(),
      columns: ["open", "high", "low", "close"],
      closeMin: parseFloat(stats.price_min),
      closeMax: parseFloat(stats.price_max),
      closeMean: Math.round(parseFloat(stats.price_mean) * 10000) / 10000,
    };

    const chart: ChartRow[] = reservoir.map((r) => ({
      time: r.period.toISOString(),
      open: parseFloat(r.open),
      high: parseFloat(r.high),
      low: parseFloat(r.low),
      close: parseFloat(r.close),
    }));

    return { available: true, summary, chart };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return { available: false, reason: `Database error: ${msg}` };
  }
}

// ── Route handler ─────────────────────────────────────────────────────────────

export async function GET(
  _req: Request,
  { params }: { params: { name: string } }
) {
  const { name } = params;

  const ds = getDatasource(name);
  if (!ds) {
    return NextResponse.json<PreviewResponse>(
      { available: false, reason: "Datasource not found" },
      { status: 404 }
    );
  }

  if (ds.type === "csv") {
    const result = await buildPreviewFromCsv(ds.filePath);
    return NextResponse.json<PreviewResponse>(result);
  }

  // Simulator: roll up tick data from PostgreSQL
  const status = readStatus(name);
  if (!status) {
    return NextResponse.json<PreviewResponse>({
      available: false,
      reason: "No simulation has been run yet",
    });
  }

  const samplerRule = (ds as SimulatorDatasource).samplerRule ?? "min";

  if (status.phase === "running") {
    // Show partial data while still running
    const result = await buildPreviewFromDb(name, status.simulationId, samplerRule);
    if (result.available) {
      return NextResponse.json<PreviewResponse>({
        ...result,
        reason: "partial — simulation still running",
      });
    }
    return NextResponse.json<PreviewResponse>({
      available: false,
      reason: "Simulation in progress — no data yet",
    });
  }

  const result = await buildPreviewFromDb(name, status.simulationId, samplerRule);
  return NextResponse.json<PreviewResponse>(result);
}
