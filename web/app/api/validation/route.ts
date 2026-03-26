import { NextResponse } from "next/server";
import { spawnSync } from "child_process";
import fs from "fs";
import path from "path";
import { getDatasource } from "@/lib/datasources";
import type { CsvDatasource, SimulatorDatasource } from "@/lib/datasourceTypes";
import { readStatus, exportSimulationToCsv } from "@/lib/simulations";

export const dynamic = "force-dynamic";

const PROJECT_ROOT = path.resolve(process.cwd(), "..");
const SCRIPT = path.join(PROJECT_ROOT, "scripts", "compute_validation.py");

export async function GET(req: Request) {
  const url = new URL(req.url);
  const names = (url.searchParams.get("names") ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

  if (names.length === 0) {
    return NextResponse.json({ error: "Provide ?names=name1,name2" }, { status: 400 });
  }

  const args: string[] = [];
  const tempFiles: string[] = [];

  try {
    for (const name of names) {
      const ds = getDatasource(name);
      if (!ds) {
        return NextResponse.json({ error: `Datasource not found: ${name}` }, { status: 404 });
      }

      let filePath: string;

      if (ds.type === "csv") {
        filePath = (ds as CsvDatasource).filePath;
        const volCol = (ds as CsvDatasource).volumeColumn;
        args.push(`${filePath}:${name}${volCol ? `?vol=${volCol}` : ""}`);
        continue;
      } else {
        // Simulator: export tick data from PostgreSQL to a temp OHLC CSV
        const status = readStatus(name);
        const usable =
          status?.simulationId &&
          (status.phase === "completed" || status.phase === "stopped" || status.phase === "running");
        if (!usable) {
          return NextResponse.json(
            { error: `"${name}" has no simulation data. Start a simulation first.` },
            { status: 400 }
          );
        }
        try {
          const samplerRule = (ds as SimulatorDatasource).samplerRule ?? "min";
          filePath = await exportSimulationToCsv(
            name,
            status!.simulationId,
            samplerRule,
          );
          tempFiles.push(filePath);
        } catch (err: unknown) {
          const msg = err instanceof Error ? err.message : String(err);
          return NextResponse.json(
            { error: `Failed to export simulation data for "${name}": ${msg}` },
            { status: 500 }
          );
        }
      }

      args.push(`${filePath}:${name}`);
    }

    const result = spawnSync("python", [SCRIPT, ...args], {
      cwd: PROJECT_ROOT,
      encoding: "utf-8",
      timeout: 120_000,
    });

    if (result.error) {
      return NextResponse.json({ error: result.error.message }, { status: 500 });
    }
    if (result.status !== 0) {
      return NextResponse.json(
        { error: result.stderr?.trim() || "Validation script failed" },
        { status: 500 }
      );
    }

    try {
      return NextResponse.json(JSON.parse(result.stdout));
    } catch {
      return NextResponse.json({ error: "Failed to parse script output" }, { status: 500 });
    }
  } finally {
    // Clean up temporary CSV exports
    for (const f of tempFiles) {
      try { fs.unlinkSync(f); } catch { /* ignore */ }
    }
  }
}
