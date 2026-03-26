import { NextResponse } from "next/server";
import {
  startSimulation,
  stopSimulation,
  checkpointSimulation,
  exportSimulationToCsv,
  isSimulating,
  readStatus,
  readSimLog,
  defaultSimulationId,
} from "@/lib/simulations";
import { getDatasource } from "@/lib/datasources";
import type { SimulatorDatasource } from "@/lib/datasourceTypes";

export const dynamic = "force-dynamic";

export async function GET(
  _req: Request,
  { params }: { params: { name: string } }
) {
  const { name } = params;
  const running = isSimulating(name);
  const status = readStatus(name);
  const log = readSimLog(name);

  const phase = running ? "running" : (status?.phase ?? null);

  return NextResponse.json({ running, phase, status, log });
}

export async function POST(
  req: Request,
  { params }: { params: { name: string } }
) {
  const { name } = params;
  const body = await req.json() as {
    action: "start" | "stop" | "checkpoint" | "export";
    simulationId?: string;
    length?: number | null;
  };

  if (body.action === "start") {
    if (isSimulating(name)) {
      return NextResponse.json({ error: "Already running" }, { status: 409 });
    }
    const simulationId = body.simulationId?.trim() || defaultSimulationId(name);
    const length = body.length ?? null;
    try {
      const { pid } = startSimulation(name, simulationId, length);
      return NextResponse.json({ ok: true, pid, simulationId });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      return NextResponse.json({ error: msg }, { status: 500 });
    }
  }

  if (body.action === "stop") {
    if (!isSimulating(name)) {
      return NextResponse.json({ error: "Not running" }, { status: 409 });
    }
    try {
      stopSimulation(name);
      return NextResponse.json({ ok: true });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      return NextResponse.json({ error: msg }, { status: 500 });
    }
  }

  if (body.action === "checkpoint") {
    try {
      const { checkpointId } = await checkpointSimulation(name);
      return NextResponse.json({ ok: true, checkpointId });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      return NextResponse.json({ error: msg }, { status: 500 });
    }
  }

  if (body.action === "export") {
    const status = readStatus(name);
    if (!status?.simulationId) {
      return NextResponse.json(
        { error: "No simulation data found. Run a simulation first." },
        { status: 400 }
      );
    }
    try {
      const ds = getDatasource(name);
      const samplerRule =
        ds?.type === "simulator" ? (ds as SimulatorDatasource).samplerRule : "min";
      const filePath = await exportSimulationToCsv(name, status.simulationId, samplerRule);
      return NextResponse.json({ ok: true, filePath });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      return NextResponse.json({ error: msg }, { status: 500 });
    }
  }

  return NextResponse.json({ error: "Invalid action" }, { status: 400 });
}
