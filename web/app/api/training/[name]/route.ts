import { NextResponse } from "next/server";
import { startTraining, stopTraining, isRunning, getRunningInfo } from "@/lib/processes";
import { readStdoutLog } from "@/lib/configs";

export const dynamic = "force-dynamic";

export async function GET(
  _req: Request,
  { params }: { params: { name: string } }
) {
  const info = getRunningInfo(params.name);
  const log = readStdoutLog(params.name);
  return NextResponse.json({
    running: isRunning(params.name),
    info,
    log,
  });
}

export async function POST(
  req: Request,
  { params }: { params: { name: string } }
) {
  const body = await req.json() as { action: "start" | "stop" };

  if (body.action === "start") {
    if (isRunning(params.name)) {
      return NextResponse.json({ error: "Already running" }, { status: 409 });
    }
    try {
      const { pid } = startTraining(params.name);
      return NextResponse.json({ ok: true, pid });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      return NextResponse.json({ error: msg }, { status: 500 });
    }
  }

  if (body.action === "stop") {
    if (!isRunning(params.name)) {
      return NextResponse.json({ error: "Not running" }, { status: 409 });
    }
    try {
      stopTraining(params.name);
      return NextResponse.json({ ok: true });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      return NextResponse.json({ error: msg }, { status: 500 });
    }
  }

  return NextResponse.json({ error: "Invalid action" }, { status: 400 });
}
