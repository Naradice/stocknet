import { NextResponse } from "next/server";
import { getConfig, saveConfig, deleteConfig } from "@/lib/configs";
import { listModelVersions } from "@/lib/logs";
import { isRunning, stopTraining } from "@/lib/processes";
import fs from "fs";
import path from "path";

export const dynamic = "force-dynamic";

const LOGS_ROOT = path.resolve(process.cwd(), "../logs");

export async function GET(
  _req: Request,
  { params }: { params: { name: string } }
) {
  const config = getConfig(params.name);
  if (!config) {
    return NextResponse.json({ error: "Config not found" }, { status: 404 });
  }
  return NextResponse.json({ name: params.name, config, running: isRunning(params.name) });
}

export async function PUT(
  req: Request,
  { params }: { params: { name: string } }
) {
  if (isRunning(params.name)) {
    return NextResponse.json(
      { error: "Cannot update config while training is running" },
      { status: 409 }
    );
  }
  const body = await req.json();
  saveConfig(params.name, body.config);
  return NextResponse.json({ ok: true });
}

export async function DELETE(
  req: Request,
  { params }: { params: { name: string } }
) {
  // Stop training if running
  if (isRunning(params.name)) {
    try { stopTraining(params.name); } catch { /* ignore */ }
  }

  deleteConfig(params.name);

  // Optionally delete logs
  const url = new URL(req.url);
  if (url.searchParams.get("deleteLogs") === "true") {
    const logDir = path.join(LOGS_ROOT, params.name);
    if (fs.existsSync(logDir)) {
      fs.rmSync(logDir, { recursive: true, force: true });
    }
  }

  // Also delete log versions from other places (versions might differ from model name)
  const versions = listModelVersions(params.name);
  if (url.searchParams.get("deleteLogs") === "true" && versions.length > 0) {
    // Already handled above by deleting the whole dir
  }

  return NextResponse.json({ ok: true });
}
