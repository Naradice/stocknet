import { NextResponse } from "next/server";
import { listConfigNames, getConfig, saveConfig } from "@/lib/configs";
import { isRunning } from "@/lib/processes";

export const dynamic = "force-dynamic";

export async function GET() {
  const names = listConfigNames();
  const configs = names.map((name) => ({
    name,
    config: getConfig(name),
    running: isRunning(name),
  }));
  return NextResponse.json({ configs });
}

export async function POST(req: Request) {
  const body = await req.json();
  const { name, config } = body as { name: string; config: Record<string, unknown> };

  if (!name || typeof name !== "string") {
    return NextResponse.json({ error: "name is required" }, { status: 400 });
  }
  if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
    return NextResponse.json(
      { error: "name must contain only letters, numbers, underscores, hyphens" },
      { status: 400 }
    );
  }

  saveConfig(name, config);
  return NextResponse.json({ ok: true, name });
}
