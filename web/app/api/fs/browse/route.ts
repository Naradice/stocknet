import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import os from "os";

export const dynamic = "force-dynamic";

export interface FsEntry {
  name: string;
  type: "file" | "dir";
  path: string;
}

/** Detect available drives on Windows by probing A-Z, or return ["/"] on Unix. */
function listDrives(): string[] {
  if (process.platform !== "win32") return ["/"];
  const drives: string[] = [];
  for (let i = 65; i <= 90; i++) {
    const drive = String.fromCharCode(i) + ":\\";
    try {
      fs.accessSync(drive);
      drives.push(drive);
    } catch {
      // drive not accessible
    }
  }
  return drives;
}

export async function GET(req: Request) {
  const url = new URL(req.url);
  const defaultDir = path.resolve(process.cwd(), "..");
  const rawPath = url.searchParams.get("path") || defaultDir;

  // Normalize — support bare drive letters like "Z:" → "Z:\"
  let resolved = rawPath.trim();
  if (/^[A-Za-z]:$/.test(resolved)) resolved = resolved + "\\";
  let dir = path.resolve(resolved);

  if (!fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) {
    dir = defaultDir;
  }

  const parentDir = path.dirname(dir);
  const atRoot = parentDir === dir;

  let entries: FsEntry[] = [];
  try {
    const items = fs.readdirSync(dir, { withFileTypes: true });
    entries = items
      .map((item) => ({
        name: item.name,
        type: item.isDirectory() ? ("dir" as const) : ("file" as const),
        path: path.join(dir, item.name),
      }))
      .sort((a, b) => {
        if (a.type !== b.type) return a.type === "dir" ? -1 : 1;
        return a.name.localeCompare(b.name);
      })
      .filter((e) => !e.name.startsWith("."));
  } catch {
    // permission denied — return empty list
  }

  return NextResponse.json({
    current: dir,
    parent: atRoot ? null : parentDir,
    homedir: os.homedir(),
    defaultDir,
    drives: listDrives(),
    entries,
  });
}
