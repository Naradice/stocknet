"use client";

import { useEffect, useState, useCallback } from "react";

interface FsEntry {
  name: string;
  type: "file" | "dir";
  path: string;
}

interface BrowseResult {
  current: string;
  parent: string | null;
  homedir: string;
  defaultDir: string;
  drives: string[];
  entries: FsEntry[];
}

interface Props {
  /** File extensions to accept, e.g. [".csv"]. Empty = all files. */
  accept?: string[];
  onSelect: (filePath: string) => void;
  onClose: () => void;
}

export default function FilePicker({ accept = [".csv"], onSelect, onClose }: Props) {
  const [data, setData] = useState<BrowseResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [manualPath, setManualPath] = useState("");
  const [selected, setSelected] = useState<string | null>(null);

  const browse = useCallback(async (dirPath?: string) => {
    setLoading(true);
    setError(null);
    try {
      const url = "/api/fs/browse" + (dirPath ? `?path=${encodeURIComponent(dirPath)}` : "");
      const res = await fetch(url, { cache: "no-store" });
      if (!res.ok) { setError("Failed to browse"); return; }
      const json: BrowseResult = await res.json();
      setData(json);
      setManualPath(json.current);
      setSelected(null);
    } catch {
      setError("Network error");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { browse(); }, [browse]);

  function handleManualGo() {
    const p = manualPath.trim();
    if (p) browse(p);
  }

  function handleEntryClick(entry: FsEntry) {
    if (entry.type === "dir") {
      browse(entry.path);
    } else {
      setSelected((prev) => (prev === entry.path ? null : entry.path));
    }
  }

  function handleConfirm() {
    if (selected) { onSelect(selected); onClose(); }
  }

  const isAccepted = (name: string) =>
    accept.length === 0 || accept.some((ext) => name.toLowerCase().endsWith(ext));

  const visibleEntries = (data?.entries ?? []).filter(
    (e) => e.type === "dir" || isAccepted(e.name)
  );

  const segments = data ? buildSegments(data.current) : [];

  // Which drive letter is currently active
  const activeDrive = data?.current.match(/^([A-Za-z]:\\)/)?.[1].toUpperCase() ?? null;

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/70 p-4">
      <div className="bg-gray-900 border border-gray-700 rounded-xl w-full max-w-2xl flex flex-col" style={{ maxHeight: "85vh" }}>

        {/* ── Header ─────────────────────────────────────────── */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-gray-800 shrink-0">
          <h3 className="font-semibold text-sm">Select File</h3>
          <button onClick={onClose} className="text-gray-400 hover:text-white text-xl leading-none">×</button>
        </div>

        {/* ── Drive selector (Windows only) ──────────────────── */}
        {data && data.drives.length > 1 && (
          <div className="px-4 py-2 border-b border-gray-800 flex items-center gap-1.5 overflow-x-auto shrink-0">
            <span className="text-xs text-gray-500 mr-1 shrink-0">Drive:</span>
            {data.drives.map((drive) => {
              const label = drive.replace(/\\$/, ""); // "C:" not "C:\"
              const isActive = activeDrive === drive.toUpperCase();
              return (
                <button
                  key={drive}
                  onClick={() => browse(drive)}
                  className={`px-2.5 py-1 text-xs rounded border transition-colors shrink-0 font-mono ${
                    isActive
                      ? "bg-blue-600 border-blue-500 text-white"
                      : "bg-gray-800 border-gray-700 text-gray-300 hover:bg-gray-700 hover:border-gray-600"
                  }`}
                >
                  {label}
                </button>
              );
            })}
          </div>
        )}

        {/* ── Address bar ────────────────────────────────────── */}
        <div className="px-4 py-2 border-b border-gray-800 flex gap-2 shrink-0">
          <input
            className="flex-1 bg-gray-800 border border-gray-700 rounded-md px-3 py-1.5 text-xs font-mono focus:outline-none focus:border-blue-500 min-w-0"
            value={manualPath}
            onChange={(e) => setManualPath(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") handleManualGo(); }}
            placeholder="Paste a path and press Enter"
          />
          <button
            onClick={handleManualGo}
            className="px-3 py-1.5 text-xs bg-gray-700 hover:bg-gray-600 rounded-md transition-colors shrink-0"
          >
            Go
          </button>
        </div>

        {/* ── Breadcrumb ─────────────────────────────────────── */}
        {segments.length > 0 && (
          <div className="px-4 py-1.5 border-b border-gray-800 flex items-center gap-1 overflow-x-auto shrink-0 text-xs text-gray-400">
            {segments.map((seg, i) => (
              <span key={i} className="flex items-center gap-1 shrink-0">
                {i > 0 && <span className="text-gray-600">/</span>}
                <button
                  onClick={() => browse(seg.path)}
                  className="hover:text-white transition-colors truncate max-w-[140px]"
                  title={seg.path}
                >
                  {seg.label}
                </button>
              </span>
            ))}
          </div>
        )}

        {/* ── File list ──────────────────────────────────────── */}
        <div className="flex-1 overflow-y-auto min-h-0">
          {loading && (
            <div className="flex items-center justify-center py-10 text-gray-500 text-sm">
              Loading…
            </div>
          )}
          {!loading && error && (
            <div className="px-4 py-3 text-red-400 text-sm">{error}</div>
          )}
          {!loading && !error && (
            <ul>
              {data?.parent && (
                <li>
                  <button
                    onClick={() => browse(data.parent!)}
                    className="w-full flex items-center gap-3 px-4 py-2 hover:bg-gray-800 transition-colors text-left text-sm text-gray-400"
                  >
                    <span>↩</span>
                    <span className="font-mono text-xs">..</span>
                  </button>
                </li>
              )}
              {visibleEntries.length === 0 && (
                <li className="px-4 py-8 text-center text-gray-600 text-sm">
                  {accept.length > 0
                    ? `No ${accept.join(", ")} files in this folder`
                    : "Empty folder"}
                </li>
              )}
              {visibleEntries.map((entry) => {
                const isSelected = selected === entry.path;
                return (
                  <li key={entry.path}>
                    <button
                      onClick={() => handleEntryClick(entry)}
                      className={`w-full flex items-center gap-3 px-4 py-2 transition-colors text-left text-sm ${
                        isSelected
                          ? "bg-blue-600/20 text-blue-300"
                          : "hover:bg-gray-800 text-gray-300"
                      }`}
                    >
                      <span className="shrink-0">{entry.type === "dir" ? "📁" : "📄"}</span>
                      <span className="truncate flex-1">{entry.name}</span>
                      {entry.type === "file" && (
                        <span className="text-xs text-gray-500 shrink-0 font-mono">
                          {entry.name.split(".").pop()?.toUpperCase()}
                        </span>
                      )}
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>

        {/* ── Footer ─────────────────────────────────────────── */}
        <div className="px-4 py-3 border-t border-gray-800 shrink-0">
          <div className="flex items-center gap-3">
            <div className="flex-1 min-w-0">
              {selected ? (
                <p className="text-xs text-gray-300 font-mono truncate" title={selected}>
                  {selected}
                </p>
              ) : (
                <p className="text-xs text-gray-600">No file selected</p>
              )}
            </div>
            <button
              onClick={onClose}
              className="px-3 py-1.5 text-xs bg-gray-800 hover:bg-gray-700 rounded-md transition-colors shrink-0"
            >
              Cancel
            </button>
            <button
              onClick={handleConfirm}
              disabled={!selected}
              className="px-4 py-1.5 text-xs bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed rounded-md transition-colors font-medium shrink-0"
            >
              Select
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function buildSegments(fullPath: string): { label: string; path: string }[] {
  const isWindows = /^[A-Za-z]:\\/.test(fullPath);
  const sep = isWindows ? "\\" : "/";
  const parts = fullPath.split(/[\\/]/).filter(Boolean);

  const segments: { label: string; path: string }[] = [];
  let accumulated = "";

  for (let i = 0; i < parts.length; i++) {
    if (isWindows && i === 0) {
      accumulated = parts[0] + "\\";
      segments.push({ label: parts[0], path: accumulated });
    } else {
      accumulated = accumulated
        ? accumulated.replace(/[\\/]$/, "") + sep + parts[i]
        : sep + parts[i];
      segments.push({ label: parts[i], path: accumulated });
    }
  }

  return segments;
}
