"use client";

import { useState } from "react";

interface Props {
  modelName: string;
  onClose: () => void;
  onDeleted: () => void;
}

export default function DeleteModal({ modelName, onClose, onDeleted }: Props) {
  const [deleteLogs, setDeleteLogs] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleDelete() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `/api/configs/${encodeURIComponent(modelName)}?deleteLogs=${deleteLogs}`,
        { method: "DELETE" }
      );
      const data = await res.json();
      if (!res.ok) { setError(data.error ?? "Delete failed"); return; }
      onDeleted();
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <div className="bg-gray-900 border border-gray-700 rounded-xl w-full max-w-sm p-6">
        <h2 className="text-lg font-semibold mb-2">Delete Model</h2>
        <p className="text-sm text-gray-400 mb-4">
          Delete config for <span className="text-white font-medium">{modelName}</span>?
        </p>

        <label className="flex items-center gap-2 text-sm text-gray-300 mb-6 cursor-pointer">
          <input
            type="checkbox"
            checked={deleteLogs}
            onChange={(e) => setDeleteLogs(e.target.checked)}
            className="accent-red-500"
          />
          Also delete training logs and checkpoints
        </label>

        {error && <p className="text-red-400 text-sm mb-4">{error}</p>}

        <div className="flex gap-3 justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm bg-gray-800 hover:bg-gray-700 rounded-md transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleDelete}
            disabled={loading}
            className="px-4 py-2 text-sm bg-red-600 hover:bg-red-500 disabled:opacity-50 rounded-md transition-colors font-medium"
          >
            {loading ? "Deleting…" : "Delete"}
          </button>
        </div>
      </div>
    </div>
  );
}
