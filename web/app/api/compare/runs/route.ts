import { NextRequest, NextResponse } from "next/server";
import pool from "@/lib/db";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  const model = req.nextUrl.searchParams.get("model");
  try {
    const { rows } = await pool.query(
      `SELECT
         r.id,
         r.model_name,
         r.version,
         r.created_at,
         r.metadata,
         COUNT(l.id)::int          AS epoch_count,
         MIN(l.train_loss)         AS best_train_loss,
         MIN(l.val_loss)           AS best_val_loss
       FROM training_runs r
       LEFT JOIN training_logs l ON l.run_id = r.id
       ${model ? "WHERE r.model_name = $1" : ""}
       GROUP BY r.id
       ORDER BY r.created_at DESC`,
      model ? [model] : []
    );
    return NextResponse.json(rows);
  } catch (e) {
    console.error(e);
    return NextResponse.json({ error: "DB error" }, { status: 500 });
  }
}
