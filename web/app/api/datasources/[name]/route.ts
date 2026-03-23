import { NextResponse } from "next/server";
import { getDatasource, saveDatasource, deleteDatasource, Datasource } from "@/lib/datasources";

export const dynamic = "force-dynamic";

export async function GET(
  _req: Request,
  { params }: { params: { name: string } }
) {
  const ds = getDatasource(params.name);
  if (!ds) return NextResponse.json({ error: "Not found" }, { status: 404 });
  return NextResponse.json(ds);
}

export async function PUT(
  req: Request,
  { params }: { params: { name: string } }
) {
  const body = await req.json() as Datasource;
  // name in body must match URL param (or be absent — we set it from URL)
  const ds: Datasource = { ...body, name: params.name };
  saveDatasource(ds);
  return NextResponse.json({ ok: true });
}

export async function DELETE(
  _req: Request,
  { params }: { params: { name: string } }
) {
  deleteDatasource(params.name);
  return NextResponse.json({ ok: true });
}
