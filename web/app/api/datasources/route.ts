import { NextResponse } from "next/server";
import { listDatasources, saveDatasource, getDatasource, Datasource } from "@/lib/datasources";

export const dynamic = "force-dynamic";

export async function GET() {
  const list = listDatasources();
  return NextResponse.json(list);
}

export async function POST(req: Request) {
  const body = await req.json() as Datasource;

  if (!body.name || typeof body.name !== "string") {
    return NextResponse.json({ error: "name is required" }, { status: 400 });
  }
  if (!/^[a-zA-Z0-9_-]+$/.test(body.name)) {
    return NextResponse.json(
      { error: "name must contain only letters, numbers, underscores, hyphens" },
      { status: 400 }
    );
  }
  if (getDatasource(body.name)) {
    return NextResponse.json({ error: "A data source with that name already exists" }, { status: 409 });
  }

  saveDatasource(body);
  return NextResponse.json({ ok: true, name: body.name }, { status: 201 });
}
