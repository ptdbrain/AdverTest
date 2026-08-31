import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

function upstreamUrl(path, search) {
  const origin = process.env.API_ORIGIN || process.env.NEXT_PUBLIC_API_URL;
  if (!origin) throw new Error("API_ORIGIN is not configured");
  return `${origin.replace(/\/$/, "")}/api/${path.join("/")}${search}`;
}

async function proxy(request, { params }) {
  try {
    const { path } = await params;
    const headers = new Headers(request.headers);
    headers.delete("host");
    headers.delete("content-length");
    const method = request.method;
    const body = method === "GET" || method === "HEAD" ? undefined : await request.arrayBuffer();
    const url = upstreamUrl(path, new URL(request.url).search);
    const upstreamRequest = () => fetch(url, {
      method,
      headers,
      body,
      redirect: "manual",
      cache: "no-store",
    });
    let upstream = await upstreamRequest();
    // Render/upstream cold starts can briefly respond with 429 before the
    // request reaches FastAPI. A 429 is a refusal, not an accepted job, so a
    // bounded retry cannot enqueue a duplicate inference run.
    if (upstream.status === 429) {
      const retryAfter = Number(upstream.headers.get("retry-after"));
      await new Promise((resolve) => setTimeout(resolve, Number.isFinite(retryAfter) ? retryAfter * 1000 : 750));
      upstream = await upstreamRequest();
    }
    const responseHeaders = new Headers(upstream.headers);
    responseHeaders.delete("content-encoding");
    responseHeaders.delete("content-length");
    return new NextResponse(upstream.body, { status: upstream.status, headers: responseHeaders });
  } catch (error) {
    return NextResponse.json({ detail: `API proxy unavailable: ${error.message}` }, { status: 502 });
  }
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
export const OPTIONS = proxy;
