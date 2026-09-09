import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

function upstreamUrl(path, search) {
  const origin =
    process.env.API_ORIGIN ||
    process.env.NEXT_PUBLIC_API_URL ||
    process.env.API_PROXY_TARGET ||
    "https://advertest-portable-api-smoke-20260905.onrender.com";
  return `${origin.replace(/\/$/, "")}/api/${path.join("/")}${search}`;
}

async function proxy(request, { params }) {
  try {
    const { path } = await params;
    const headers = new Headers(request.headers);
    headers.delete("host");
    headers.delete("content-length");
    headers.delete("connection");
    headers.delete("transfer-encoding");
    const method = request.method;
    const body = method === "GET" || method === "HEAD" ? undefined : await request.arrayBuffer();
    const url = upstreamUrl(path, new URL(request.url).search);
    const upstreamRequest = () =>
      fetch(url, {
        method,
        headers,
        body,
        redirect: "manual",
        cache: "no-store",
      });
    let upstream = await upstreamRequest();
    // Render/upstream cold starts can briefly respond with 429 or 502/503/504
    // before the request reaches FastAPI. Retrying once allows transient gateway
    // restarts to settle without failing the user interaction.
    if (
      upstream.status === 429 ||
      (upstream.status >= 502 && upstream.status <= 504) ||
      (method === "GET" && upstream.status === 500)
    ) {
      const retryAfter = Number(upstream.headers.get("retry-after"));
      const delay = upstream.status === 429 ? (Number.isFinite(retryAfter) ? retryAfter * 1000 : 750) : 1200;
      await new Promise((resolve) => setTimeout(resolve, delay));
      upstream = await upstreamRequest();
    }
    const contentType = upstream.headers.get("content-type") || "";
    if (upstream.status >= 500 && !contentType.includes("application/json")) {
      const text = await upstream.text().catch(() => "");
      return NextResponse.json(
        {
          detail: `Hệ thống backend đang khởi động lại hoặc tạm thời quá tải (HTTP ${upstream.status}). Vui lòng thử lại sau vài giây.`,
          upstream_status: upstream.status,
          upstream_snippet: text.slice(0, 300),
        },
        { status: upstream.status }
      );
    }
    const responseHeaders = new Headers(upstream.headers);
    responseHeaders.delete("content-encoding");
    responseHeaders.delete("content-length");
    return new NextResponse(upstream.body, { status: upstream.status, headers: responseHeaders });
  } catch (error) {
    console.error("API proxy error:", error);
    return NextResponse.json({ detail: `API proxy unavailable: ${error.message}` }, { status: 502 });
  }
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
export const OPTIONS = proxy;
