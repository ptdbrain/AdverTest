/** @type {import('next').NextConfig} */

// When the browser calls the API same-origin (empty api base), this Next
// server proxies /api to the backend.  This keeps the session cookie
// first-party: a SameSite=Lax cookie is never attached to cross-site fetch
// requests, so letting the browser talk to http://127.0.0.1:8000 directly
// from http://localhost:3000 silently logs everyone out (401 on every
const rawProxyTarget =
  process.env.API_PROXY_TARGET ||
  process.env.API_ORIGIN ||
  process.env.NEXT_PUBLIC_API_URL ||
  "https://advertest-portable-api-smoke-20260905.onrender.com";
const API_PROXY_TARGET = rawProxyTarget.replace(/\/$/, "");

const nextConfig = {
  images: {
    remotePatterns: [
      { protocol: "http", hostname: "localhost", port: "8000", pathname: "/data/**" },
    ],
  },
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${API_PROXY_TARGET}/api/:path*` },
      { source: "/data/:path*", destination: `${API_PROXY_TARGET}/data/:path*` },
    ];
  },
  // The frontend is deployed independently from the API.  Never let a shared
  // cache keep an old HTML shell after a release: it can reference a previous
  // client bundle (including obsolete full-page overlays) and make navigation
  // appear unresponsive.
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          {
            key: "Cache-Control",
            value: "no-store, max-age=0, must-revalidate",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
