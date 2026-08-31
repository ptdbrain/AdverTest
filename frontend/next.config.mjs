/** @type {import('next').NextConfig} */
const apiOrigin = process.env.API_ORIGIN || process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const nextConfig = {
  images: {
    remotePatterns: [
      { protocol: "http", hostname: "localhost", port: "8000", pathname: "/data/**" },
    ],
  },
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  /* Proxy API requests to backend during development */
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${apiOrigin}/api/:path*`,
      },
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
