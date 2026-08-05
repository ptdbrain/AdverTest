/** @type {import('next').NextConfig} */
const nextConfig = {
  /* Proxy API requests to backend during development */
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
    ];
  },
};

export default nextConfig;
