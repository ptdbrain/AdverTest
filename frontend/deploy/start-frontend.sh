#!/bin/sh
set -eu

# Next.js embeds NEXT_PUBLIC_* values during `next build`.  Render supplies
# service variables when the container starts, so publish the API URL as a
# tiny runtime config before serving the prebuilt frontend.
node -e '
  const fs = require("fs");
  fs.mkdirSync("public", { recursive: true });
  const config = { apiUrl: process.env.NEXT_PUBLIC_API_URL || "" };
  fs.writeFileSync("public/runtime-config.js", `window.__ADVERTEST_RUNTIME_CONFIG__ = ${JSON.stringify(config)};\n`);
'

exec npm run start -- -p "${PORT:-3000}"
