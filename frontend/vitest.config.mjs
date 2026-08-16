import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig({
  plugins: [react({ include: /src\/.*\.[jt]sx?$/ })],
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.js"],
  },
  resolve: {
    alias: { "@": path.resolve(import.meta.dirname, "./src") },
  },
});
