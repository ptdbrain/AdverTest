import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";

const eslintConfig = defineConfig([
  ...nextVitals,
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
  ]),
  {
    rules: {
      // New React Compiler guidance is useful for new code, but NewUI still
      // hydrates persisted UI state inside effects. Keep this non-blocking
      // until those flows are migrated incrementally.
      "react-hooks/set-state-in-effect": "off",
    },
  },
]);

export default eslintConfig;
