import { describe, expect, it } from "vitest";
import config from "../../../next.config.mjs";

describe("frontend media proxy", () => {
  it("forwards both API and artifact media through the same target", async () => {
    const rewrites = await config.rewrites();

    expect(rewrites).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ source: "/api/:path*", destination: expect.stringMatching(/\/api\/:path\*$/) }),
        expect.objectContaining({ source: "/data/:path*", destination: expect.stringMatching(/\/data\/:path\*$/) }),
      ]),
    );
  });
});
