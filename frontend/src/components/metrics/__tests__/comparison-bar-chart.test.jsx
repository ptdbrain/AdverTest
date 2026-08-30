import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import ComparisonBarChart from "../ComparisonBarChart";

describe("ComparisonBarChart", () => {
  it("does not replace absent API metrics with illustrative values", () => {
    render(<ComparisonBarChart data={[]} />);
    expect(screen.getByText(/No data/i)).toBeTruthy();
  });
});
