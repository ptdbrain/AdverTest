import React from "react";
import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

import ClassMappingCard from "@/components/ClassMappingCard";

describe("ClassMappingCard", () => {
  it("keeps an explicit mapping when parent arrays are recreated", () => {
    const onChange = vi.fn();
    const props = {
      datasetClasses: ["Car", "Pedestrian"],
      modelClasses: ["car", "person"],
      source: "manifest",
      onChange,
    };
    const { rerender } = render(<ClassMappingCard {...props} />);

    fireEvent.change(screen.getByLabelText("Mapping Pedestrian"), {
      target: { value: "person" },
    });
    expect(onChange).toHaveBeenLastCalledWith({ Car: "car", Pedestrian: "person" });

    rerender(
      <ClassMappingCard
        {...props}
        datasetClasses={["Car", "Pedestrian"]}
        modelClasses={["car", "person"]}
      />,
    );
    expect(screen.getByLabelText("Mapping Pedestrian")).toHaveValue("person");
  });

  it("records intentionally ignored labels", () => {
    const onChange = vi.fn();
    render(
      <ClassMappingCard
        datasetClasses={["Cyclist"]}
        modelClasses={["car"]}
        source="manifest"
        onChange={onChange}
      />,
    );

    fireEvent.change(screen.getByLabelText("Mapping Cyclist"), {
      target: { value: "__ignore__" },
    });
    expect(onChange).toHaveBeenLastCalledWith({ Cyclist: "ignore" });
  });
});
