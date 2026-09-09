import { fireEvent, render, screen } from "@testing-library/react";
import React from "react";
import { describe, expect, it, vi } from "vitest";

import ClassMappingCard from "@/components/ClassMappingCard";

function ControlledMapping(props) {
  const [value, setValue] = React.useState({});
  return <ClassMappingCard {...props} value={value} onChange={setValue} />;
}

describe("ClassMappingCard", () => {
  it("keeps an explicit mapping when parent arrays are recreated", () => {
    const props = {
      datasetClasses: ["Car", "Pedestrian"],
      modelClasses: ["car", "person"],
      source: "manifest",
    };
    const { rerender } = render(<ControlledMapping {...props} />);

    fireEvent.change(screen.getByLabelText("Mapping Pedestrian"), {
      target: { value: "person" },
    });
    rerender(<ControlledMapping {...props} datasetClasses={["Car", "Pedestrian"]} modelClasses={["car", "person"]} />);
    expect(screen.getByLabelText("Mapping Pedestrian")).toHaveValue("person");
  });

  it("records intentionally ignored labels", () => {
    render(
      <ControlledMapping datasetClasses={["Cyclist"]} modelClasses={["car"]} source="manifest" />,
    );

    fireEvent.change(screen.getByLabelText("Mapping Cyclist"), {
      target: { value: "__ignore__" },
    });
    expect(screen.getByLabelText("Mapping Cyclist")).toHaveValue("__ignore__");
  });
});
