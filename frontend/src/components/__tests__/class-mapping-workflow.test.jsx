import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import ClassMappingCard from "@/components/ClassMappingCard";

describe("strict class mapping workflow", () => {
  afterEach(cleanup);
  it("renders the persisted controlled value and does not overwrite it after rerender", () => {
    const onChange = vi.fn();
    const { rerender } = render(
      <ClassMappingCard
        datasetClasses={["Car", "Pedestrian"]}
        modelClasses={["car", "person"]}
        source="manifest"
        value={{ Car: "person", Pedestrian: "ignore" }}
        onChange={onChange}
      />,
    );
    expect(screen.getByLabelText("Mapping Car")).toHaveValue("person");
    expect(screen.getByLabelText("Mapping Pedestrian")).toHaveValue("__ignore__");

    rerender(
      <ClassMappingCard
        datasetClasses={["Car", "Pedestrian"]}
        modelClasses={["car", "person"]}
        source="manifest"
        value={{ Car: "car", Pedestrian: "person" }}
        onChange={onChange}
      />,
    );
    expect(screen.getByLabelText("Mapping Car")).toHaveValue("car");
    expect(screen.getByLabelText("Mapping Pedestrian")).toHaveValue("person");
  });

  it("reports every missing dataset class and emits the complete controlled mapping", () => {
    const onChange = vi.fn();
    render(
      <ClassMappingCard
        datasetClasses={["Car", "Pedestrian"]}
        modelClasses={["car", "person"]}
        source="manifest"
        value={{ Car: "car", Pedestrian: null }}
        onChange={onChange}
      />,
    );

    expect(screen.getByRole("alert")).toHaveTextContent("Pedestrian");
    fireEvent.change(screen.getByLabelText("Mapping Pedestrian"), { target: { value: "__ignore__" } });
    expect(onChange).toHaveBeenLastCalledWith({ Car: "car", Pedestrian: "ignore" });
  });
});
