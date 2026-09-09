import { describe, expect, test } from "vitest";
import { visibleAttacks } from "@/lib/attackCatalog";

describe("task-aware attack catalog", () => {
  test("visibleAttacks keeps only attacks declared for the selected task", () => {
    const catalog = [
      { name: "gaussian_noise", task_ids: ["detection2d", "segmentation"] },
      { name: "lidar_xyz_noise", task_ids: ["detection3d"] },
    ];

    expect(visibleAttacks(catalog, "detection3d").map((item) => item.name)).toEqual(["lidar_xyz_noise"]);
  });
});
