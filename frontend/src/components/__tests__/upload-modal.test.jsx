import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import UploadModal from "@/components/UploadModal.jsx";

vi.mock("@/lib/api", () => ({
  createUploadBatch: vi.fn(), uploadImage: vi.fn(), startFolderDatasetImport: vi.fn(), getFolderDatasetImportJob: vi.fn(),
  finalizeUploadBatch: vi.fn(), saveBatchAnnotation: vi.fn(),
}));

afterEach(cleanup);

describe("UploadModal", () => {
  it("offers a separate attacked dataset ingestion flow", () => {
    render(<UploadModal isOpen onClose={vi.fn()} onDatasetCreated={vi.fn()} taskId="detection2d" />);

    expect(screen.getByRole("button", { name: /attacked dataset/i })).toBeVisible();
  });
});
