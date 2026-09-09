import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import AdminView from "@/components/AdminView";

afterEach(cleanup);

const users = [
  {
    id: "usr-admin-1",
    email: "admin@advertest.ai",
    display_name: "System Administrator",
    role: "ADMIN",
    status: "ACTIVE",
    storage_quota_bytes: 10 * 1024 ** 3,
    compute_quota_hours: 100,
    last_login_at: "2026-09-01T08:00:00+00:00",
  },
  {
    id: "usr-eng-2",
    email: "engineer@advertest.ai",
    display_name: "Alex (ML Engineer)",
    role: "ENGINEER",
    status: "ACTIVE",
    storage_quota_bytes: 5 * 1024 ** 3,
    compute_quota_hours: 40.5,
    last_login_at: null,
  },
];

const quotas = {
  total_users: 2,
  active_users: 2,
  total_storage_used_bytes: 0,
  total_storage_used_mb: 0,
  total_runs_executed: 3,
  total_compute_hours: 1.5,
};

function jsonResponse(body, ok = true) {
  return { ok, status: ok ? 200 : 403, json: async () => body };
}

function mockFetch() {
  return vi.fn((url, options = {}) => {
    if (url === "/api/v1/admin/users") return Promise.resolve(jsonResponse(users));
    if (url === "/api/v1/admin/quotas") return Promise.resolve(jsonResponse(quotas));
    if (url === "/api/v1/admin/audit-logs") return Promise.resolve(jsonResponse([]));
    if (url === "/api/v1/admin/users/usr-eng-2/status" && options.method === "POST") {
      return Promise.resolve(jsonResponse({ ...users[1], ...JSON.parse(options.body) }));
    }
    return Promise.resolve(jsonResponse({}, false));
  });
}

/** Locate an action button inside the engineer's table row (rows render async). */
async function engineerRowButton(name) {
  const email = await screen.findByText("engineer@advertest.ai");
  return within(email.closest("tr")).getByRole("button", { name });
}

describe("AdminView", () => {
  beforeEach(() => {
    localStorage.setItem("advertest_auth_token", "test-token");
  });

  it("renders the user table and quota telemetry", async () => {
    global.fetch = mockFetch();
    render(<AdminView />);

    expect(await screen.findByText("admin@advertest.ai")).toBeInTheDocument();
    expect(screen.getByText("engineer@advertest.ai")).toBeInTheDocument();
    expect(screen.getByText("Total Registered Users")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
  });

  it("sends the Authorization header on admin API calls", async () => {
    const fetchMock = mockFetch();
    global.fetch = fetchMock;
    render(<AdminView />);

    await screen.findByText("admin@advertest.ai");
    const headers = fetchMock.mock.calls[0][1].headers;
    expect(headers.Authorization).toBe("Bearer test-token");
  });

  it("toggles a user status via the Suspend button", async () => {
    const fetchMock = mockFetch();
    global.fetch = fetchMock;
    const user = userEvent.setup();
    render(<AdminView />);

    await user.click(await engineerRowButton("Suspend"));

    await waitFor(() => {
      const statusCall = fetchMock.mock.calls.find(
        ([url, options]) => url === "/api/v1/admin/users/usr-eng-2/status" && options.method === "POST",
      );
      expect(statusCall).toBeTruthy();
      expect(JSON.parse(statusCall[1].body)).toEqual({ status: "SUSPENDED" });
    });
  });

  it("edits role and quotas through the Edit modal", async () => {
    const fetchMock = mockFetch();
    global.fetch = fetchMock;
    const user = userEvent.setup();
    render(<AdminView />);

    await user.click(await engineerRowButton("Edit"));

    const dialog = await screen.findByRole("dialog");
    expect(dialog).toHaveTextContent("engineer@advertest.ai");

    const roleSelect = screen.getByDisplayValue("ENGINEER");
    await user.selectOptions(roleSelect, "REVIEWER");
    const storageInput = screen.getByDisplayValue("5");
    await user.clear(storageInput);
    await user.type(storageInput, "20");
    const computeInput = screen.getByDisplayValue("40.5");
    await user.clear(computeInput);
    await user.type(computeInput, "60");

    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await waitFor(() => {
      const saveCall = fetchMock.mock.calls.find(
        ([url, options]) => url === "/api/v1/admin/users/usr-eng-2/status" && options.method === "POST",
      );
      expect(saveCall).toBeTruthy();
      expect(JSON.parse(saveCall[1].body)).toEqual({
        role: "REVIEWER",
        status: "ACTIVE",
        storage_quota_bytes: 20 * 1024 ** 3,
        compute_quota_hours: 60,
      });
    });
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });

  it("surfaces API errors inside the modal without closing it", async () => {
    global.fetch = vi.fn((url, options = {}) => {
      if (url === "/api/v1/admin/users") return Promise.resolve(jsonResponse(users));
      if (url === "/api/v1/admin/quotas") return Promise.resolve(jsonResponse(quotas));
      if (url === "/api/v1/admin/audit-logs") return Promise.resolve(jsonResponse([]));
      if (options.method === "POST") {
        return Promise.resolve({
          ok: false,
          status: 403,
          json: async () => ({ detail: "Administrator role required." }),
        });
      }
      return Promise.resolve(jsonResponse({}, false));
    });
    const user = userEvent.setup();
    render(<AdminView />);

    await user.click(await engineerRowButton("Edit"));
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    expect(await screen.findByText("Administrator role required.")).toBeInTheDocument();
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });

  it("shows the restricted-access panel for non-admin users", async () => {
    global.fetch = vi.fn(() => Promise.resolve(jsonResponse({}, false)));
    render(<AdminView />);

    expect(await screen.findByText("Administrator privileges required to view this panel.")).toBeInTheDocument();
  });
});
