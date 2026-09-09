import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import AuthModal from "@/components/AuthModal";

afterEach(cleanup);

const mockRegister = vi.fn(async () => ({ user: { id: "usr-new" } }));
const mockLogin = vi.fn(async () => ({ user: { id: "usr-admin" } }));
const mockGoogleLogin = vi.fn(async () => ({ user: { id: "usr-g" } }));

vi.mock("@/context/AuthContext", () => ({
  useAuth: () => ({
    isAuthModalOpen: true,
    closeAuthModal: vi.fn(),
    googleConfig: { client_id: "", configured: false, demo_profiles: [] },
    loginWithGoogle: mockGoogleLogin,
    loginWithCredentials: mockLogin,
    registerWithCredentials: mockRegister,
    isLoading: false,
  }),
}));

vi.mock("@/context/LanguageContext", () => ({
  useLanguage: () => ({ t: (key) => key }),
}));

function openRegisterTab() {
  return userEvent.setup().click(screen.getByRole("button", { name: "Đăng ký" }));
}

function fieldInput(labelText) {
  return within(screen.getByText(labelText).closest("div")).getByPlaceholderText("••••••••");
}

/** The eye button belonging to the field labelled `labelText` ("Mật khẩu" | "Nhập lại mật khẩu"). */
function eyeFor(labelText) {
  return within(screen.getByText(labelText).closest("div")).getByRole("button", { name: /mật khẩu$/i });
}

describe("AuthModal — password confirmation & visibility toggle", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("register form has a confirm-password field", async () => {
    render(<AuthModal isOpen />);
    await openRegisterTab();

    expect(screen.getByText("Nhập lại mật khẩu")).toBeInTheDocument();
    expect(fieldInput("Nhập lại mật khẩu")).toHaveAttribute("type", "password");
  });

  it("shows the minimum-length hint and rejects short passwords before submitting", async () => {
    const user = userEvent.setup();
    render(<AuthModal isOpen />);
    await openRegisterTab();

    expect(screen.getByText("Tối thiểu 8 ký tự")).toBeInTheDocument();

    await user.type(screen.getByPlaceholderText("Nguyen Van A"), "Tester");
    await user.type(screen.getByPlaceholderText("user@organization.vn"), "tester@example.com");
    await user.type(fieldInput("Mật khẩu"), "abc1234");
    await user.type(fieldInput("Nhập lại mật khẩu"), "abc1234");

    await user.click(screen.getByRole("button", { name: "Đăng ký & Đăng nhập ngay" }));

    expect(await screen.findByText("Mật khẩu phải có ít nhất 8 ký tự.")).toBeInTheDocument();
    expect(mockRegister).not.toHaveBeenCalled();
  });

  it("eye toggle reveals the password being typed on the register form", async () => {
    const user = userEvent.setup();
    render(<AuthModal isOpen />);
    await openRegisterTab();

    const passwordInput = fieldInput("Mật khẩu");
    expect(passwordInput).toHaveAttribute("type", "password");

    await user.type(passwordInput, "MatKhauBiMat123");
    await user.click(eyeFor("Mật khẩu"));

    expect(passwordInput).toHaveAttribute("type", "text");
    expect(passwordInput).toHaveValue("MatKhauBiMat123");

    await user.click(eyeFor("Mật khẩu"));
    expect(passwordInput).toHaveAttribute("type", "password");
  });

  it("eye toggle also works for the confirm-password and login fields", async () => {
    const user = userEvent.setup();
    render(<AuthModal isOpen />);
    await openRegisterTab();

    const confirmInput = fieldInput("Nhập lại mật khẩu");
    await user.type(confirmInput, "KhacNhau99");
    await user.click(eyeFor("Nhập lại mật khẩu"));
    expect(confirmInput).toHaveAttribute("type", "text");
    expect(confirmInput).toHaveValue("KhacNhau99");

    await user.click(screen.getByRole("button", { name: "Đăng nhập" }));
    const loginInput = fieldInput("Mật khẩu");
    await user.type(loginInput, "LoginPw1!");
    await user.click(eyeFor("Mật khẩu"));
    expect(loginInput).toHaveAttribute("type", "text");
    expect(loginInput).toHaveValue("LoginPw1!");
  });

  it("rejects registration when the re-entered password does not match", async () => {
    const user = userEvent.setup();
    render(<AuthModal isOpen />);
    await openRegisterTab();

    await user.type(screen.getByPlaceholderText("Nguyen Van A"), "Tester");
    await user.type(screen.getByPlaceholderText("user@organization.vn"), "tester@example.com");
    await user.type(fieldInput("Mật khẩu"), "abc12345");
    await user.type(fieldInput("Nhập lại mật khẩu"), "abc123456");

    await user.click(screen.getByRole("button", { name: "Đăng ký & Đăng nhập ngay" }));

    expect(await screen.findByText("Mật khẩu nhập lại không khớp. Vui lòng kiểm tra lại.")).toBeInTheDocument();
    expect(mockRegister).not.toHaveBeenCalled();
  });

  it("submits registration when passwords match", async () => {
    const user = userEvent.setup();
    render(<AuthModal isOpen />);
    await openRegisterTab();

    await user.type(screen.getByPlaceholderText("Nguyen Van A"), "Tester");
    await user.type(screen.getByPlaceholderText("user@organization.vn"), "tester@example.com");
    await user.type(fieldInput("Mật khẩu"), "abc12345");
    await user.type(fieldInput("Nhập lại mật khẩu"), "abc12345");

    await user.click(screen.getByRole("button", { name: "Đăng ký & Đăng nhập ngay" }));

    await waitFor(() =>
      expect(mockRegister).toHaveBeenCalledWith({
        email: "tester@example.com",
        password: "abc12345",
        display_name: "Tester",
      }),
    );
  });
});
