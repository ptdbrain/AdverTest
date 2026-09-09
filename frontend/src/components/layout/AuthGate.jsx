"use client";

import { usePathname, useRouter } from "next/navigation";
import { useEffect, useRef } from "react";
import { useAuth } from "@/context/AuthContext";

// Pages any visitor may browse without an account ("/" renders the dashboard).
const PUBLIC_PATHS = ["/", "/dashboard"];
const isPublic = (pathname) => PUBLIC_PATHS.includes(pathname);

/**
 * Layout-level auth guard.
 *
 * Workflow §1: after successful authentication the user is taken to the
 * overview dashboard — so signing in from any internal page redirects there.
 * Conversely, expired sessions never show a blank shell: losing authentication
 * on an internal page logs the user back out, redirects to the dashboard and
 * re-opens the global login modal.
 */
export default function AuthGate({ children }) {
  const pathname = usePathname() || "/";
  const router = useRouter();
  const { isAuthenticated, isLoading, openAuthModal } = useAuth();
  // Track the previous auth state so a false → true transition (successful
  // login) can trigger exactly one dashboard redirect.
  const wasAuthenticated = useRef(null);

  useEffect(() => {
    if (isLoading) return;

    if (isAuthenticated) {
      if (wasAuthenticated.current === false && !isPublic(pathname)) {
        router?.push?.("/dashboard");
      }
    } else if (!isPublic(pathname)) {
      // Not authenticated on an internal page: land on the dashboard and show
      // the global login modal instead of gating behind a blank screen.
      router?.push?.("/dashboard");
      openAuthModal();
    }

    wasAuthenticated.current = isAuthenticated;
  }, [isAuthenticated, isLoading, openAuthModal, pathname, router]);

  // Render children regardless — the dashboard stays browsable while signed
  // out (its sidebar carries the login entry point), and internal pages only
  // flash for the instant the redirect takes over.
  return children;
}
