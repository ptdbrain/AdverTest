"""Authentication router: user registration, login, Google SSO, and profile introspection."""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status

from src.auth.contracts import BrowserSessionOut, GoogleAuthIn, LoginIn, TokenOut, UserCreateIn, UserOut
from src.auth.dependencies import get_auth_service, get_current_user, get_current_user_optional
from src.auth.service import AuthService
from src.config import get_settings

router = APIRouter(prefix="/auth", tags=["Authentication"])


def _set_browser_session(response: Response, token: str) -> None:
    settings = get_settings()
    response.set_cookie(
        key=settings.auth_cookie_name,
        value=token,
        httponly=True,
        # The public UI is hosted on a different site from the API. Production
        # therefore needs a cross-site cookie, which browsers only accept when
        # it is both SameSite=None and Secure.
        secure=settings.auth_cookie_secure or settings.app_env == "production",
        samesite="none" if settings.app_env == "production" else settings.auth_cookie_samesite,
        max_age=24 * 3600,
    )


@router.post("/register", response_model=BrowserSessionOut, status_code=status.HTTP_201_CREATED)
async def register_user(
    payload: UserCreateIn,
    response: Response,
    auth_service: AuthService = Depends(get_auth_service),
    ) -> BrowserSessionOut:
    """Register a new user account with default RESEARCHER privileges."""
    try:
        user_out, token = auth_service.register(payload)
        _set_browser_session(response, token)
        return BrowserSessionOut(user=user_out)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/login", response_model=BrowserSessionOut)
async def login_user(
    payload: LoginIn,
    response: Response,
    auth_service: AuthService = Depends(get_auth_service),
) -> BrowserSessionOut:
    """Authenticate user credentials and receive a JWT Bearer access token."""
    try:
        user_out, token = auth_service.authenticate(payload.email, payload.password)
        _set_browser_session(response, token)
        return BrowserSessionOut(user=user_out)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


@router.post("/google", response_model=TokenOut)
async def login_google_sso(
    payload: GoogleAuthIn,
    response: Response,
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenOut:
    """Authenticate via Google Single Sign-On (ID Token or OAuth2 credential)."""
    try:
        user_out, token = auth_service.authenticate_google(payload)
        _set_browser_session(response, token)
        return TokenOut(
            access_token=token,
            expires_in_seconds=24 * 3600,
            user=user_out,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


@router.get("/google/config", operation_id="get_google_auth_config_canonical")
@router.get("/google-config", operation_id="get_google_auth_config_alias")
async def get_google_auth_config() -> dict[str, Any]:
    """Retrieve public Google OAuth2 Client ID, SSO status, and demo accounts.

    One-click demo cards are only exposed when demo bootstrap is enabled:
    local development, ``allow_dev_bootstrap_accounts``, or an intentional
    public demo deployment (``demo_bootstrap_accounts``). In a hardened
    production deploy they are omitted entirely.
    """
    settings = get_settings()
    client_id = os.getenv("GOOGLE_CLIENT_ID", "")
    demo_enabled = (
        settings.app_env != "production"
        or settings.allow_dev_bootstrap_accounts
        or settings.demo_bootstrap_accounts
    )
    demo_profiles: list[dict[str, Any]] = []
    if demo_enabled:
        demo_profiles = [
            {
                "email": "engineer@advertest.ai",
                "password": settings.demo_engineer_password,
                "display_name": "Alex (ML Engineer)",
                "avatar_url": "https://api.dicebear.com/7.x/bottts/svg?seed=alex",
                "role": "ENGINEER",
            },
            {
                "email": "admin@advertest.ai",
                "password": settings.admin_default_password,
                "display_name": "System Administrator",
                "avatar_url": "https://api.dicebear.com/7.x/bottts/svg?seed=admin",
                "role": "ADMIN",
            },
        ]
        if getattr(settings, "demo_fake_sessions", False):
            demo_profiles.append(
                {
                    "email": settings.demo_fake_engineer_email,
                    "password": settings.demo_fake_engineer_password,
                    "display_name": "Demo Engineer (Prepared Walkthrough)",
                    "avatar_url": "https://api.dicebear.com/7.x/bottts/svg?seed=demo-engineer",
                    "role": "ENGINEER",
                }
            )
    return {
        "client_id": client_id,
        "configured": bool(client_id),
        "demo_profiles": demo_profiles,
    }


@router.get("/me", response_model=UserOut)
async def get_my_profile(
    current_user: UserOut = Depends(get_current_user),
) -> UserOut:
    """Get the authenticated user's profile and active quotas."""
    return current_user


@router.get("/session", response_model=UserOut | None)
async def get_browser_session(
    current_user: UserOut | None = Depends(get_current_user_optional),
) -> UserOut | None:
    """Return the optional browser session without turning a logged-out visit into a 401."""
    return current_user


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
async def logout_user() -> Response:
    """End the browser session by expiring its HttpOnly cookie."""
    settings = get_settings()
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    response.delete_cookie(
        key=settings.auth_cookie_name,
        httponly=True,
        secure=settings.auth_cookie_secure or settings.app_env == "production",
        samesite="none" if settings.app_env == "production" else settings.auth_cookie_samesite,
    )
    return response
