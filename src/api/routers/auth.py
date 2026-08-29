"""Authentication router: user registration, login, Google SSO, and profile introspection."""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from src.auth.contracts import GoogleAuthIn, LoginIn, TokenOut, UserCreateIn, UserOut
from src.auth.dependencies import get_auth_service, get_current_user
from src.auth.service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
async def register_user(
    payload: UserCreateIn,
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenOut:
    """Register a new user account with default RESEARCHER privileges."""
    try:
        user_out, token = auth_service.register(payload)
        return TokenOut(
            access_token=token,
            expires_in_seconds=24 * 3600,
            user=user_out,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/login", response_model=TokenOut)
async def login_user(
    payload: LoginIn,
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenOut:
    """Authenticate user credentials and receive a JWT Bearer access token."""
    try:
        user_out, token = auth_service.authenticate(payload.email, payload.password)
        return TokenOut(
            access_token=token,
            expires_in_seconds=24 * 3600,
            user=user_out,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


@router.post("/google", response_model=TokenOut)
async def login_google_sso(
    payload: GoogleAuthIn,
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenOut:
    """Authenticate via Google Single Sign-On (ID Token or OAuth2 credential)."""
    try:
        user_out, token = auth_service.authenticate_google(payload)
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
    """Retrieve public Google OAuth2 Client ID and SSO status."""
    client_id = os.getenv("GOOGLE_CLIENT_ID", "")
    return {
        "client_id": client_id,
        "configured": bool(client_id),
        "demo_profiles": [
            {
                "email": "alex.engineer@gmail.com",
                "display_name": "Alex Nguyen (Google ML Engineer)",
                "avatar_url": "https://api.dicebear.com/7.x/bottts/svg?seed=alex",
                "role": "ENGINEER",
            },
            {
                "email": "admin@advertest.ai",
                "display_name": "Elena Vance (System Administrator)",
                "avatar_url": "https://api.dicebear.com/7.x/bottts/svg?seed=admin",
                "role": "ADMIN",
            },
        ],
    }


@router.get("/me", response_model=UserOut)
async def get_my_profile(
    current_user: UserOut = Depends(get_current_user),
) -> UserOut:
    """Get the authenticated user's profile and active quotas."""
    return current_user
