"""Authentication router: user registration, login, and profile introspection."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from src.auth.contracts import LoginIn, TokenOut, UserCreateIn, UserOut
from src.auth.dependencies import get_auth_service, get_current_user
from src.auth.service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
async def register_user(
    payload: UserCreateIn,
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenOut:
    """Register a new user account. The first registered user automatically becomes ADMIN."""
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


@router.get("/me", response_model=UserOut)
async def get_my_profile(
    current_user: UserOut = Depends(get_current_user),
) -> UserOut:
    """Get the authenticated user's profile and active quotas."""
    return current_user
