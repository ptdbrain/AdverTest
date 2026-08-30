"""FastAPI dependency providers for authentication and RBAC authorization."""

from __future__ import annotations

import functools

from fastapi import Depends, Header, HTTPException, status

from src.api.dependencies import get_store
from src.auth.contracts import UserOut
from src.auth.security import decode_access_token
from src.auth.service import AuthService


@functools.lru_cache
def get_auth_service() -> AuthService:
    """Singleton AuthService instance."""
    return AuthService(get_store())


async def get_current_user_optional(
    authorization: str | None = Header(default=None, alias="Authorization"),
    auth_service: AuthService = Depends(get_auth_service),
) -> UserOut | None:
    """Extract current user if Authorization header is provided, or return None."""
    if not authorization or not authorization.startswith("Bearer "):
        return None

    token = authorization.removeprefix("Bearer ").strip()
    claims = decode_access_token(token)
    if not claims or "sub" not in claims:
        return None

    return auth_service.get_user_by_id(claims["sub"])


async def get_current_user(
    authorization: str | None = Header(default=None, alias="Authorization"),
    auth_service: AuthService = Depends(get_auth_service),
) -> UserOut:
    """Require valid Bearer token and return active user profile."""
    user = await get_current_user_optional(authorization, auth_service)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if user.status != "ACTIVE":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Account is {user.status.lower()}.",
        )
    return user


async def require_admin(
    current_user: UserOut = Depends(get_current_user),
) -> UserOut:
    """Enforce ADMIN role authorization."""
    if current_user.role != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="INSUFFICIENT_PERMISSION: Administrator role required.",
        )
    return current_user
