from slowapi import Limiter
import jwt
from typing import Optional
from fastapi import Request


def get_rate_limit_key(request: Request) -> str:
    """
    Extracts a rate limiting key from the request.
    For authenticated users, uses their user ID.
    For unauthenticated requests, falls back to their IP address.
    """
    user_id = _extract_user_id(request)
    if user_id:
        print(f"Rate limiting key: user:{user_id}")
        return f"user:{user_id}"

    ip = _get_client_ip(request)
    print(f"Rate limiting key: ip:{ip}")
    return f"ip:{ip}"


def _extract_user_id(request: Request) -> Optional[str]:
    """
    Extracts user ID from JWT token in Authorization header.
    Returns None if no valid token found.
    """
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        return None

    try:
        token = auth_header.split(" ")[1]
        payload = jwt.decode(token, options={"verify_signature": False})
        return payload.get("sub")
    except (jwt.InvalidTokenError, IndexError):
        return None


def _get_client_ip(request: Request) -> str:
    """
    Gets the original client IP from Cloudflare headers,
    falling back to X-Forwarded-For if CF headers aren't present.
    """
    # Try Cloudflare-specific header first
    cf_connecting_ip = request.headers.get("cf-connecting-ip")
    if cf_connecting_ip:
        return cf_connecting_ip

    # Fall back to X-Forwarded-For
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    return request.client.host


limiter = Limiter(key_func=get_rate_limit_key)
