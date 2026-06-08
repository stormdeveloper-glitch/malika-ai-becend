import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fastapi import Depends, FastAPI, HTTPException, Request as FastAPIRequest, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "backend_templates"
LOG_FILE = BASE_DIR / "app_monitoring.log"

LOGGER_NAME = "malika_ai_backend"
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


APP_ENV = os.getenv("APP_ENV", "development").lower()
JWT_SECRET = os.getenv("JWT_SECRET") or os.getenv("MALIKA_JWT_SECRET")
JWT_ISSUER = os.getenv("JWT_ISSUER", "https://api.texnoo.com")
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "malika-ai-users")
JWT_EXPIRES_SECONDS = int(os.getenv("JWT_EXPIRES_SECONDS", "3600"))
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"

if not JWT_SECRET:
    if APP_ENV == "production":
        logger.error("JWT_SECRET is missing in production environment")
        raise RuntimeError("JWT_SECRET must be configured in production")
    JWT_SECRET = "local-development-secret-change-before-production"
    logger.error("JWT_SECRET is missing; using development-only fallback secret")


app = FastAPI(
    title="Malika AI Core Backend",
    description="FastAPI backend service for Malika AI running under api.texnoo.com",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://texnoo.com", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

app.mount(
    "/backend-static",
    StaticFiles(directory=TEMPLATES_DIR),
    name="backend-static",
)

bearer_scheme = HTTPBearer(auto_error=False)


class GoogleAuthRequest(BaseModel):
    token: str = Field(..., min_length=20, description="Google ID token from frontend")


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class AIGenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    metadata: dict[str, Any] | None = None


class AIGenerateResponse(BaseModel):
    result: str
    session_user: str
    tools_enabled: bool


def _base64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _base64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _json_dumps(data: dict[str, Any]) -> bytes:
    return json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")


def create_access_token(subject: str, extra_claims: dict[str, Any] | None = None) -> str:
    issued_at = int(time.time())
    payload: dict[str, Any] = {
        "sub": subject,
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "iat": issued_at,
        "exp": issued_at + JWT_EXPIRES_SECONDS,
    }
    if extra_claims:
        payload.update(extra_claims)

    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = (
        f"{_base64url_encode(_json_dumps(header))}."
        f"{_base64url_encode(_json_dumps(payload))}"
    )
    signature = hmac.new(
        JWT_SECRET.encode("utf-8"),
        signing_input.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{signing_input}.{_base64url_encode(signature)}"


def verify_access_token(token: str) -> dict[str, Any]:
    try:
        header_part, payload_part, signature_part = token.split(".")
        signing_input = f"{header_part}.{payload_part}"
        expected_signature = hmac.new(
            JWT_SECRET.encode("utf-8"),
            signing_input.encode("ascii"),
            hashlib.sha256,
        ).digest()
        actual_signature = _base64url_decode(signature_part)

        if not hmac.compare_digest(expected_signature, actual_signature):
            raise ValueError("JWT signature mismatch")

        header = json.loads(_base64url_decode(header_part))
        payload = json.loads(_base64url_decode(payload_part))

        if header.get("alg") != "HS256":
            raise ValueError("Unsupported JWT algorithm")
        if payload.get("iss") != JWT_ISSUER:
            raise ValueError("Invalid JWT issuer")
        if payload.get("aud") != JWT_AUDIENCE:
            raise ValueError("Invalid JWT audience")
        if int(payload.get("exp", 0)) < int(time.time()):
            raise ValueError("JWT has expired")

        return payload
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.error("JWT verification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict[str, Any]:
    if credentials is None or credentials.scheme.lower() != "bearer":
        logger.error("Protected endpoint requested without Bearer token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer session token is required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return verify_access_token(credentials.credentials)


def _fetch_google_tokeninfo_sync(token: str) -> dict[str, Any]:
    query = urlencode({"id_token": token})
    request = Request(
        f"{GOOGLE_TOKENINFO_URL}?{query}",
        headers={"Accept": "application/json"},
        method="GET",
    )

    try:
        with urlopen(request, timeout=8) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        logger.error("Google token validation HTTP error: %s", exc.code)
        raise ValueError("Google token validation failed") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.error("Google token validation transport error: %s", exc)
        raise ValueError("Google token validation is unavailable") from exc


async def validate_google_token(token: str) -> dict[str, Any]:
    token_info = await asyncio.to_thread(_fetch_google_tokeninfo_sync, token)

    if GOOGLE_CLIENT_ID and token_info.get("aud") != GOOGLE_CLIENT_ID:
        logger.error("Google token audience mismatch")
        raise ValueError("Invalid Google token audience")

    if token_info.get("email_verified") not in (True, "true", "True", "1"):
        logger.error("Google token email is not verified")
        raise ValueError("Google account email is not verified")

    expires_at = int(token_info.get("exp", 0))
    if expires_at < int(time.time()):
        logger.error("Google token has expired")
        raise ValueError("Google token has expired")

    return token_info


@app.middleware("http")
async def request_logging_middleware(
    request: FastAPIRequest,
    call_next: Any,
) -> Any:
    started_at = time.perf_counter()
    logger.info("Incoming request: %s %s", request.method, request.url.path)

    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled request error: %s %s", request.method, request.url.path)
        raise

    elapsed_ms = (time.perf_counter() - started_at) * 1000
    logger.info(
        "Completed request: %s %s status=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    logger.info("Health check requested")
    return {"status": "ok", "service": "malika-ai-core"}


@app.get("/admin/monitoring", response_class=HTMLResponse, tags=["admin"])
async def monitoring_page() -> HTMLResponse:
    monitoring_file = TEMPLATES_DIR / "monitoring.html"
    logger.info("Admin monitoring page requested")

    if not monitoring_file.exists() or not monitoring_file.is_file():
        logger.error("Monitoring HTML file not found: %s", monitoring_file)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Monitoring page is not configured",
        )

    html = await asyncio.to_thread(monitoring_file.read_text, encoding="utf-8")
    return HTMLResponse(content=html, status_code=status.HTTP_200_OK)


@app.post("/api/v1/auth/google", response_model=AuthResponse, tags=["auth"])
async def google_auth(payload: GoogleAuthRequest) -> AuthResponse:
    logger.info("Google auth request received")

    try:
        google_user = await validate_google_token(payload.token)
    except ValueError as exc:
        logger.error("Google auth rejected: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google authentication failed",
        ) from exc

    user_email = google_user.get("email")
    google_subject = google_user.get("sub")

    if not user_email or not google_subject:
        logger.error("Google token missing required user fields")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google authentication failed",
        )

    access_token = create_access_token(
        subject=str(google_subject),
        extra_claims={"email": user_email, "provider": "google"},
    )
    logger.info("Session created for Google user: %s", user_email)

    return AuthResponse(access_token=access_token, expires_in=JWT_EXPIRES_SECONDS)


@app.post("/api/v1/ai/generate", response_model=AIGenerateResponse, tags=["ai"])
async def ai_generate(
    payload: AIGenerateRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> AIGenerateResponse:
    user_identifier = str(current_user.get("email") or current_user.get("sub"))
    logger.info("AI generate request received from user: %s", user_identifier)

    normalized_prompt = payload.prompt.strip()
    if not normalized_prompt:
        logger.error("AI generate rejected empty prompt for user: %s", user_identifier)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Prompt must not be empty",
        )

    logger.info(
        "AI prompt accepted for user=%s prompt_length=%s",
        user_identifier,
        len(normalized_prompt),
    )

    return AIGenerateResponse(
        result="Malika AI yadrosi so'rovni qabul qildi. AI tools/function calling moduli keyingi bosqichda ulanadi.",
        session_user=user_identifier,
        tools_enabled=False,
    )
