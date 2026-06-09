import os
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from api.schemas import TokenResponse

SECRET_KEY = os.getenv("JWT_SECRET", "dev-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

USERS = {
    "admin": "admin123",
    "trader": "trader123",
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def authenticate_user(username: str, password: str) -> bool:
    return USERS.get(username) == password


def create_access_token(username: str) -> TokenResponse:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": username, "exp": expire}
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return TokenResponse(access_token=token)


def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")
        if username is None:
            raise exc
    except JWTError:
        raise exc
    return username
