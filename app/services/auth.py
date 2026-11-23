"""API-key authentication and collection-level access control."""
from __future__ import annotations

import hashlib
import secrets
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import Collection, CollectionAccess, User


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def create_user(db: Session, username: str, role: str = "user", api_key: Optional[str] = None) -> tuple[User, str]:
    key = api_key or "rag_" + secrets.token_urlsafe(24)
    user = User(username=username, api_key_hash=hash_key(key), role=role)
    db.add(user)
    db.flush()
    return user, key


def authenticate(db: Session, api_key: str) -> Optional[User]:
    return db.scalar(select(User).where(User.api_key_hash == hash_key(api_key)))


def bootstrap_admin(db: Session) -> None:
    s = get_settings()
    if not db.scalar(select(User).where(User.username == s.admin_username)):
        create_user(db, s.admin_username, role="admin", api_key=s.admin_api_key)


def can_read(db: Session, user: User, collection: Collection) -> bool:
    if user.role == "admin" or collection.owner_id == user.id:
        return True
    return db.get(CollectionAccess, (collection.id, user.id)) is not None


def can_write(user: User, collection: Collection) -> bool:
    return user.role == "admin" or collection.owner_id == user.id


def readable_collections(db: Session, user: User) -> list[Collection]:
    if user.role == "admin":
        return list(db.scalars(select(Collection).order_by(Collection.id)))
    shared = select(CollectionAccess.collection_id).where(CollectionAccess.user_id == user.id)
    return list(db.scalars(select(Collection).where(
        (Collection.owner_id == user.id) | (Collection.id.in_(shared))).order_by(Collection.id)))
