#!/usr/bin/env python3
"""PostgreSQL 用户认证：建表、种子管理员、校验账号密码。"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from werkzeug.security import check_password_hash, generate_password_hash

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(64) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


class AuthDBError(Exception):
    """认证库不可用或操作失败。"""


class AuthDB:
    def __init__(self, config=None):
        self._dsn = ""
        self._default_user = "admin"
        self._default_password = "admin"
        if config is not None:
            self._dsn = (getattr(config, "DATABASE_URL", "") or "").strip()
            self._default_user = (
                getattr(config, "AUTH_DEFAULT_ADMIN_USER", "admin") or "admin"
            ).strip() or "admin"
            self._default_password = getattr(config, "AUTH_DEFAULT_ADMIN_PASSWORD", "admin") or "admin"

    @property
    def dsn(self) -> str:
        return self._dsn

    def _connect(self):
        if not self._dsn:
            raise AuthDBError("DATABASE_URL 未配置")
        try:
            import psycopg2
        except ImportError as e:
            raise AuthDBError("缺少 psycopg2-binary 依赖") from e
        try:
            return psycopg2.connect(self._dsn)
        except Exception as e:
            raise AuthDBError(f"无法连接 PostgreSQL：{type(e).__name__}") from e

    def ensure_ready(self) -> None:
        """建表；若无用户则写入默认 admin。"""
        conn = self._connect()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(_SCHEMA_SQL)
                    cur.execute("SELECT COUNT(*) FROM users")
                    count = int(cur.fetchone()[0])
                    if count == 0:
                        cur.execute(
                            "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                            (
                                self._default_user,
                                generate_password_hash(self._default_password),
                            ),
                        )
                        logger.info("已种子默认管理员用户: %s", self._default_user)
        finally:
            conn.close()

    def verify_user(self, username: str, password: str) -> Tuple[bool, Optional[str]]:
        """
        校验账号密码。
        返回 (ok, error_code)：
          ok=True 时 error_code=None；
          ok=False 时 error_code 为 invalid / unavailable。
        """
        user = (username or "").strip()
        pwd = password or ""
        if not user or not pwd:
            return False, "invalid"
        try:
            self.ensure_ready()
            conn = self._connect()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT password_hash FROM users WHERE username = %s",
                        (user,),
                    )
                    row = cur.fetchone()
            finally:
                conn.close()
        except AuthDBError:
            logger.exception("认证库不可用")
            return False, "unavailable"
        except Exception:
            logger.exception("认证查询失败")
            return False, "unavailable"

        if not row:
            return False, "invalid"
        if not check_password_hash(row[0], pwd):
            return False, "invalid"
        return True, None

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        user = (username or "").strip()
        if not user:
            return None
        try:
            self.ensure_ready()
            conn = self._connect()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, username, created_at FROM users WHERE username = %s",
                        (user,),
                    )
                    row = cur.fetchone()
            finally:
                conn.close()
        except Exception:
            logger.exception("查询用户失败")
            return None
        if not row:
            return None
        return {"id": row[0], "username": row[1], "created_at": str(row[2])}
