#!/usr/bin/env python3
"""认证库单元测试（不依赖真实 Postgres）。"""

import unittest
from unittest.mock import MagicMock, patch

from werkzeug.security import generate_password_hash

from services.auth_db import AuthDB, AuthDBError


class _Cfg:
    DATABASE_URL = "postgresql://earthquake:earthquake@localhost:5432/earthquake_qa"
    AUTH_DEFAULT_ADMIN_USER = "admin"
    AUTH_DEFAULT_ADMIN_PASSWORD = "admin"


class TestAuthDB(unittest.TestCase):
    def test_verify_rejects_empty(self):
        db = AuthDB(_Cfg())
        ok, err = db.verify_user("", "x")
        self.assertFalse(ok)
        self.assertEqual(err, "invalid")

    def test_verify_success_with_mock(self):
        db = AuthDB(_Cfg())
        hashed = generate_password_hash("admin")
        conn = MagicMock()
        cur = MagicMock()
        cur.fetchone.return_value = (hashed,)
        conn.cursor.return_value.__enter__.return_value = cur
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)

        with patch.object(db, "ensure_ready"), patch.object(db, "_connect", return_value=conn):
            ok, err = db.verify_user("admin", "admin")
        self.assertTrue(ok)
        self.assertIsNone(err)

    def test_verify_wrong_password(self):
        db = AuthDB(_Cfg())
        hashed = generate_password_hash("admin")
        conn = MagicMock()
        cur = MagicMock()
        cur.fetchone.return_value = (hashed,)
        conn.cursor.return_value.__enter__.return_value = cur

        with patch.object(db, "ensure_ready"), patch.object(db, "_connect", return_value=conn):
            ok, err = db.verify_user("admin", "wrong")
        self.assertFalse(ok)
        self.assertEqual(err, "invalid")

    def test_unavailable_when_connect_fails(self):
        db = AuthDB(_Cfg())
        with patch.object(db, "ensure_ready", side_effect=AuthDBError("down")):
            ok, err = db.verify_user("admin", "admin")
        self.assertFalse(ok)
        self.assertEqual(err, "unavailable")


if __name__ == "__main__":
    unittest.main()
