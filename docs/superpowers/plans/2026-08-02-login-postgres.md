# Login + PostgreSQL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a login page gated in the SPA; verify credentials against PostgreSQL with default seed user `admin`/`admin`.

**Architecture:** Frontend localStorage gate; `POST /api/auth/login` checks `users` table via psycopg2; docker-compose adds Postgres; no API-wide auth middleware.

**Tech Stack:** Flask, psycopg2-binary, werkzeug password hash, Vue 3, Postgres 16 (Docker)

## Global Constraints

- Frontend gate only (API business routes remain open)
- Password stored as hash, never plaintext
- Default seed only when users table is empty
- DATABASE_URL configurable via env

---

## File map

| File | Responsibility |
|---|---|
| `docker-compose.yml` | postgres service |
| `requirements.txt` | psycopg2-binary |
| `config/config.py` | DATABASE_URL + default admin |
| `.env.example` | document DATABASE_URL |
| `services/auth_db.py` | connect, schema, seed, verify |
| `app.py` | `/api/auth/login|logout|me` |
| `tests/test_auth_db.py` | hash/verify + seed logic (mockable) |
| `frontend/src/components/Login.vue` | login UI |
| `frontend/src/api.js` | login/logout helpers |
| `frontend/src/App.vue` | gate + logout |
| `frontend/src/components/AppHeader.vue` | optional logout slot/props |
| `frontend/src/assets/main.css` | login styles |

## Task 1: Infra + auth_db

- [ ] Add postgres to docker-compose
- [ ] Add psycopg2-binary + config
- [ ] Implement `services/auth_db.py`
- [ ] Unit tests for verify/seed with test doubles or skip-if-no-db
- [ ] Wire login routes in app.py

## Task 2: Frontend gate

- [ ] Login.vue + api.js
- [ ] App.vue conditional render + logout
- [ ] Rebuild SPA (Node 24)

## Task 3: Bring up and verify

- [ ] `docker compose up -d postgres`
- [ ] Restart Flask, login with admin/admin
