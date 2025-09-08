# SIH-ML-Project

**Integrated Platform for Crowdsourced Ocean Hazard Reporting & Social Media Analytics**

## What this repo contains
- `ml_service/` — FastAPI microservice for text/image scoring (`/ml/score`)
- `alert_service/` — FastAPI alert engine using PostGIS (`/alerts/send`)
- `db/init.sql` — PostGIS schema
- `docker-compose.yml` — run everything locally via Docker

## Quick start (local)
1. Create `.env` from `.env.example` and fill secrets.
2. Run:
   ```bash
   docker compose up --build
