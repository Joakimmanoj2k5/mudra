# MUDRA Launch Checklist (Phase 8)

## 1. Security Hardening
- Set strong `MUDRA_SECRET_KEY` (>=24 chars, not default)
- Set `MUDRA_DB_PATH`, `MUDRA_API_HOST`, `MUDRA_API_PORT`
- Run: `python scripts/security_audit.py`

## 2. Pre-Release Validation
- Run: `make migrate`
- Run: `make backup`
- Run: `python scripts/preflight.py`
- Run: `make ci`

## 3. Performance Gate
- Start API: `python -m backend.run_api`
- Run load test: `python scripts/load_test_api.py --requests 300 --workers 25`
- Pass criteria: success_rate >= 0.98, p95 < 250ms (local baseline)

## 4. UAT Sign-off
- Login flow (demo/admin)
- Practice mode realtime prediction
- Quiz scoring and attempt logging
- Analytics table + confusion matrix rendering
- Admin model activate/register/rollback + predictor reload

## 5. Deployment
- Docker: `docker compose -f deploy/docker-compose.yml up -d --build`
- Or systemd using `deploy/systemd/mudra-api.service`
- Verify `GET /health` and `GET /ready`

## 6. Post-Deploy Monitoring
- Run: `python scripts/monitor_health.py --iterations 12 --interval 5`
- Check logs and alert if >=3 consecutive failures

## 7. Rollback Plan
- Rollback model family from Admin UI or API
- Restore DB if needed:
  - `python scripts/restore_db.py --backup <backup_file>`
