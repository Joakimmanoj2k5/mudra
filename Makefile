PY ?= python3

.PHONY: bootstrap preflight ci test run api migrate backup restore package release security loadtest monitor

bootstrap:
	bash scripts/bootstrap.sh

preflight:
	$(PY) scripts/preflight.py

ci:
	$(PY) scripts/ci_checks.py

test:
	$(PY) -m pytest tests/unit tests/integration -q

run:
	$(PY) mudra_app.py

api:
	$(PY) -m backend.run_api

migrate:
	$(PY) scripts/migrate.py

backup:
	$(PY) scripts/backup_db.py

restore:
	$(PY) scripts/restore_db.py --backup $(BACKUP)

package:
	bash scripts/package_desktop.sh

release:
	bash scripts/build_release_bundle.sh $(TAG)

security:
	$(PY) scripts/security_audit.py

loadtest:
	$(PY) scripts/load_test_api.py

monitor:
	$(PY) scripts/monitor_health.py --iterations 12
