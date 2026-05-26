# Variables por defecto (por si te olvidás de pasarlas en la terminal)
COIN ?= bitcoin
DATE ?= $(shell date +%Y-%m-%d)
END_DATE ?=
DB_FLAG ?= --db

.PHONY: up down logs shell fetch fetch-range drop-recovery monthly-avg train

# --- CONTROL DEL ENTORNO ---

up:
	docker compose up -d

down:
	docker compose down

down-v:
	docker compose down -v

logs:
	docker compose logs -f app

shell:
	docker compose exec -it app bash

# --- COMANDOS DE EXTRACCIÓN Y ANALÍTICA ---

fetch:
	docker compose exec app uv run cli.py fetch $(COIN) $(DATE) $(DB_FLAG)

fetch-range:
	@if [ -z "$(END_DATE)" ]; then \
		echo "ERROR: Debés especificar END_DATE. Ejemplo: make fetch-range END_DATE=2025-07-26"; \
		exit 1; \
	fi
	docker compose exec app uv run cli.py fetch $(COIN) $(DATE) --end-date $(END_DATE) $(DB_FLAG) --missing-only

drop-recovery:
	docker compose exec app uv run cli.py drop-recovery

monthly-avg:
	docker compose exec app uv run cli.py monthly-avg

forecast:
	docker compose exec app uv run analytics/forecasting.py