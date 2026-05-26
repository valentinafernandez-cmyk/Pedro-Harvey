## Environment Setup

To guarantee complete reproducibility and cross-platform portability, this project containerizes both the Python runtime and the PostgreSQL database using Docker. The internal Python environment is managed seamlessly inside the container using **`uv`**.

### 1. Prerequisites
Ensure you have Docker and Docker Compose installed on your system.

### 2. Configuration
Create your local environment file from the template:

```bash
cp .env.example .env
```

Open the `.env` file and configure your credentials. Ensure `DATABASE_URL` targets the internal Docker service host (`db`):

```bash
DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/coingecko
```

### Initialization & Usage

**Start the full environment:**
Builds the Python image with necessary C-libraries, updates dependencies via `uv sync`, initializes the Postgres schema, and starts the 24/7 background scheduler daemon.

```bash
docker compose up -d
```

**Run interactive or manual commands:**
Access the container's shell to run commands

```bash
docker compose exec -it app bash
```

Once inside the container shell, run commands directly:

#### 1. Fetch single day coin history.
```bash
make fetch COIN= DATE= DB_FLAG=
```

#### 2. Fetch coin history for a range of dates. Include `--missing-only` to fetch only the missing dates in the range.
```bash
make fetch COIN= DATE= END_DATE= DB_FLAG=
```

#### 3. Run drop and recovery analysis
```bash
make drop-recovery
```

#### 4. Run monthly average analysis
```bash
make monthly-avg
```

#### 5. Training and forecasting evaluation
```bash
make forecast
```


Exit the container environment at any time:

```bash
exit
```

**Stop the environment:**
Safely stops both the scheduler and database containers while preserving your data.

docker compose down

To completely wipe the database volumes and start with a fresh schema:

```bash
docker compose down -v
```