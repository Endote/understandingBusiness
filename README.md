# PaperRush Database Setup

This project includes a one-shot loader that:
- drops and recreates the `paperrush` database
- loads raw train and July holdout CSV files
- validates the raw entities
- builds typed `raw`, `core`, and `holdout` schemas
- stores embeddings in `pgvector`

Every run resets the database to ground zero from the raw files.

## Requirements

- PostgreSQL server running and reachable
- PostgreSQL client tools in `PATH`:
  - `psql`
  - `createdb`
  - `dropdb`
- Python 3
- `pgvector` installed on the PostgreSQL server

macOS install for `pgvector`:

```bash
brew install pgvector
```

## Run

macOS / Linux:

```bash
python3 scripts/setup_paperrush_db.py
```

or:

```bash
./scripts/setup_paperrush_db.sh
```

Windows:

```bat
py -3 scripts\setup_paperrush_db.py
```

or:

```bat
scripts\setup_paperrush_db.bat
```

## Expected Success Output

You should see output like:

```text
[YYYY-MM-DD HH:MM:SS] Starting PaperRush database setup
[YYYY-MM-DD HH:MM:SS] Target database: paperrush
[YYYY-MM-DD HH:MM:SS] Creating database paperrush
[YYYY-MM-DD HH:MM:SS] RUN: ... psql ... -f .../sql/setup_paperrush.sql
...
PaperRush database setup finished successfully
Trace log written to .../logs/setup_paperrush_YYYYMMDD_HHMMSS.log
```

## Expected Reset Behavior

If `paperrush` already exists, the script will drop it first:

```text
[YYYY-MM-DD HH:MM:SS] Database paperrush exists; dropping for ground-zero rebuild
```

That is expected.

## Common Failure

If `pgvector` is not installed, you will see:

```text
ERROR: pgvector is not installed on the PostgreSQL server. Install the extension first, then rerun the setup.
```

Install it, then rerun the setup command.

## What Gets Created

Schemas:
- `raw`
- `core`
- `holdout`

Important notes:
- `raw` keeps file-ingested tables
- `core` contains typed train tables for modeling
- `holdout` contains typed July pilot tables
- `demographic` is provisionally deduplicated by `postal_code` using `MAX(population)`
- `core.dim_store.postal_code -> core.demographic.postal_code` is enforced
- negative `SOLDQTY` values are allowed because the source fact data contains adjustment/return-like rows; these are logged, not rejected

## Logs

Each run writes a trace log to:

```text
logs/setup_paperrush_YYYYMMDD_HHMMSS.log
```
