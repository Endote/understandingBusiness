#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path


DB_NAME = "paperrush"


def sql_path(path: Path) -> str:
    return path.resolve().as_posix()


def sql_literal(text: str) -> str:
    return text.replace("\\", "/").replace("'", "''")


def log(message: str, handle) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line)
    handle.write(line + "\n")
    handle.flush()


def run_command(cmd: list[str], log_handle, env: dict[str, str] | None = None) -> None:
    log(f"RUN: {' '.join(cmd)}", log_handle)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        log_handle.write(line)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def query_scalar(cmd: list[str], env: dict[str, str] | None = None) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
    return result.stdout.strip()


def require_command(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(
            f"Required command '{name}' was not found in PATH. "
            "Install PostgreSQL client tools and make sure they are available in PATH."
        )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset and load the PaperRush PostgreSQL database.")
    parser.add_argument(
        "--maintenance-db",
        default=os.environ.get("MAINTENANCE_DB", "postgres"),
        help="Maintenance database used for create/drop operations. Default: postgres",
    )
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    sql_file = root_dir / "sql" / "setup_paperrush.sql"
    log_dir = root_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"setup_paperrush_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    required_files = [
        root_dir / "input" / "FACT_TABLE.csv",
        root_dir / "input" / "DIM_PRODUCT.csv",
        root_dir / "input" / "DIM_STORE.csv",
        root_dir / "input" / "DEMOGRAPHICS.csv",
        root_dir / "input" / "EMBEDDINGS.csv",
        root_dir / "input" / "Printing Schedule (On Sale July 2024)" / "Printing Schedule.csv",
        root_dir / "input" / "Printing Schedule (On Sale July 2024)" / "DIM_PRODUCT.csv",
        root_dir / "input" / "Printing Schedule (On Sale July 2024)" / "DIM_STORE.csv",
        root_dir / "input" / "Printing Schedule (On Sale July 2024)" / "DEMOGRAPHICS.csv",
        root_dir / "input" / "Printing Schedule (On Sale July 2024)" / "EMBEDDINGS.csv",
        sql_file,
    ]

    psql = require_command("psql")
    createdb = require_command("createdb")
    dropdb = require_command("dropdb")

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    env = os.environ.copy()
    database_created = False

    with log_file.open("w", encoding="utf-8") as log_handle:
        try:
            log("Starting PaperRush database setup", log_handle)
            log(f"Root directory: {root_dir}", log_handle)
            log(f"Log file: {log_file}", log_handle)
            log(f"Target database: {DB_NAME}", log_handle)
            log(f"Maintenance database: {args.maintenance_db}", log_handle)

            vector_available = query_scalar(
                [psql, "-d", args.maintenance_db, "-Atqc", "select count(*) from pg_available_extensions where name = 'vector'"],
                env=env,
            )
            if vector_available != "1":
                raise RuntimeError(
                    "pgvector is not installed on the PostgreSQL server. "
                    "Install the extension first, then rerun the setup. "
                    "Typical macOS/Homebrew command: brew install pgvector"
                )

            db_exists = query_scalar(
                [psql, "-d", args.maintenance_db, "-Atqc", f"select 1 from pg_database where datname = '{DB_NAME}'"],
                env=env,
            )
            if db_exists == "1":
                log(f"Database {DB_NAME} exists; dropping for ground-zero rebuild", log_handle)
                run_command([dropdb, DB_NAME], log_handle, env=env)

            log(f"Creating database {DB_NAME}", log_handle)
            run_command([createdb, DB_NAME], log_handle, env=env)
            database_created = True

            template_sql = sql_file.read_text(encoding="utf-8")
            replacements = {
                "__TRAIN_FACT_CSV__": sql_literal(sql_path(root_dir / "input" / "FACT_TABLE.csv")),
                "__TRAIN_PRODUCT_CSV__": sql_literal(sql_path(root_dir / "input" / "DIM_PRODUCT.csv")),
                "__TRAIN_STORE_CSV__": sql_literal(sql_path(root_dir / "input" / "DIM_STORE.csv")),
                "__TRAIN_DEMOGRAPHIC_CSV__": sql_literal(sql_path(root_dir / "input" / "DEMOGRAPHICS.csv")),
                "__TRAIN_EMBEDDING_CSV__": sql_literal(sql_path(root_dir / "input" / "EMBEDDINGS.csv")),
                "__HOLDOUT_SCHEDULE_CSV__": sql_literal(sql_path(root_dir / "input" / "Printing Schedule (On Sale July 2024)" / "Printing Schedule.csv")),
                "__HOLDOUT_PRODUCT_CSV__": sql_literal(sql_path(root_dir / "input" / "Printing Schedule (On Sale July 2024)" / "DIM_PRODUCT.csv")),
                "__HOLDOUT_STORE_CSV__": sql_literal(sql_path(root_dir / "input" / "Printing Schedule (On Sale July 2024)" / "DIM_STORE.csv")),
                "__HOLDOUT_DEMOGRAPHIC_CSV__": sql_literal(sql_path(root_dir / "input" / "Printing Schedule (On Sale July 2024)" / "DEMOGRAPHICS.csv")),
                "__HOLDOUT_EMBEDDING_CSV__": sql_literal(sql_path(root_dir / "input" / "Printing Schedule (On Sale July 2024)" / "EMBEDDINGS.csv")),
            }
            rendered_sql = template_sql
            for placeholder, value in replacements.items():
                rendered_sql = rendered_sql.replace(placeholder, value)

            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".sql", delete=False) as tmp_sql:
                tmp_sql.write(rendered_sql)
                rendered_sql_path = Path(tmp_sql.name)

            cmd = [
                psql,
                "-d",
                DB_NAME,
                "-v",
                "ON_ERROR_STOP=1",
                "-f",
                str(rendered_sql_path),
            ]
            try:
                run_command(cmd, log_handle, env=env)
            finally:
                if rendered_sql_path.exists():
                    rendered_sql_path.unlink()

            log("PaperRush database setup finished successfully", log_handle)
            log(f"Trace log written to {log_file}", log_handle)
        except Exception as exc:
            log(f"ERROR: {exc}", log_handle)
            if database_created:
                log(f"Dropping database {DB_NAME} after failed setup", log_handle)
                try:
                    run_command([dropdb, DB_NAME], log_handle, env=env)
                except Exception as drop_exc:  # noqa: BLE001
                    log(f"ERROR while dropping failed database: {drop_exc}", log_handle)
            raise

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
