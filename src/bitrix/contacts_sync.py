from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import requests
import clickhouse_connect
from requests.exceptions import RequestException
from dateutil import parser

# -------------------- CONFIG (ENV) --------------------
BITRIX_BASE_URL = os.getenv("BITRIX_BASE_URL", "").strip()  # e.g. https://your.bitrix24.ru/rest/<id>/<token>/
CONTACTS_ENDPOINT = os.getenv("BITRIX_CONTACTS_ENDPOINT", "crm.contact.list.json").strip()
CONTACTS_URL = f"{BITRIX_BASE_URL.rstrip('/')}/{CONTACTS_ENDPOINT.lstrip('/')}"

CH_HOST = os.getenv("CH_HOST", "localhost").strip()
CH_PORT = int(os.getenv("CH_PORT", "8123"))
CH_USER = os.getenv("CH_USER", "default").strip()
CH_PASSWORD = os.getenv("CH_PASSWORD", "").strip()
CH_DATABASE = os.getenv("CH_DATABASE", "default").strip()

TABLE_NAME = os.getenv("CH_BITRIX_CONTACTS_TABLE", "bitrix_contacts").strip()

HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "30"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
BACKOFF_BASE_SEC = float(os.getenv("BACKOFF_BASE_SEC", "1.5"))

PAGE_SLEEP_SEC = float(os.getenv("PAGE_SLEEP_SEC", "0.1"))

# Bitrix paging is usually 50 by default; keep it explicit
BATCH_SIZE = int(os.getenv("BITRIX_BATCH_SIZE", "50"))

# -------------------- COLUMNS --------------------
COLUMNS: List[str] = [
    "ID", "POST", "COMMENTS", "HONORIFIC", "NAME", "SECOND_NAME", "LAST_NAME",
    "LEAD_ID", "TYPE_ID", "SOURCE_ID", "SOURCE_DESCRIPTION", "COMPANY_ID",
    "BIRTHDATE", "EXPORT", "HAS_PHONE", "HAS_EMAIL", "HAS_IMOL",
    "DATE_CREATE", "DATE_MODIFY", "ASSIGNED_BY_ID", "CREATED_BY_ID",
    "MODIFY_BY_ID", "OPENED", "ORIGINATOR_ID", "ORIGIN_ID", "ORIGIN_VERSION",
    "ADDRESS", "ADDRESS_2", "ADDRESS_CITY", "ADDRESS_POSTAL_CODE",
    "ADDRESS_REGION", "ADDRESS_PROVINCE", "ADDRESS_COUNTRY", "ADDRESS_LOC_ADDR_ID",
    "UTM_SOURCE", "UTM_MEDIUM", "UTM_CAMPAIGN", "UTM_CONTENT", "UTM_TERM",
    "PARENT_ID_1032", "LAST_COMMUNICATION_TIME", "LAST_ACTIVITY_BY", "LAST_ACTIVITY_TIME",
]

INT_COLUMNS = [
    "ID", "LEAD_ID", "COMPANY_ID",
    "ASSIGNED_BY_ID", "CREATED_BY_ID", "MODIFY_BY_ID",
    "LAST_ACTIVITY_BY",
]

DATETIME_COLUMNS = [
    "DATE_CREATE", "DATE_MODIFY", "BIRTHDATE", "LAST_ACTIVITY_TIME", "LAST_COMMUNICATION_TIME"
]

# -------------------- HELPERS --------------------
def require_env(value: str, name: str) -> None:
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")

def ch_client():
    return clickhouse_connect.get_client(
        host=CH_HOST,
        port=CH_PORT,
        username=CH_USER,
        password=CH_PASSWORD,
        database=CH_DATABASE,
    )

def http_get_json_with_retries(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except RequestException as e:
            if attempt == MAX_RETRIES:
                raise
            sleep_s = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
            print(f"[HTTP] Error: {e}. Attempt {attempt}/{MAX_RETRIES}. Sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)
    return {}

def parse_dt(v: Any) -> Optional[pd.Timestamp]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    if s in ("", "NaT", "None"):
        return None
    try:
        # Bitrix often returns ISO8601 with TZ. Keep timezone-aware then convert to naive if needed.
        return pd.Timestamp(parser.isoparse(s))
    except Exception:
        return None

def safe_str(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (list, tuple, np.ndarray)):
        return ",".join(map(str, v))
    return str(v)

# -------------------- DDL --------------------
def ensure_table(client) -> None:
    client.command(f"CREATE DATABASE IF NOT EXISTS `{CH_DATABASE}`")

    # Store datetimes as DateTime64 for BI friendliness; keep UPDATED_AT as version for ReplacingMergeTree.
    client.command(f"""
    CREATE TABLE IF NOT EXISTS `{CH_DATABASE}`.`{TABLE_NAME}` (
        ID UInt64,
        POST String,
        COMMENTS String,
        HONORIFIC String,
        NAME String,
        SECOND_NAME String,
        LAST_NAME String,
        LEAD_ID UInt64,
        TYPE_ID String,
        SOURCE_ID String,
        SOURCE_DESCRIPTION String,
        COMPANY_ID UInt64,

        BIRTHDATE DateTime64(0),
        EXPORT String,
        HAS_PHONE String,
        HAS_EMAIL String,
        HAS_IMOL String,

        DATE_CREATE DateTime64(0),
        DATE_MODIFY DateTime64(0),

        ASSIGNED_BY_ID UInt64,
        CREATED_BY_ID UInt64,
        MODIFY_BY_ID UInt64,
        OPENED String,
        ORIGINATOR_ID String,
        ORIGIN_ID String,
        ORIGIN_VERSION String,

        ADDRESS String,
        ADDRESS_2 String,
        ADDRESS_CITY String,
        ADDRESS_POSTAL_CODE String,
        ADDRESS_REGION String,
        ADDRESS_PROVINCE String,
        ADDRESS_COUNTRY String,
        ADDRESS_LOC_ADDR_ID String,

        UTM_SOURCE String,
        UTM_MEDIUM String,
        UTM_CAMPAIGN String,
        UTM_CONTENT String,
        UTM_TERM String,

        PARENT_ID_1032 String,
        LAST_COMMUNICATION_TIME DateTime64(0),
        LAST_ACTIVITY_BY UInt64,
        LAST_ACTIVITY_TIME DateTime64(0),

        IS_DELETED UInt8 DEFAULT 0,
        UPDATED_AT DateTime DEFAULT now()
    )
    ENGINE = ReplacingMergeTree(UPDATED_AT)
    ORDER BY ID
    """)  # noqa: E501

def get_watermark_date_modify(client) -> Optional[pd.Timestamp]:
    # watermark should follow business change timestamp, not ingestion time
    rows = client.query(
        f"SELECT max(DATE_MODIFY) FROM `{CH_DATABASE}`.`{TABLE_NAME}`"
    ).result_rows
    if rows and rows[0] and rows[0][0]:
        return pd.Timestamp(rows[0][0])
    return None

# -------------------- EXTRACT --------------------
def extract_contacts(since: Optional[pd.Timestamp]) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    start = 0

    print("[Bitrix] Extracting contacts...")

    while True:
        params: Dict[str, Any] = {
            "start": start,
            "select[]": COLUMNS,
        }
        if since is not None:
            # Bitrix expects ISO without milliseconds
            params["filter[>DATE_MODIFY]"] = since.strftime("%Y-%m-%dT%H:%M:%S")

        data = http_get_json_with_retries(CONTACTS_URL, params=params)
        chunk = data.get("result", []) or []
        next_start = data.get("next", 0)

        if not chunk:
            break

        all_rows.extend(chunk)
        print(f"[Bitrix] +{len(chunk)} (total: {len(all_rows)})")

        if not next_start:
            break

        start = next_start
        time.sleep(PAGE_SLEEP_SEC)

    return all_rows

# -------------------- TRANSFORM --------------------
def transform(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)

    # Ensure all expected columns exist
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Datetimes
    for col in DATETIME_COLUMNS:
        df[col] = df[col].apply(parse_dt)

    # Ints
    for col in INT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("uint64")

    # Strings (everything else)
    for col in df.columns:
        if col in DATETIME_COLUMNS or col in INT_COLUMNS:
            continue
        df[col] = df[col].apply(safe_str)

    df["IS_DELETED"] = 0
    df["UPDATED_AT"] = pd.Timestamp.now()

    # Keep only table columns in correct order
    ordered_cols = (
        COLUMNS
        + ["IS_DELETED", "UPDATED_AT"]
    )
    return df[ordered_cols]

# -------------------- LOAD --------------------
def load(client, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    client.insert_df(f"{CH_DATABASE}.{TABLE_NAME}", df)
    return len(df)

def main():
    require_env(BITRIX_BASE_URL, "BITRIX_BASE_URL")

    client = ch_client()
    ensure_table(client)

    watermark = get_watermark_date_modify(client)
    if watermark is None:
        print("[CH] No watermark found. First load (full history returned by API).")
    else:
        print(f"[CH] Watermark (max DATE_MODIFY): {watermark}")

    rows = extract_contacts(watermark)
    if not rows:
        print("[Bitrix] No new/updated contacts. Nothing to load.")
        return

    df = transform(rows)
    inserted = load(client, df)
    print(f"[CH] Inserted/updated rows: {inserted}")

if __name__ == "__main__":
    main()
