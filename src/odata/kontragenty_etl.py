from __future__ import annotations

import os
import re
import time
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import requests
import clickhouse_connect
from requests.auth import HTTPBasicAuth
from requests.exceptions import RequestException

# CONFIG 

# OData
ODATA_BASE = os.getenv("ODATA_BASE_URL", "").strip()
ODATA_USER = os.getenv("ODATA_USER", "").strip()
ODATA_PASSWORD = os.getenv("ODATA_PASSWORD", "").strip()

# ClickHouse
CH_HOST = os.getenv("CH_HOST", "localhost").strip()
CH_PORT = int(os.getenv("CH_PORT", "8123"))
CH_USER = os.getenv("CH_USER", "default").strip()
CH_PASSWORD = os.getenv("CH_PASSWORD", "").strip()
CH_DATABASE = os.getenv("CH_DATABASE", "default").strip()

# Tables
MAIN_TABLE = os.getenv("CH_MAIN_TABLE", "Kontragenty").strip()
CONTACT_TABLE = os.getenv("CH_CONTACT_TABLE", "Kontragenty_Contact").strip()

# Runtime / tuning
ODATA_PAGE_SIZE = int(os.getenv("ODATA_PAGE_SIZE", "1000"))
INSERT_BATCH = int(os.getenv("INSERT_BATCH", "5000"))
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "30"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))

# VALIDATION

def require_env(value: str, name: str) -> None:
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")

def get_auth() -> Optional[HTTPBasicAuth]:
    # If OData does not require auth, you can omit credentials.
    if ODATA_USER and ODATA_PASSWORD:
        return HTTPBasicAuth(ODATA_USER, ODATA_PASSWORD)
    return None

def ch_get_client():
    return clickhouse_connect.get_client(
        host=CH_HOST,
        port=CH_PORT,
        username=CH_USER,
        password=CH_PASSWORD,
        database=CH_DATABASE,
    )

# HTTP LAYER

def http_get_with_retries(
    url: str,
    auth: Optional[HTTPBasicAuth] = None,
    headers: Optional[Dict[str, str]] = None,
    retries: int = MAX_RETRIES,
) -> Dict[str, Any]:
    """GET with retries + exponential backoff."""
    headers = headers or {"Accept": "application/json"}

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, auth=auth, headers=headers, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except RequestException as e:
            print(f"[HTTP] Error: {e}. Attempt {attempt}/{retries}")
            if attempt == retries:
                raise
            wait = 2 ** attempt
            time.sleep(wait)

# ODATA EXTRACTION

def load_all_odata(base_url: str, auth: Optional[HTTPBasicAuth] = None) -> List[Dict[str, Any]]:
    """
    Load all records from OData using paging ($top/$skip) with deterministic ordering by Ref_Key.
    Also performs a basic DQ-check: uniqueness of Ref_Key across pages.
    """
    results: List[Dict[str, Any]] = []
    seen_ref_keys = set()

    skip = 0
    page = 1

    while True:
        url = (
            f"{base_url}"
            f"?$format=json&$top={ODATA_PAGE_SIZE}&$skip={skip}&$orderby=Ref_Key"
        )

        data = http_get_with_retries(url, auth=auth)
        chunk = data.get("value", [])
        if not chunk:
            break

        chunk_ref_keys = [r.get("Ref_Key") for r in chunk]
        duplicate_in_chunk = [k for k in chunk_ref_keys if k in seen_ref_keys and k is not None]
        if duplicate_in_chunk:
            print(f"[DQ] Duplicate Ref_Key detected on page {page}: {duplicate_in_chunk[:10]}...")

        seen_ref_keys.update([k for k in chunk_ref_keys if k is not None])
        results.extend(chunk)

        last_keys = chunk_ref_keys[-5:]
        print(
            f"Page {page}: {len(chunk)} rows, total: {len(results)}, "
            f"last Ref_Key: {last_keys}, unique Ref_Key: {len(seen_ref_keys)}"
        )

        if len(chunk) < ODATA_PAGE_SIZE:
            break

        skip += ODATA_PAGE_SIZE
        page += 1

    print(f"Done. Total rows: {len(results)}, unique Ref_Key: {len(seen_ref_keys)}")
    return results

# CLICKHOUSE DDL

def ensure_ch_tables(client) -> None:
    """Create DB and target tables if they don't exist."""
    client.command(f"CREATE DATABASE IF NOT EXISTS `{CH_DATABASE}`")

    # NOTE: For production-grade determinism with ReplacingMergeTree, consider adding a version column (UPDATED_AT).
    client.command(f"""
    CREATE TABLE IF NOT EXISTS `{CH_DATABASE}`.`{MAIN_TABLE}` (
        `Ref_Key` String,
        `Code` String,
        `Description` String,
        `ДатаРегистрации` String,
        `ДатаРожденияКлиента` String,
        `Имя` String,
        `Отчество` String,
        `Фамилия` String,
        `ПолКлиента` String,
        `ТипЗубнойФормулы` String,
        `ТипКонтрагента` String,
        `КаналПривлечения_Key` String,
        `Комментарий` String,
        `Куратор_Key` String,
        `СтруктурнаяЕдиница_Key` String,
        `СотрудникРегистрации_Key` String,
        `НаименованиеПолное` String,
        `Телефон` String,
        `ЭтоНеСтраховаяКомпания` UInt8
    )
    ENGINE = ReplacingMergeTree()
    ORDER BY Ref_Key
    """)

    client.command(f"""
    CREATE TABLE IF NOT EXISTS `{CH_DATABASE}`.`{CONTACT_TABLE}` (
        `Ref_Key` String,
        `Тип` String,
        `Вид_Key` String,
        `НомерТелефона` String,
        `НомерТелефонаБезКодаСтраны` String,
        `НомерТелефонаБезКодов` String,
        `НомерТелефонаПоследниеЦифры` String
    )
    ENGINE = ReplacingMergeTree()
    ORDER BY Ref_Key
    """)

def truncate_tables(client) -> None:
    print("Truncating target tables before load...")
    client.command(f"TRUNCATE TABLE `{CH_DATABASE}`.`{MAIN_TABLE}`")
    client.command(f"TRUNCATE TABLE `{CH_DATABASE}`.`{CONTACT_TABLE}`")

# TRANSFORMS

def normalize_phone(phone: str) -> str:
    """Normalize phone number to 7XXXXXXXXXX where possible."""
    if not phone:
        return ""
    digits = re.sub(r"\D", "", str(phone))
    if digits.startswith("8") and len(digits) >= 11:
        digits = "7" + digits[1:]
    elif digits.startswith("9") and len(digits) == 10:
        digits = "7" + digits
    return digits

def prepare_rows(records: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Transform OData JSON into two dataframes (main + contacts)."""
    if not records:
        return pd.DataFrame(), pd.DataFrame()

    main_rows: List[Dict[str, Any]] = []
    contact_rows: List[Dict[str, Any]] = []

    def parse_dt(val) -> str:
        if val and val != "0001-01-01T00:00:00":
            try:
                return pd.to_datetime(val).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                return ""
        return ""

    for item in records:
        main_rows.append({
            "Ref_Key": item.get("Ref_Key") or "",
            "Code": item.get("Code") or "",
            "Description": item.get("Description") or "",
            "ДатаРегистрации": parse_dt(item.get("ДатаРегистрации")),
            "ДатаРожденияКлиента": parse_dt(item.get("ДатаРожденияКлиента")),
            "Имя": item.get("Имя") or "",
            "Отчество": item.get("Отчество") or "",
            "Фамилия": item.get("Фамилия") or "",
            "ПолКлиента": item.get("ПолКлиента") or "",
            "ТипЗубнойФормулы": item.get("ТипЗубнойФормулы") or "",
            "ТипКонтрагента": item.get("ТипКонтрагента") or "",
            "КаналПривлечения_Key": item.get("КаналПривлечения_Key") or "",
            "Комментарий": item.get("Комментарий") or "",
            "Куратор_Key": item.get("Куратор_Key") or "",
            "СтруктурнаяЕдиница_Key": item.get("СтруктурнаяЕдиница_Key") or "",
            "СотрудникРегистрации_Key": item.get("СотрудникРегистрации_Key") or "",
            "НаименованиеПолное": item.get("НаименованиеПолное") or "",
            "Телефон": normalize_phone(item.get("Телефон") or ""),
            "ЭтоНеСтраховаяКомпания": 1 if item.get("ЭтоНеСтраховаяКомпания") else 0,
        })

        for contact in item.get("КонтактнаяИнформация") or []:
            phone_raw = contact.get("НомерТелефона") or ""
            contact_rows.append({
                "Ref_Key": contact.get("Ref_Key") or "",
                "Тип": contact.get("Тип") or "",
                "Вид_Key": contact.get("Вид_Key") or "",
                "НомерТелефона": normalize_phone(phone_raw),
                "НомерТелефонаБезКодаСтраны": contact.get("НомерТелефонаБезКодаСтраны") or "",
                "НомерТелефонаБезКодов": contact.get("НомерТелефонаБезКодов") or "",
                "НомерТелефонаПоследниеЦифры": contact.get("НомерТелефонаПоследниеЦифры") or "",
            })

    df_main = pd.DataFrame(main_rows)
    df_contact = pd.DataFrame(contact_rows)

    if not df_main.empty:
        df_main["ЭтоНеСтраховаяКомпания"] = df_main["ЭтоНеСтраховаяКомпания"].astype("uint8")

    return df_main, df_contact

# LOAD

def insert_in_batches(client, table_name: str, df: pd.DataFrame, batch_size: int = INSERT_BATCH) -> None:
    if df.empty:
        print(f"No rows to insert into {table_name}.")
        return

    total = len(df)
    start = 0

    while start < total:
        end = min(start + batch_size, total)
        chunk = df.iloc[start:end].copy()
        client.insert_df(f"{CH_DATABASE}.{table_name}", chunk)

        start = end
        percent = int(start / total * 100)
        print(f"\rInsert {table_name}: {percent}% ({start}/{total})", end="", flush=True)

    print(f"\rInsert {table_name}: 100% ({total}/{total})")

# ENTRYPOINT

def main():
    require_env(ODATA_BASE, "ODATA_BASE_URL")

    auth = get_auth()
    client = ch_get_client()

    print("Ensuring ClickHouse tables...")
    ensure_ch_tables(client)

    # Full refresh pattern (by design, see TECHNICAL_NOTES)
    truncate_tables(client)

    print("Loading data from OData...")
    raw = load_all_odata(ODATA_BASE, auth=auth)
    if not raw:
        print("No data returned from OData.")
        return

    df_main, df_contact = prepare_rows(raw)
    print(f"Prepared main rows: {len(df_main)}")
    print(f"Prepared contact rows: {len(df_contact)}")

    insert_in_batches(client, MAIN_TABLE, df_main)
    insert_in_batches(client, CONTACT_TABLE, df_contact)

    print("ETL completed successfully.")

if __name__ == "__main__":
    main()
