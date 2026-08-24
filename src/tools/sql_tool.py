"""
Güvenli SQL çalıştırma aracı.

Agent LLM tarafından üretilen SQL'i doğrudan çalıştırmak risklidir
(yanlışlıkla ya da kötü niyetli bir prompt ile DROP/DELETE/UPDATE
üretilebilir). Bu modül SADECE SELECT sorgularına izin verir ve
sonucu satır sayısı sınırlı şekilde (varsayılan 200 satır) döner.
"""
import re
import pandas as pd

from src.db import get_connection

FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|PRAGMA|REPLACE|TRUNCATE)\b",
    re.IGNORECASE,
)


class UnsafeQueryError(Exception):
    pass


def run_sql(query: str, max_rows: int = 200) -> pd.DataFrame:
    """LLM'in ürettiği SQL sorgusunu doğrulayıp çalıştırır.

    Args:
        query: Çalıştırılacak SQL (yalnızca SELECT olmalı)
        max_rows: Dönecek maksimum satır sayısı (agent'ın context'ini
                  şişirmemek için)

    Raises:
        UnsafeQueryError: Sorgu SELECT dışı bir komut içeriyorsa
    """
    cleaned = query.strip().rstrip(";")

    if not cleaned.lower().startswith("select"):
        raise UnsafeQueryError("Yalnızca SELECT sorgularına izin verilir.")
    if FORBIDDEN_KEYWORDS.search(cleaned):
        raise UnsafeQueryError("Sorgu izin verilmeyen bir anahtar kelime içeriyor.")
    if ";" in cleaned:
        raise UnsafeQueryError("Çoklu ifade (;) içeren sorgulara izin verilmiyor.")

    conn = get_connection()
    try:
        df = pd.read_sql_query(cleaned, conn)
    finally:
        conn.close()

    truncated = len(df) > max_rows
    return df.head(max_rows), truncated


if __name__ == "__main__":
    df, trunc = run_sql("SELECT city, COUNT(*) AS n FROM branches GROUP BY city")
    print(df)
    print("Kesildi mi:", trunc)
