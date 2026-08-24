"""
Veritabanı bağlantı ve şema yardımcı fonksiyonları.

Agent'ın doğru SQL üretebilmesi için LLM'e her seferinde tablo şemasını
(kolon adları + tipleri + kısa açıklama) prompt içinde vermemiz gerekir.
Bu modül şemayı otomatik çıkarır, böylece veri seti değişse bile agent
kodu değişmeden çalışmaya devam eder.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "finnova_bank.db"

# Kolon bazlı iş açıklamaları (LLM'in doğru SQL üretmesine yardımcı olur)
COLUMN_NOTES = {
    "transactions.amount": "Pozitif = hesaba giren para, negatif = hesaptan çıkan para",
    "transactions.is_flagged": "1 = şüpheli/incelemeye alınmış işlem, 0 = normal",
    "accounts.balance": "Güncel hesap bakiyesi (kredi kartı/kredi hesaplarında negatif olabilir)",
    "customers.credit_score": "300-900 arası kredi skoru, düşük değer yüksek risk demektir",
}


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_schema_description() -> str:
    """Tüm tabloların kolon/tip bilgisini ve iş notlarını tek bir metin
    olarak döner. Bu metin, text-to-SQL prompt'una doğrudan enjekte edilir."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r["name"] for r in cur.fetchall()]

    lines = []
    for t in tables:
        cur.execute(f"PRAGMA table_info({t})")
        cols = cur.fetchall()
        col_lines = []
        for c in cols:
            note = COLUMN_NOTES.get(f"{t}.{c['name']}")
            suffix = f"  -- {note}" if note else ""
            col_lines.append(f"    {c['name']} ({c['type']}){suffix}")
        lines.append(f"TABLE {t}:\n" + "\n".join(col_lines))
    conn.close()
    return "\n\n".join(lines)


if __name__ == "__main__":
    print(get_schema_description())
