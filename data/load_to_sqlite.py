"""CSV dosyalarını SQLite veritabanına yükler ve temel indeksleri oluşturur."""
import sqlite3
import pandas as pd

DB_PATH = "/home/claude/finnova-ai-agent/data/finnova_bank.db"
DATA_DIR = "/home/claude/finnova-ai-agent/data"

conn = sqlite3.connect(DB_PATH)

tables = {
    "branches": f"{DATA_DIR}/branches.csv",
    "customers": f"{DATA_DIR}/customers.csv",
    "accounts": f"{DATA_DIR}/accounts.csv",
    "transactions": f"{DATA_DIR}/transactions.csv",
}

for table_name, csv_path in tables.items():
    df = pd.read_csv(csv_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    print(f"{table_name}: {len(df):,} satır yüklendi")

# Text-to-SQL agent'ın hız kaybetmemesi için temel indeksler
cur = conn.cursor()
cur.execute("CREATE INDEX IF NOT EXISTS idx_tx_account ON transactions(account_id)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_tx_date ON transactions(transaction_date)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_tx_type ON transactions(transaction_type)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_acc_customer ON accounts(customer_id)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_acc_branch ON accounts(branch_id)")
conn.commit()

print("\nİndeksler oluşturuldu.")
print(f"Veritabanı hazır: {DB_PATH}")
conn.close()
