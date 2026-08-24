"""
FinNova Bank - Sentetik finans veri seti üretici
--------------------------------------------------
Bu script, "Otonom İş Zekası Ajanı" portfolyo projesi için gerçekçi ama
tamamen sentetik bir banka veri seti üretir. Gerçek kişi/kurum verisi
İÇERMEZ. Amaç: text-to-SQL agent'ın test edebileceği, iş sorularına
anlamlı cevaplar verebileceği bir veri tabanı kurmaktır.

Çıktılar (bu dosyayla aynı klasöre yazılır):
  - branches.csv      : şube bilgileri
  - customers.csv      : müşteri bilgileri
  - accounts.csv        : hesap bilgileri
  - transactions.csv    : işlem geçmişi (~55.000 kayıt, 24 ay)

Kasıtlı olarak eklenen desenler (agent'ın "keşfedebileceği" içgörüler için):
  - İstanbul ve İzmir şubelerinde son 6 ayda işlem hacminde belirgin artış
  - Kış aylarında (Ara-Şub) "Fatura Ödemesi" işlemlerinde artış
  - Kredi kartı işlemlerinde diğer işlem tiplerine göre daha yüksek
    "flagged" (şüpheli/inceleme gereken) oranı
  - Düşük kredi skorlu (<550) müşterilerde kredi/loan hesap başına
    ortalama bakiyenin daha volatil olması
"""

import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
from faker import Faker

random.seed(42)
np.random.seed(42)
fake = Faker("tr_TR")
Faker.seed(42)

OUT_DIR = "data"

# ---------------------------------------------------------------------------
# 1) ŞUBELER
# ---------------------------------------------------------------------------
CITIES = [
    ("İstanbul", 6), ("Ankara", 3), ("İzmir", 4), ("Bursa", 2),
    ("Antalya", 2), ("Adana", 1), ("Konya", 1), ("Gaziantep", 1),
]

branches = []
branch_id = 1
for city, n in CITIES:
    for _ in range(n):
        branches.append({
            "branch_id": f"BR{branch_id:03d}",
            "city": city,
            "region": {
                "İstanbul": "Marmara", "Bursa": "Marmara",
                "Ankara": "İç Anadolu", "Konya": "İç Anadolu",
                "İzmir": "Ege", "Antalya": "Akdeniz",
                "Adana": "Akdeniz", "Gaziantep": "Güneydoğu Anadolu",
            }[city],
        })
        branch_id += 1

branches_df = pd.DataFrame(branches)

# ---------------------------------------------------------------------------
# 2) MÜŞTERİLER
# ---------------------------------------------------------------------------
N_CUSTOMERS = 900
SEGMENTS = ["Bireysel", "Bireysel", "Bireysel", "KOBİ", "Kurumsal"]
EMPLOYMENT = ["Ücretli", "Serbest Meslek", "Kendi İşi", "Emekli", "Öğrenci"]

customers = []
for i in range(1, N_CUSTOMERS + 1):
    join_date = fake.date_between(start_date="-6y", end_date="-1M")
    credit_score = int(np.clip(np.random.normal(650, 90), 300, 900))
    customers.append({
        "customer_id": f"CUS{i:05d}",
        "full_name": fake.name(),
        "age": random.randint(18, 78),
        "city": random.choices([c for c, _ in CITIES], weights=[n for _, n in CITIES])[0],
        "segment": random.choice(SEGMENTS),
        "employment": random.choice(EMPLOYMENT),
        "credit_score": credit_score,
        "join_date": join_date.isoformat(),
    })

customers_df = pd.DataFrame(customers)

# ---------------------------------------------------------------------------
# 3) HESAPLAR (bir müşterinin birden fazla hesabı olabilir)
# ---------------------------------------------------------------------------
ACCOUNT_TYPES = ["Vadesiz", "Vadeli", "Kredi Kartı", "Bireysel Kredi", "KOBİ Kredi"]

accounts = []
account_id = 1
for _, cust in customers_df.iterrows():
    n_accounts = random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
    chosen_types = random.sample(ACCOUNT_TYPES, k=min(n_accounts, len(ACCOUNT_TYPES)))
    for acc_type in chosen_types:
        open_date = fake.date_between(start_date=pd.to_datetime(cust["join_date"]).date(), end_date="-1M")
        is_credit_like = acc_type in ("Kredi Kartı", "Bireysel Kredi", "KOBİ Kredi")
        base_balance = np.random.gamma(2, 8000) if not is_credit_like else np.random.gamma(1.5, 15000)
        # düşük kredi skoru -> daha volatil / negatif bakiye olasılığı
        if is_credit_like and cust["credit_score"] < 550:
            base_balance *= np.random.uniform(-1.2, 0.6)
        accounts.append({
            "account_id": f"ACC{account_id:06d}",
            "customer_id": cust["customer_id"],
            "account_type": acc_type,
            "branch_id": random.choice(branches_df["branch_id"].tolist()),
            "open_date": open_date.isoformat(),
            "currency": random.choices(["TRY", "USD", "EUR"], weights=[0.8, 0.13, 0.07])[0],
            "balance": round(float(base_balance), 2),
            "interest_rate": round(random.uniform(1.5, 4.5), 2) if acc_type == "Vadeli" else
                              round(random.uniform(30, 65), 2) if is_credit_like else 0.0,
            "loan_amount": round(float(np.random.gamma(2, 25000)), 2) if "Kredi" in acc_type else 0.0,
        })
        account_id += 1

accounts_df = pd.DataFrame(accounts)

# ---------------------------------------------------------------------------
# 4) İŞLEMLER (24 ay, kasıtlı mevsimsel/bölgesel desenlerle)
# ---------------------------------------------------------------------------
TX_TYPES = ["Havale/EFT", "Fatura Ödemesi", "Market/Alışveriş", "ATM Para Çekme",
            "Maaş Yatışı", "Kredi Kartı Harcaması", "Yatırım İşlemi", "Döviz İşlemi"]
MERCHANT_CATEGORIES = ["Market", "Akaryakıt", "E-ticaret", "Restoran", "Fatura/Kamu",
                        "Sağlık", "Eğitim", "Seyahat", "Elektronik", "Diğer"]

END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=730)

high_growth_cities = {"İstanbul", "İzmir"}
recent_cutoff = END_DATE - timedelta(days=180)

acc_records = accounts_df.merge(customers_df[["customer_id", "city", "credit_score"]], on="customer_id")

rows = []
tx_id = 1
for _, acc in acc_records.iterrows():
    acc_open = pd.to_datetime(acc["open_date"]).date()
    tx_start = max(START_DATE, acc_open)
    n_days = (END_DATE - tx_start).days
    if n_days <= 0:
        continue

    # ortalama işlem sıklığı hesap tipine göre değişir
    base_rate = {
        "Vadesiz": 0.9, "Kredi Kartı": 0.7, "Vadeli": 0.08,
        "Bireysel Kredi": 0.25, "KOBİ Kredi": 0.3,
    }[acc["account_type"]]

    for d in range(n_days):
        cur_date = tx_start + timedelta(days=d)
        rate = base_rate
        # büyüme şehirlerinde son 6 ayda hacim artışı
        if acc["city"] in high_growth_cities and cur_date >= recent_cutoff:
            rate *= 1.6
        # kışın fatura ödemesi artışı
        winter_boost = cur_date.month in (12, 1, 2)

        if random.random() < rate / 4:  # günlük işlem olasılığı
            tx_type = random.choice(TX_TYPES)
            if winter_boost and random.random() < 0.3:
                tx_type = "Fatura Ödemesi"

            if tx_type == "Maaş Yatışı":
                amount = round(np.random.normal(22000, 6000), 2)
            elif tx_type == "Kredi Kartı Harcaması":
                amount = round(abs(np.random.gamma(2, 700)), 2)
            elif tx_type == "ATM Para Çekme":
                amount = -round(abs(np.random.gamma(1.5, 900)), 2)
            elif tx_type in ("Fatura Ödemesi",):
                amount = -round(abs(np.random.gamma(1.8, 450)), 2)
            else:
                amount = round(np.random.normal(0, 3500), 2)

            # kredi kartı işlemlerinde flagged (şüpheli) oranı daha yüksek
            flag_prob = 0.045 if tx_type == "Kredi Kartı Harcaması" else 0.008
            is_flagged = random.random() < flag_prob

            rows.append({
                "transaction_id": f"TX{tx_id:08d}",
                "account_id": acc["account_id"],
                "transaction_date": cur_date.isoformat(),
                "transaction_time": f"{random.randint(0,23):02d}:{random.randint(0,59):02d}",
                "transaction_type": tx_type,
                "amount": amount,
                "currency": acc["currency"],
                "branch_id": random.choice(branches_df["branch_id"].tolist()),
                "merchant_category": random.choice(MERCHANT_CATEGORIES) if tx_type in
                    ("Kredi Kartı Harcaması", "Market/Alışveriş") else None,
                "is_flagged": is_flagged,
            })
            tx_id += 1

transactions_df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# KAYDET
# ---------------------------------------------------------------------------
branches_df.to_csv(f"{OUT_DIR}/branches.csv", index=False)
customers_df.to_csv(f"{OUT_DIR}/customers.csv", index=False)
accounts_df.to_csv(f"{OUT_DIR}/accounts.csv", index=False)
transactions_df.to_csv(f"{OUT_DIR}/transactions.csv", index=False)

print(f"branches:     {len(branches_df):>7,}")
print(f"customers:    {len(customers_df):>7,}")
print(f"accounts:     {len(accounts_df):>7,}")
print(f"transactions: {len(transactions_df):>7,}")
