# %% [markdown]
# # FinNova Bank — Şüpheli İşlem (Fraud) Tahmin Modeli
#
# Bu notebook, "Otonom İş Zekası Ajanı" projesinin **klasik makine öğrenmesi**
# katmanını oluşturur. Ana projedeki LLM agent, doğal dil sorularını cevaplar;
# bu notebook ise farklı bir problem çözer: **bir işlem gerçekleşmeden/işlenirken
# şüpheli olma olasılığını önceden tahmin etmek**.
#
# İzlenen adımlar (uçtan uca bir ML iş akışı):
# 1. Veriyi yükleme ve keşifsel veri analizi (EDA)
# 2. Özellik mühendisliği (feature engineering)
# 3. Eğitim/test ayrımı ve sınıf dengesizliği ile başa çıkma
# 4. İki farklı modeli (Logistic Regression, Random Forest) eğitip karşılaştırma
# 5. Değerlendirme: precision, recall, F1, ROC-AUC, confusion matrix
# 6. Özellik önem sıralaması (hangi faktörler fraud'u tetikliyor?)

# %%
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
)

sns.set_style("whitegrid")
pd.set_option("display.max_columns", None)

DB_PATH = "../data/finnova_bank.db"  # notebooks/ klasöründen çalıştırıldığı varsayılıyor

# %% [markdown]
# ## 1) Veriyi yükle ve birleştir
#
# `transactions` tablosunu `accounts` ve `customers` ile birleştirerek,
# modelin işlem bazlı DEĞİL, **müşteri ve hesap bağlamını da bilerek**
# tahmin yapmasını sağlıyoruz (örn. düşük kredi skorlu bir müşterinin
# yüksek tutarlı bir işlemi, yüksek skorlu birine göre farklı bir risk taşır).

# %%
conn = sqlite3.connect(DB_PATH)
query = """
SELECT
    t.transaction_id, t.transaction_type, t.amount, t.transaction_date,
    t.transaction_time, t.merchant_category, t.is_flagged,
    a.account_type, a.currency, a.balance,
    c.age, c.segment, c.employment, c.credit_score
FROM transactions t
JOIN accounts a ON t.account_id = a.account_id
JOIN customers c ON a.customer_id = c.customer_id
"""
df = pd.read_sql_query(query, conn)
conn.close()

print(f"Toplam işlem sayısı: {len(df):,}")
print(f"Şüpheli (flagged) işlem oranı: %{100 * df['is_flagged'].mean():.2f}")
df.head()

# %% [markdown]
# ## 2) Keşifsel Veri Analizi (EDA)
#
# Modele geçmeden önce, hangi değişkenlerin fraud ile ilişkili göründüğünü
# görselleştirerek anlıyoruz. Bu adım, hangi özellikleri mühendisleyeceğimize
# karar vermemizi sağlıyor.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

flag_by_type = df.groupby("transaction_type")["is_flagged"].mean().sort_values(ascending=False) * 100
flag_by_type.plot(kind="bar", ax=axes[0], color="#4C72B0")
axes[0].set_title("İşlem Tipine Göre Şüpheli İşlem Oranı (%)")
axes[0].set_ylabel("%")
axes[0].tick_params(axis="x", rotation=45)

sns.boxplot(data=df, x="is_flagged", y="credit_score", ax=axes[1])
axes[1].set_title("Kredi Skoru Dağılımı: Normal vs Şüpheli İşlem")
axes[1].set_xticklabels(["Normal", "Şüpheli"])

plt.tight_layout()
plt.savefig("eda_overview.png", dpi=120)
plt.show()

# %% [markdown]
# **Gözlem:** Kredi kartı harcamaları belirgin şekilde daha yüksek şüpheli
# işlem oranına sahip — bu, modelin `transaction_type`'ı güçlü bir sinyal
# olarak kullanacağını gösteriyor. Kredi skoru dağılımında ise iki grup
# arasında büyük bir fark yok; bu da tek başına kredi skorunun güçlü bir
# ayırt edici olmadığını, ama diğer özelliklerle birlikte katkı sağlayabileceğini
# gösteriyor.

# %% [markdown]
# ## 3) Özellik Mühendisliği (Feature Engineering)
#
# Ham kolonlardan modelin daha iyi öğrenebileceği yeni özellikler türetiyoruz:
# - `transaction_hour`: günün hangi saati (gece yarısı işlemleri daha riskli olabilir)
# - `is_weekend`: hafta sonu mu
# - `amount_abs`: işlem tutarının mutlak değeri (yön değil, büyüklük önemli)
# - `high_amount_flag`: tutar, o işlem tipinin medyanının kaç katı

# %%
df["transaction_date"] = pd.to_datetime(df["transaction_date"])
df["transaction_hour"] = df["transaction_time"].str.split(":").str[0].astype(int)
df["is_weekend"] = df["transaction_date"].dt.dayofweek.isin([5, 6]).astype(int)
df["amount_abs"] = df["amount"].abs()

median_by_type = df.groupby("transaction_type")["amount_abs"].transform("median")
df["amount_vs_type_median"] = df["amount_abs"] / median_by_type.replace(0, 1)

FEATURES_NUMERIC = ["amount_abs", "amount_vs_type_median", "balance", "age",
                     "credit_score", "transaction_hour", "is_weekend"]
FEATURES_CATEGORICAL = ["transaction_type", "account_type", "segment", "employment"]

X = df[FEATURES_NUMERIC + FEATURES_CATEGORICAL].copy()
y = df["is_flagged"].astype(int)

print("Özellik seti boyutu:", X.shape)
print("Sınıf dağılımı:\n", y.value_counts(normalize=True).round(4))

# %% [markdown]
# ## 4) Eğitim/Test Ayrımı
#
# `stratify=y` kullanıyoruz çünkü sınıflar dengesiz (%~1-2 fraud) — bu,
# eğitim ve test setlerinde fraud oranının aynı kalmasını garanti eder.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Eğitim seti: {len(X_train):,} satır | Test seti: {len(X_test):,} satır")

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), FEATURES_NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CATEGORICAL),
])

# %% [markdown]
# ## 5) Model Eğitimi: Logistic Regression vs Random Forest
#
# İki modeli karşılaştırıyoruz: Logistic Regression (basit, yorumlanabilir,
# iyi bir başlangıç noktası) ve Random Forest (doğrusal olmayan ilişkileri
# yakalayabilen, genelde daha güçlü bir model). `class_weight="balanced"`
# kullanarak azınlık sınıfına (fraud) daha fazla önem veriyoruz.

# %%
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
    ),
}

results = {}
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", clf)])
    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)
    results[name] = {"pipeline": pipe, "y_proba": y_proba, "y_pred": y_pred}
    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    print(f"\n{'='*50}\n{name}  |  ROC-AUC: {auc:.3f}  |  Average Precision: {ap:.3f}\n{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Şüpheli"]))

# %% [markdown]
# ## 6) Değerlendirme Görselleri: ROC ve Precision-Recall Eğrileri
#
# Sınıflar dengesiz olduğu için (fraud ~%1-2), **sadece accuracy yanıltıcıdır**
# — bir model her şeye "normal" deyip bile %98+ accuracy alabilir. Bu yüzden
# ROC-AUC ve özellikle Precision-Recall eğrisine (Average Precision) bakıyoruz.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, res['y_proba']):.3f})")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Eğrisi")
axes[0].legend()

for name, res in results.items():
    prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
    axes[1].plot(rec, prec, label=name)
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Eğrisi")
axes[1].legend()

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=120)
plt.show()

# %% [markdown]
# ## 7) En İyi Modelin Confusion Matrix'i
#
# Average Precision'a göre daha iyi performans gösteren modeli seçip
# confusion matrix ile hatalarının niteliğine bakıyoruz: false positive'ler
# (gereksiz yere incelemeye alınan normal işlemler) ile false negative'ler
# (kaçırılan gerçek fraud'lar) arasındaki dengeyi görmek iş açısından kritik.

# %%
best_name = max(results, key=lambda n: average_precision_score(y_test, results[n]["y_proba"]))
best = results[best_name]
print(f"En iyi model (Average Precision'a göre): {best_name}")

cm = confusion_matrix(y_test, best["y_pred"])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Şüpheli"], yticklabels=["Normal", "Şüpheli"])
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
plt.show()

# %% [markdown]
# ## 8) Özellik Önem Sıralaması (Random Forest)
#
# Hangi özelliklerin modelin kararında en çok ağırlığı olduğunu görmek,
# hem modeli yorumlamak hem de iş birimine "buna dikkat edin" diyebilmek
# için önemli.

# %%
rf_pipe = results["Random Forest"]["pipeline"]
feature_names = rf_pipe.named_steps["prep"].get_feature_names_out()
importances = rf_pipe.named_steps["model"].feature_importances_

importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
importance_df = importance_df.sort_values("importance", ascending=False).head(12)

plt.figure(figsize=(8, 5))
sns.barplot(data=importance_df, x="importance", y="feature", color="#4C72B0")
plt.title("En Önemli 12 Özellik (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120)
plt.show()

importance_df

# %% [markdown]
# ## Sonuç ve İş Etkisi
#
# Bu model, FinNova Bank için işlemler gerçekleşirken (ya da gerçekleştikten
# hemen sonra) şüpheli olma olasılığını tahmin ederek, manuel inceleme
# ekibinin önceliklendirme yapmasına yardımcı olabilir. Üretime alınması
# durumunda önerilen sonraki adımlar:
#
# - **Eşik (threshold) optimizasyonu**: Precision-Recall eğrisine göre iş
#   birimiyle birlikte "kaç false positive'e katlanabiliriz" sorusuna göre
#   karar eşiği belirlenmeli (varsayılan 0.5 yerine).
# - **Zamansal doğrulama**: Şu an rastgele train/test ayrımı yapıldı; gerçek
#   üretimde zaman bazlı ayrım (geçmiş veriyle eğitip, gelecek veride test
#   etmek) daha gerçekçi bir performans tahmini verir.
# - **Model izleme**: Ana projedeki LLM agent'a "model performansı bu ay
#   nasıldı?" gibi sorulara cevap verebilecek bir araç eklenebilir — klasik
#   ML ve LLM agent katmanlarını birbirine bağlayan doğal bir sonraki adım.
