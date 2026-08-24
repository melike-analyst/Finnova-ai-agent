# FinNova Bank — Otonom İş Zekası Ajanı

LLM tabanlı, çok adımlı bir **AI agent** kullanarak doğal dilde sorulan finans/bankacılık
sorularını otomatik olarak SQL sorgusuna çeviren, analiz eden, yorumlayan ve
haftalık raporları otomatik gönderen uçtan uca bir sistem.


> **Neden bu proje?** Çoğu "AI portfolyo projesi" tek seferlik bir chatbot demosudur.
> Bu proje bunun yerine üç şeyi aynı anda gösterir: (1) LLM'i harici araçlarla (SQL,
> grafik) birleştiren **agent orkestrasyonu**, (2) bunu **otomasyonla** gerçek bir iş
> akışına bağlamak (haftalık rapor → Slack), (3) **dağıtılmış (deployed)**, çalışan bir
> ürün — sadece Jupyter notebook değil.

---

## 1) Mimari

```
Kullanıcı sorusu / zamanlı tetik
            │
            ▼
   Agent orkestratörü (Google Gemini + native function-calling)
            │
            ├──▶ run_sql   → SQLite veritabanı (yalnızca SELECT, güvenli)
            └──▶ make_chart → otomatik grafik üretimi
            │
            ▼
   Doğal dilde yorumlanmış cevap + (varsa) grafik
            │
            ├──▶ Streamlit arayüzü (canlı demo)
            └──▶ Haftalık otomasyon → Slack / e-posta
```

Agent döngüsü framework'süz, doğrudan Google Gemini'nin native function-calling
API'siyle yazılmıştır (`src/agent.py`). Bu bilinçli bir tercihtir: LangChain/LangGraph
gibi kütüphaneler bu mantığı "kutunun içinde" yapar; burada mantığı şeffaf tutarak
agent kavramının nasıl çalıştığını göstermek hedeflendi. Framework'e geçmek
isterseniz aynı akışı LangGraph `StateGraph` ile yeniden kurabilirsiniz (bkz.
§7 "Genişletme fikirleri").

**LLM sağlayıcı notu:** Proje Google Gemini API kullanır (`gemini-3.5-flash`)
çünkü Gemini, kredi kartı gerektirmeyen gerçek bir ücretsiz katman sunuyor —
portfolyo/demo projeleri için pratik bir tercih. Aynı agent mimarisi küçük
bir değişiklikle Anthropic Claude ya da OpenAI API'sine de taşınabilir;
`src/agent.py` içindeki tool-calling mantığı sağlayıcıdan bağımsız tasarlandı.

## 2) Veri seti

Veri, **tamamen sentetik** ve kurgusal bir "FinNova Bank" için üretilmiştir
(`data/generate_dataset.py`, Faker + numpy ile). Gerçek kişi/kurum verisi
içermez. Bunun bilinçli bir tercih olduğunu not edin: gerçek finans verisi
KVKK/GDPR kısıtları ve lisans sorunları nedeniyle portfolyo projeleri için
uygun değildir; sentetik veri hem bu sorunu ortadan kaldırır hem de veri
üzerinde istediğiniz iş senaryolarını (mevsimsellik, bölgesel büyüme, risk
sinyalleri) kurgulama özgürlüğü verir.

| Tablo | Satır sayısı | Açıklama |
|---|---|---|
| `branches` | 20 | 8 şehirde şube bilgisi |
| `customers` | 900 | Müşteri demografisi, kredi skoru, segment |
| `accounts` | 1.470 | Hesap tipi, bakiye, faiz oranı, kredi tutarı |
| `transactions` | 75.616 | 24 aylık işlem geçmişi |

Veriye kasıtlı olarak gerçekçi iş desenleri gömülüdür (agent'ın bunları
"keşfetmesi" beklenir):

- Kredi kartı harcamalarında şüpheli (flagged) işlem oranı **%4,51**, diğer
  işlem tiplerinde ortalama **~%1** (doğrulandı)
- Kış aylarında (Ara–Şub) fatura ödemesi oranı **%38,5**, diğer mevsimlerde
  **%12,7** (doğrulandıı)
- Son 6 ayda İstanbul ve İzmir şubelerinde aylık ortalama işlem hacminde
  belirgin artış (doğrulandı)

## 3) Kurulum

```bash
git clone <repo-url>
cd finnova-ai-agent
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Veri setini üret ve veritabanına yükle
python data/generate_dataset.py
python data/load_to_sqlite.py

# .env dosyanızı hazırlayın
cp .env.example .env
# .env içine kendi GEMINI_API_KEY'inizi girin (ücretsiz, kredi kartı gerekmez)
# (https://aistudio.google.com/apikey)
```

## 4) Çalıştırma

**Terminalden tek soru sormak için:**
```bash
python -m src.agent "Hangi işlem tipinde şüpheli işlem oranı en yüksek?"
```

**Streamlit arayüzü (canlı demo):**
```bash
streamlit run app/app.py
```

**Haftalık otomasyonu test etmek için (tek seferlik):**
```bash
python automation/scheduler.py --once
```

**Haftalık otomasyonu sürekli servis olarak çalıştırmak için:**
```bash
python automation/scheduler.py
```

## 5) Örnek kullanım

**Soru:** *"Hangi işlem tipinde şüpheli işlem oranı en yüksek?"*

Agent'ın adımları:
1. `run_sql` aracını çağırır → transaction_type bazında flagged oranını hesaplayan SQL üretir
2. Sonucu alır (Kredi Kartı Harcaması: %4,51 — diğerlerine göre ~4,5× daha yüksek)
3. Doğal dilde yorumlar: *"Kredi kartı harcamaları, diğer işlem tiplerine kıyasla
   belirgin şekilde daha yüksek şüpheli işlem oranına sahip. Bu, kart bazlı
   dolandırıcılık kontrolüne öncelik verilmesi gerektiğine işaret ediyor."*

## 6) Güvenlik notları

- `src/tools/sql_tool.py`, LLM'in ürettiği SQL'i çalıştırmadan önce doğrular:
  yalnızca `SELECT` ifadelerine izin verilir, `INSERT/UPDATE/DELETE/DROP` gibi
  komutlar ve çoklu ifadeler (`;`) reddedilir.
- `.env` dosyası `.gitignore` içinde — API anahtarınız asla repoya gitmez.
- Sonuçlar 200 satırla sınırlandırılır (agent'ın context penceresini şişirmemek
  ve olası maliyet artışını önlemek için).

## 7) Genişletme fikirleri (sonraki adımlar)

- **RAG katmanı ekleyin:** Şirket politika/prosedür PDF'lerini de sorgulayabilen
  bir vektör veritabanı (Chroma) entegre edin — "Kredi onay politikamıza göre
  bu müşteri uygun mu?" gibi sorulara cevap verebilsin.
- **LangGraph'a geçiş:** `src/agent.py`'deki döngüyü bir `StateGraph` olarak
  yeniden modelleyin; çoklu-agent (planlayıcı + analist + yazar) mimarisine
  genişletin (CrewAI ile de yapılabilir).
- **Değerlendirme (eval) seti:** 20-30 soru-beklenen SQL çiftinden oluşan bir
  test seti kurup "doğru SQL üretme oranı" metriğini CI/CD'ye (GitHub Actions)
  bağlayın.
- **n8n entegrasyonu:** `automation/scheduler.py`'nin yaptığını no-code bir
  n8n workflow'una taşıyıp Slack yerine Notion/Google Sheets'e de yazdırın.

## 8) Proje yapısı

```
finnova-ai-agent/
├── data/
│   ├── generate_dataset.py   # sentetik veri üretimi
│   ├── load_to_sqlite.py     # CSV → SQLite
│   └── finnova_bank.db
├── src/
│   ├── db.py                 # bağlantı + şema çıkarımı
│   ├── agent.py              # agent orkestrasyon döngüsü
│   └── tools/
│       ├── sql_tool.py       # güvenli SQL çalıştırma
│       └── chart_tool.py     # otomatik grafik üretimi
├── app/
│   └── app.py                # Streamlit arayüzü
├── automation/
│   └── scheduler.py          # haftalık otomatik rapor
├── requirements.txt
├── .env.example
└── README.md
```

## 9) Kullanılan teknolojiler

`Python` · `Google Gemini API (function-calling / agentic loop)` · `SQLite` ·
`pandas` · `Streamlit` · `APScheduler` · `Slack Webhooks` · (opsiyonel
genişletme: `LangGraph`, `Chroma`, `n8n`)

---

*Not: Bu proje bir portfolyo/öğrenme çalışmasıdır. Kullanılan tüm veriler
sentetiktir, gerçek bir banka veya müşteri verisiyle ilişkisi yoktur.*
