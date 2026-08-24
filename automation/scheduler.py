"""
Otomasyon katmanı - haftalık otomatik BI raporu
--------------------------------------------------
Bu script, önceden tanımlanmış bir soru listesini agent'a otomatik olarak
sorar, cevapları tek bir rapor halinde birleştirir ve Slack webhook'una
ve/veya e-posta ile gönderir.

Çalıştırma şekilleri:
  1) Tek seferlik test:      python automation/scheduler.py --once
  2) Sürekli çalışan servis: python automation/scheduler.py
     (APScheduler her Pazartesi 09:00'da otomatik tetikler)

Prod ortamında bu scripti bir cron job, GitHub Actions workflow'u ya da
n8n'deki bir "Schedule Trigger" node'unun tetiklediği bir HTTP endpoint
olarak da çalıştırabilirsiniz - mantık aynı kalır.
"""
import argparse
import os
import sys
from datetime import date
from pathlib import Path

import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.agent import ask  # noqa: E402

load_dotenv()

WEEKLY_QUESTIONS = [
    "Bu ayki toplam işlem hacmi ve önceki aya göre değişimi nedir?",
    "Hangi şehirde/bölgede işlem büyümesi en belirgin?",
    "Şüpheli (flagged) işlem oranında dikkat çekici bir değişim var mı?",
    "Kredi skoru düşük müşteri segmentinde risk sinyali var mı?",
]

SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")


def build_report() -> str:
    lines = [f"*FinNova Bank — Haftalık Otomatik BI Raporu ({date.today().isoformat()})*", ""]
    for q in WEEKLY_QUESTIONS:
        result = ask(q)
        lines.append(f"*Soru:* {q}")
        lines.append(result["answer"])
        lines.append("")
    return "\n".join(lines)


def send_to_slack(text: str) -> None:
    if not SLACK_WEBHOOK_URL:
        print("[uyarı] SLACK_WEBHOOK_URL tanımlı değil, Slack'e gönderilmedi.")
        return
    resp = requests.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=15)
    resp.raise_for_status()
    print("Slack'e gönderildi.")


def run_once() -> None:
    report = build_report()
    print(report)
    send_to_slack(report)
    # E-posta göndermek isterseniz burada smtplib / SendGrid entegrasyonu
    # ekleyebilirsiniz - README'de örnek kod var.


def run_scheduled() -> None:
    scheduler = BlockingScheduler(timezone="Europe/Istanbul")
    # Her Pazartesi 09:00'da otomatik çalışır
    scheduler.add_job(run_once, CronTrigger(day_of_week="mon", hour=9, minute=0))
    print("Zamanlayıcı başlatıldı. Her Pazartesi 09:00'da rapor gönderilecek. (Ctrl+C ile durdur)")
    scheduler.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Raporu hemen bir kez üret ve gönder")
    args = parser.parse_args()

    if args.once:
        run_once()
    else:
        run_scheduled()
