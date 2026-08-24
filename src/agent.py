"""
Otonom İş Zekası Ajanı - agent orkestrasyon katmanı
------------------------------------------------------
Bu modül, Anthropic'in native tool-use (function calling) API'sini
kullanarak bir "agent döngüsü" (agentic loop) kurar:

    1. Kullanıcı doğal dilde soru sorar
    2. LLM, soruyu cevaplamak için hangi aracı çağırması gerektiğine
       karar verir (run_sql, make_chart)
    3. Araç çalıştırılır, sonucu tekrar LLM'e verilir
    4. LLM ya başka bir araç çağırır ya da nihai, doğal dilde
       yorumlanmış cevabı üretir

Bu, LangChain/LangGraph gibi framework'lerin "arka planda" yaptığı işin
şeffaf, framework'süz bir versiyonudur - agent kavramını gerçekten
anladığınızı göstermek için bilinçli bir tercihtir. Framework'e geçmek
isterseniz aynı mantığı LangGraph StateGraph ile de kurabilirsiniz
(bkz. README, "Framework'e geçiş" bölümü).
"""
import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv

from src.db import get_schema_description
from src.tools.chart_tool import make_chart
from src.tools.sql_tool import UnsafeQueryError, run_sql

load_dotenv()

MODEL = "claude-sonnet-5"  # Güncel model listesi için: https://platform.claude.com/docs/en/about-claude/models/overview
MAX_TURNS = 5  # sonsuz döngüyü önlemek için güvenlik sınırı

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

TOOLS = [
    {
        "name": "run_sql",
        "description": (
            "FinNova Bank veritabanında salt-okunur bir SELECT sorgusu "
            "çalıştırır ve sonucu tablo olarak döner. Kullanıcının sorusunu "
            "cevaplamak için gereken veriyi çekmek üzere kullan."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Çalıştırılacak SELECT sorgusu"},
                "purpose": {"type": "string", "description": "Bu sorgunun neyi ölçtüğüne dair kısa açıklama"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "make_chart",
        "description": (
            "Bir önceki run_sql sonucundan görsel bir grafik (çizgi ya da "
            "bar) üretir. Kullanıcı trend, karşılaştırma ya da dağılım "
            "istediğinde kullan."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        },
    },
]


def _system_prompt() -> str:
    schema = get_schema_description()
    return f"""Sen FinNova Bank için çalışan bir kıdemli veri analisti agentısın.
Görevin, iş kullanıcılarının doğal dilde sorduğu soruları veritabanını
sorgulayarak, analiz ederek ve anlaşılır şekilde yorumlayarak cevaplamak.

VERİTABANI ŞEMASI:
{schema}

KURALLAR:
- Sadece SELECT sorguları çalıştırabilirsin (run_sql aracı zaten bunu zorunlu kılar).
- Sayısal sonuçları yorumla: sadece rakam verme, ne anlama geldiğini açıkla.
- Mümkünse iş etkisini vurgula (risk, fırsat, trend).
- Emin olmadığın bir varsayım yapıyorsan bunu açıkça belirt.
- Türkçe, kısa ve öz cevap ver. Gereksiz teknik jargon kullanma.
"""


def _execute_tool(name: str, tool_input: dict, last_df):
    if name == "run_sql":
        try:
            df, truncated = run_sql(tool_input["query"])
        except UnsafeQueryError as e:
            return {"error": str(e)}, last_df
        result_text = df.to_markdown(index=False) if not df.empty else "(sonuç boş)"
        if truncated:
            result_text += "\n\n[Not: sonuç 200 satırla sınırlandı]"
        return result_text, df

    if name == "make_chart":
        if last_df is None or last_df.empty:
            return "Grafik için önce run_sql ile veri çekilmeli.", last_df
        path = make_chart(last_df, title=tool_input.get("title", ""))
        return (f"Grafik kaydedildi: {path}" if path else "Bu veriyle uygun bir grafik üretilemedi."), last_df

    return f"Bilinmeyen araç: {name}", last_df


def ask(question: str) -> dict:
    """Kullanıcı sorusunu agent döngüsüne sokar, nihai cevabı ve varsa
    üretilen grafik yolunu döner."""
    messages = [{"role": "user", "content": question}]
    last_df = None
    chart_path = None

    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=_system_prompt(),
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            final_text = "".join(b.text for b in response.content if b.type == "text")
            return {"answer": final_text, "chart_path": chart_path}

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result_text, last_df = _execute_tool(block.name, block.input, last_df)
            if block.name == "make_chart" and isinstance(result_text, str) and result_text.startswith("Grafik kaydedildi"):
                chart_path = result_text.split(": ", 1)[1]
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(result_text),
            })
        messages.append({"role": "user", "content": tool_results})

    return {"answer": "Üzgünüm, soruyu maksimum adım sayısında çözemedim.", "chart_path": chart_path}


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "Hangi işlem tipinde şüpheli işlem oranı en yüksek?"
    print(json.dumps(ask(q), ensure_ascii=False, indent=2))
