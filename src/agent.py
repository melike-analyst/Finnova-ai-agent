"""
Otonom İş Zekası Ajanı - agent orkestrasyon katmanı (Google Gemini sürümü)
----------------------------------------------------------------------------
Bu modül, Google Gemini API'sinin native function-calling özelliğini
kullanarak bir "agent döngüsü" kurar.
"""
import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.db import get_schema_description
from src.tools.chart_tool import make_chart
from src.tools.sql_tool import UnsafeQueryError, run_sql

load_dotenv()

MODEL = "gemini-3.5-flash-lite"  # Ücretsiz katmanda kullanılabilen, hızlı model
MAX_TURNS = 8  # sonsuz döngüyü önlemek için güvenlik sınırı

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

RUN_SQL_DECLARATION = types.FunctionDeclaration(
    name="run_sql",
    description=(
        "FinNova Bank veritabanında salt-okunur bir SELECT sorgusu "
        "çalıştırır ve sonucu tablo olarak döner. Kullanıcının sorusunu "
        "cevaplamak için gereken veriyi çekmek üzere kullan."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Çalıştırılacak SELECT sorgusu"},
            "purpose": {"type": "string", "description": "Bu sorgunun neyi ölçtüğüne dair kısa açıklama"},
        },
        "required": ["query"],
    },
)

MAKE_CHART_DECLARATION = types.FunctionDeclaration(
    name="make_chart",
    description=(
        "Bir önceki run_sql sonucundan görsel bir grafik (çizgi ya da "
        "bar) üretir. Kullanıcı trend, karşılaştırma ya da dağılım "
        "istediğinde kullan."
    ),
    parameters={
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    },
)

TOOLS = [types.Tool(function_declarations=[RUN_SQL_DECLARATION, MAKE_CHART_DECLARATION])]


def _system_prompt() -> str:
    schema = get_schema_description()
    return f"""Sen FinNova Bank için çalışan bir kıdemli veri analisti agentısın.
Görevin, iş kullanıcılarının doğal dilde sorduğu soruları veritabanını
sorgulayarak, analiz ederek ve anlaşılır şekilde yorumlayarak cevaplamak.

VERİTABANI ŞEMASI:
{schema}


KURALLAR:
- Karşılaştırma sorularında (örn. iki şehir, iki dönem) mümkünse TÜM veriyi TEK bir SQL sorgusuyla çek (GROUP BY / CASE WHEN kullanarak), ayrı ayrı sorgular çalıştırmak yerine.
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
    config = types.GenerateContentConfig(
        system_instruction=_system_prompt(),
        tools=TOOLS,
    )
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]
    last_df = None
    chart_path = None

    for _ in range(MAX_TURNS):
        response = client.models.generate_content(model=MODEL, contents=contents, config=config)
        candidate = response.candidates[0]
        function_calls = [p.function_call for p in candidate.content.parts if p.function_call]

        if not function_calls:
            final_text = response.text or ""
            return {"answer": final_text, "chart_path": chart_path}

        contents.append(candidate.content)  # modelin function_call içeren turu

        response_parts = []
        for fc in function_calls:
            tool_input = dict(fc.args) if fc.args else {}
            result_text, last_df = _execute_tool(fc.name, tool_input, last_df)
            if fc.name == "make_chart" and isinstance(result_text, str) and result_text.startswith("Grafik kaydedildi"):
                chart_path = result_text.split(": ", 1)[1]
            response_parts.append(
                types.Part.from_function_response(name=fc.name, response={"result": str(result_text)})
            )
        contents.append(types.Content(role="user", parts=response_parts))

    return {"answer": "Üzgünüm, soruyu maksimum adım sayısında çözemedim.", "chart_path": chart_path}


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "Hangi işlem tipinde şüpheli işlem oranı en yüksek?"
    print(json.dumps(ask(q), ensure_ascii=False, indent=2))
