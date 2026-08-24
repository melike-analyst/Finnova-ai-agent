"""
Sorgu sonucu DataFrame'inden otomatik grafik üretir.

Agent, kullanıcı "trend göster" / "karşılaştır" gibi bir şey sorduğunda
bu aracı çağırır. Basit bir sezgisel kural kullanır: tarih kolonu varsa
çizgi grafik, kategorik + sayısal kolon varsa bar grafik.
"""
import uuid
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sunucu ortamında ekran gerektirmez
import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "charts"
OUTPUT_DIR.mkdir(exist_ok=True)


def make_chart(df: pd.DataFrame, title: str = "") -> str | None:
    """DataFrame'e göre uygun grafik tipini seçip PNG olarak kaydeder.
    Grafik üretilemiyorsa (örn. veri uygun değilse) None döner."""
    if df.empty or len(df.columns) < 2:
        return None

    date_cols = [c for c in df.columns if "date" in c.lower() or "tarih" in c.lower()]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    if not numeric_cols:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.5))

    if date_cols and numeric_cols:
        x, y = date_cols[0], numeric_cols[0]
        plot_df = df.sort_values(x)
        ax.plot(plot_df[x], plot_df[y], marker="o", linewidth=2)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        plt.xticks(rotation=45, ha="right")
    elif cat_cols and numeric_cols:
        x, y = cat_cols[0], numeric_cols[0]
        plot_df = df.sort_values(y, ascending=False).head(15)
        ax.bar(plot_df[x].astype(str), plot_df[y])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        plt.xticks(rotation=45, ha="right")
    else:
        plt.close(fig)
        return None

    ax.set_title(title)
    fig.tight_layout()

    out_path = OUTPUT_DIR / f"chart_{uuid.uuid4().hex[:8]}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return str(out_path)
