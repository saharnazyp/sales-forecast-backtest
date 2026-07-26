# -*- coding: utf-8 -*-
"""مرحله ۱: خواندن فایل اکسل و تبدیل به parquet + تاریخ میلادی
هر شیت جدا پردازش و ذخیره می‌شود تا حافظه پر نشود."""
import pandas as pd, gc, sys, jdatetime
from config import SALES_FILE, SHEETS, COL_MAP, PROC

CHUNK = 200_000

def _conv_dates(df):
    u = pd.Series(df.jdate.astype(str).unique())
    def conv(s):
        try:
            y, m, d = map(int, s.split("/"))
            return pd.Timestamp(jdatetime.date(y, m, d).togregorian())
        except Exception:
            return pd.NaT
    return df.jdate.astype(str).map(dict(zip(u, u.map(conv))))

def extract_sheet(name):
    """یک شیت را می‌خواند، پاک‌سازی و ذخیره می‌کند"""
    from pyxlsb import open_workbook
    parts, rows = [], []

    def flush():
        nonlocal rows
        if not rows: return
        d = pd.DataFrame(rows, columns=list(COL_MAP.values()))
        rows = []
        d = d[d.jdate.notna()]
        for c in ["qty", "price", "discount"]:
            d[c] = pd.to_numeric(d[c], errors="coerce").astype("float32")
        d["date"] = _conv_dates(d)
        d = d.dropna(subset=["date"])
        d["jdate"] = d.jdate.astype(str)
        for c in ["branch", "item_name", "channel"]:
            d[c] = d[c].astype(str).astype("category")
        parts.append(d)
        gc.collect()

    with open_workbook(str(SALES_FILE)) as wb:
        with wb.get_sheet(name) as sh:
            for i, row in enumerate(sh.rows()):
                if i == 0: continue
                cells = {c.c: c.v for c in row}
                rows.append([cells.get(j) for j in COL_MAP])
                if len(rows) >= CHUNK: flush()
    flush()

    df = pd.concat(parts, ignore_index=True)
    del parts; gc.collect()
    out = PROC / f"_sheet_{name}.parquet"
    df.to_parquet(out, index=False)
    n = len(df)
    del df; gc.collect()
    return out, n

def main():
    if not SALES_FILE.exists():
        sys.exit(f"❌ فایل پیدا نشد: {SALES_FILE}\n"
                 f"   فایل اکسل فروش را در پوشه data/raw/ بگذار\n"
                 f"   و اگر اسمش فرق دارد، در src/config.py خط SALES_FILE را عوض کن")

    paths, total = [], 0
    for s in SHEETS:
        p, n = extract_sheet(s)
        print(f"  {s}: {n:,} ردیف")
        paths.append(p); total += n

    # ادغام نهایی
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    for c in ["branch", "item_name", "channel"]:
        df[c] = df[c].astype("category")
    df.to_parquet(PROC / "clean.parquet", index=False)
    for p in paths: p.unlink(missing_ok=True)

    print(f"✅ {len(df):,} ردیف | {df.date.min().date()} تا {df.date.max().date()}")
    print(f"   شعبه: {df.branch.nunique()} | کالا: {df.item_name.nunique()}")

if __name__ == "__main__":
    main()
