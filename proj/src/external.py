# -*- coding: utf-8 -*-
"""متغیرهای بیرونی: تقویم قمری + نرخ ارز + بازه‌های غیرعادی

نرخ ارز اختیاری است. اگر فایل نباشد، فیچرهایش صفر می‌شوند و مدل کار می‌کند.
"""
import pandas as pd, numpy as np
from pathlib import Path
from config import RAW, FX_FILE, ANOMALY_FILE

# ---------------------------------------------------------------- قمری
def add_hijri(g):
    """رمضان، عید فطر، محرم، اربعین"""
    try:
        from hijridate import Gregorian
    except ImportError:
        from hijri_converter import Gregorian

    u = pd.Series(g.date.unique())
    conv = u.map(lambda x: Gregorian(x.year, x.month, x.day).to_hijri())
    hm = dict(zip(u, conv.map(lambda x: x.month)))
    hd = dict(zip(u, conv.map(lambda x: x.day)))

    g["hmonth"] = g.date.map(hm)
    g["hday"] = g.date.map(hd)

    g["is_ramadan"] = (g.hmonth == 9).astype(int)
    # پیشرفت ماه رمضان: اول ماه با آخرش فرق دارد
    g["ramadan_prog"] = np.where(g.hmonth == 9, g.hday / 30.0, 0.0)
    g["is_last10_ramadan"] = ((g.hmonth == 9) & (g.hday >= 21)).astype(int)
    g["is_eid_fitr"] = ((g.hmonth == 10) & (g.hday <= 3)).astype(int)
    g["is_muharram"] = ((g.hmonth == 1) & (g.hday <= 12)).astype(int)
    g["is_arbaeen"] = ((g.hmonth == 2) & (g.hday.between(18, 22))).astype(int)
    return g

# ---------------------------------------------------------------- ارز
FX_COLS_DATE = ["date", "تاریخ", "jdate", "تاریخ شمسی", "Date"]
FX_COLS_VAL = ["price", "close", "قیمت", "نرخ", "value", "Close", "آخرین"]

def _read_fx():
    """فایل نرخ ارز را می‌خواند. CSV یا Excel. ستون تاریخ و قیمت را حدس می‌زند."""
    p = Path(FX_FILE)
    if not p.exists():
        return None
    df = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_excel(p)
    dcol = next((c for c in df.columns if str(c).strip() in FX_COLS_DATE), None)
    vcol = next((c for c in df.columns if str(c).strip() in FX_COLS_VAL), None)
    if dcol is None or vcol is None:
        print(f"  ⚠ ستون تاریخ/قیمت در {p.name} پیدا نشد. ستون‌ها: {list(df.columns)}")
        return None

    s = df[[dcol, vcol]].copy()
    s.columns = ["d", "v"]
    s["v"] = pd.to_numeric(s.v.astype(str).str.replace(r"[,\s]", "", regex=True),
                           errors="coerce")

    # تاریخ: شمسی یا میلادی؟
    ds = s.d.astype(str).str.strip()
    if ds.str.match(r"^1[34]\d\d[/-]").any():
        import jdatetime
        def conv(x):
            try:
                y, m, dd = map(int, x.replace("-", "/").split("/"))
                return pd.Timestamp(jdatetime.date(y, m, dd).togregorian())
            except Exception:
                return pd.NaT
        s["date"] = ds.map(conv)
    else:
        s["date"] = pd.to_datetime(ds, errors="coerce")

    s = s.dropna(subset=["date", "v"]).sort_values("date")
    return s[["date", "v"]].rename(columns={"v": "fx"})

def add_fx(g):
    """نرخ ارز به صورت تغییر نسبی — نه سطح مطلق.

    سطح مطلق روند صعودی دارد و درخت تصمیم نمی‌تواند برون‌یابی کند
    (همان اشتباهی که با price_idx رخ داد و wMAPE را به ۱۰۸٪ رساند).
    """
    fx = _read_fx()
    cols = ["fx_chg_7", "fx_chg_30", "fx_vol_14"]
    if fx is None:
        for c in cols: g[c] = 0.0
        return g, False

    full = pd.DataFrame({"date": pd.date_range(fx.date.min(), fx.date.max())})
    fx = full.merge(fx, on="date", how="left").ffill()
    fx["fx_chg_7"] = fx.fx.pct_change(7)
    fx["fx_chg_30"] = fx.fx.pct_change(30)
    fx["fx_vol_14"] = fx.fx.pct_change().rolling(14).std()

    g = g.merge(fx[["date"] + cols], on="date", how="left")
    for c in cols: g[c] = g[c].fillna(0.0)
    print(f"  ✅ نرخ ارز: {fx.date.min().date()} تا {fx.date.max().date()}")
    return g, True

# ---------------------------------------------------------------- بازه‌های غیرعادی
def add_anomalies(g):
    """بازه‌های غیرعادی از فایل CSV — از آموزش حذف می‌شوند.

    ستون‌ها: start,end,branch,reason
      start/end : تاریخ شمسی 1404/12/09
      branch    : اسم شعبه، یا ALL برای همه
    """
    p = Path(ANOMALY_FILE)
    g["is_anomaly"] = 0
    g["anomaly_reason"] = ""
    if not p.exists():
        return g, 0

    import jdatetime
    ev = pd.read_csv(p, dtype=str).dropna(subset=["start", "end"])
    def tg(x):
        y, m, d = map(int, str(x).strip().replace("-", "/").split("/"))
        return pd.Timestamp(jdatetime.date(y, m, d).togregorian())

    n = 0
    for _, r in ev.iterrows():
        try:
            s, e = tg(r["start"]), tg(r["end"])
        except Exception:
            print(f"  ⚠ تاریخ نامعتبر: {r.get('start')} تا {r.get('end')}")
            continue
        m = (g.date >= s) & (g.date <= e)
        br = str(r.get("branch", "ALL")).strip()
        if br and br.upper() != "ALL":
            m &= (g.branch == br)
        g.loc[m, "is_anomaly"] = 1
        g.loc[m, "anomaly_reason"] = str(r.get("reason", ""))[:40]
        n += int(m.sum())
    if n:
        print(f"  ✅ {len(ev)} بازه غیرعادی — {n:,} ردیف علامت خورد")
    return g, n

HIJRI_FEATS = ["is_ramadan", "ramadan_prog", "is_last10_ramadan",
               "is_eid_fitr", "is_muharram", "is_arbaeen"]
FX_FEATS = ["fx_chg_7", "fx_chg_30", "fx_vol_14"]
