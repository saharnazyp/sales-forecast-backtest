# -*- coding: utf-8 -*-
"""خلاصه ۲۸ روزه برای آیتم‌های گروه B با بازه اطمینان

چرا: برای آیتم‌های کم‌تردد پیش‌بینی روزانه بی‌معنی است — ۷۶٪ روزها صفرند
و wMAPE بالای ۱۱۰٪ می‌شود. ولی جمع ۲۸ روزه قابل اعتماد است چون در سطح
ماهانه صفرها خنثی می‌شوند.

خروجی: جمع مورد انتظار + بازه ۸۰٪ + برچسب وضعیت
"""
import pandas as pd, numpy as np
from config import PROC, FORECAST_DAYS

LOOKBACK = 84          # ۱۲ هفته
MIN_HISTORY = 35       # کمتر از این = آیتم جدید


def build(g=None):
    if g is None:
        g = pd.read_parquet(PROC / "daily_item.parquet")
    gb = g[(g.tier == "B") & (g.exclude == 0)].copy()
    if gb.empty:
        return pd.DataFrame()

    last = g.date.max()
    rec = gb[gb.date > last - pd.Timedelta(days=LOOKBACK)].copy()

    # آمار پایه
    a = rec.groupby(["branch", "item"]).agg(
        days_present=("date", "nunique"),
        days_sold=("qty", lambda x: (x > 0).sum()),
        total=("qty", "sum"),
        last_sale=("date", lambda x: x[rec.loc[x.index, "qty"] > 0].max()
                   if (rec.loc[x.index, "qty"] > 0).any() else pd.NaT),
    ).reset_index()

    a["rate_day"] = a.total / a.days_present.replace(0, np.nan)
    a["freq"] = a.days_sold / a.days_present.replace(0, np.nan)

    # نوسان هفتگی برای بازه اطمینان
    rec["wk"] = (rec.date - rec.date.min()).dt.days // 7
    wk = rec.groupby(["branch", "item", "wk"]).qty.sum().reset_index()
    v = wk.groupby(["branch", "item"]).qty.agg(["mean", "std"]).reset_index()
    v.columns = ["branch", "item", "wk_mean", "wk_std"]
    a = a.merge(v, on=["branch", "item"], how="left")
    a["wk_std"] = a.wk_std.fillna(a.wk_mean * 0.5)

    n_weeks = FORECAST_DAYS / 7.0
    a["expected"] = a.rate_day * FORECAST_DAYS
    a["sd"] = a.wk_std * np.sqrt(n_weeks)
    a["lo"] = np.maximum(0, a.expected - 1.28 * a.sd).round(0)   # ~80%
    a["hi"] = (a.expected + 1.28 * a.sd).round(0)
    a["expected"] = a.expected.round(1)

    # روزهای بی‌فروش
    a["days_since_sale"] = (last - a.last_sale).dt.days

    def label(r):
        if r.days_present < MIN_HISTORY:
            return "⚠ جدید — سابقه کوتاه"
        if pd.notna(r.days_since_sale) and r.days_since_sale > 21:
            return "⚠ راکد — بیش از ۳ هفته بی‌فروش"
        if r.freq >= 0.5:
            return "منظم"
        if r.rate_day >= 1:
            return "پرحجم پراکنده"
        return "کم‌فروش"

    a["وضعیت"] = a.apply(label, axis=1)

    # پیشنهاد سفارش: حد بالای بازه برای اقلام کم‌ریسک، انتظار برای بقیه
    a["پیشنهاد سفارش"] = np.where(
        a["وضعیت"].str.startswith("⚠"), a.hi, np.ceil(a.expected))

    out = a[["branch", "item", "وضعیت", "expected", "lo", "hi",
             "پیشنهاد سفارش", "rate_day", "freq", "days_present",
             "days_since_sale"]].copy()
    out.columns = ["شعبه", "آیتم منو", "وضعیت", f"انتظار {FORECAST_DAYS} روز",
                   "حداقل", "حداکثر", "پیشنهاد سفارش", "میانگین روزانه",
                   "نسبت روزهای فروش", "روز سابقه", "روز از آخرین فروش"]
    out["میانگین روزانه"] = out["میانگین روزانه"].round(2)
    out["نسبت روزهای فروش"] = (out["نسبت روزهای فروش"] * 100).round(0)
    return out.sort_values(["شعبه", f"انتظار {FORECAST_DAYS} روز"],
                           ascending=[True, False]).reset_index(drop=True)


if __name__ == "__main__":
    t = build()
    print(f"{len(t)} آیتم گروه B\n")
    print(t["وضعیت"].value_counts().to_string())
    print("\nنمونه:")
    print(t.head(12).to_string(index=False))
