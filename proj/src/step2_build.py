# -*- coding: utf-8 -*-
"""مرحله ۲: ساخت پنل روزانه شعبه × دسته + فیچرهای تقویم شمسی"""
import pandas as pd, numpy as np, jdatetime
from config import PROC, WAR_START, WAR_END, USE_HIJRI
from external import add_hijri, add_fx, add_anomalies
from categorize import classify

def main():
    df = pd.read_parquet(PROC / "clean.parquet")
    df["rev"] = df.qty * df.price
    df["cat"] = df.item_name.astype(str).map(classify)

    d = (df.groupby(["branch", "cat", "date"], observed=True)
           .agg(qty=("qty", "sum"), rev=("rev", "sum")).reset_index())

    # گرید کامل: هر سری از اولین تا آخرین روز فعالیتش
    alive = d.groupby(["branch", "cat"]).date.agg(["min", "max"]).reset_index()
    rows = []
    for _, r in alive.iterrows():
        for dt in pd.date_range(r["min"], r["max"]):
            rows.append((r["branch"], r["cat"], dt))
    g = pd.DataFrame(rows, columns=["branch", "cat", "date"])
    g = g.merge(d, on=["branch", "cat", "date"], how="left")
    g["qty"] = g.qty.fillna(0); g["rev"] = g.rev.fillna(0)

    # ---- تقویم شمسی ----
    jj = g.date.dt.date.map(lambda x: jdatetime.date.fromgregorian(date=x))
    g["jy"] = jj.map(lambda x: x.year)
    g["jm"] = jj.map(lambda x: x.month)
    g["jd"] = jj.map(lambda x: x.day)
    g["jdoy"] = jj.map(lambda x: (x - jdatetime.date(x.year, 1, 1)).days + 1)
    g["dow"] = (g.date.dt.dayofweek + 2) % 7        # 0=شنبه ... 6=جمعه
    g["is_thu"] = (g.dow == 5).astype(int)
    g["is_fri"] = (g.dow == 6).astype(int)
    g["is_weekend"] = g.dow.isin([5, 6]).astype(int)
    g["is_nowruz"] = ((g.jm == 1) & (g.jd <= 13)).astype(int)
    g["is_month_end"] = (g.jd >= 27).astype(int)
    g["is_month_start"] = (g.jd <= 3).astype(int)

    g["jstr"] = (g.jy.astype(str) + "/" + g.jm.astype(str).str.zfill(2)
                 + "/" + g.jd.astype(str).str.zfill(2))
    g["is_war"] = ((g.jstr >= WAR_START) & (g.jstr <= WAR_END)).astype(int)

    # ---- متغیرهای بیرونی ----
    if USE_HIJRI:
        g = add_hijri(g)
        print(f"  ✅ قمری: {g.is_ramadan.sum():,} روز رمضان، "
              f"{g.is_eid_fitr.sum():,} روز عید فطر")
    else:
        from external import HIJRI_FEATS
        for c in HIJRI_FEATS: g[c] = 0.0

    g, has_fx = add_fx(g)
    if not has_fx:
        print("  ℹ نرخ ارز نیست (اختیاری) — فایل data/raw/usd_rate.csv")

    g, n_anom = add_anomalies(g)
    if not n_anom:
        print("  ℹ بازه غیرعادی ثبت نشده — فایل data/raw/anomalies.csv")

    # جنگ و بازه‌های غیرعادی هر دو از آموزش خارج می‌شوند
    g["exclude"] = ((g.is_war == 1) | (g.is_anomaly == 1)).astype(int)

    # شاخص قیمت داخلی (فقط برای گزارش، نه فیچر مدل)
    pi = df.groupby("date").apply(lambda x: x.rev.sum() / x.qty.sum(), include_groups=False)
    pi = (pi / pi.iloc[0] * 100).rolling(7, min_periods=1).mean()
    g["price_idx"] = g.date.map(pi)

    g = g.sort_values(["branch", "cat", "date"]).reset_index(drop=True)
    g.to_parquet(PROC / "daily.parquet", index=False)
    print(f"✅ {len(g):,} ردیف روزانه | {g.groupby(['branch','cat']).ngroups} سری")
    print(f"   حذف‌شده از آموزش: {g.exclude.sum():,} ردیف "
          f"(جنگ {g.is_war.sum():,} + غیرعادی {g.is_anomaly.sum():,})")

if __name__ == "__main__":
    main()
