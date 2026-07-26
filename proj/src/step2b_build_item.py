# -*- coding: utf-8 -*-
"""مرحله ۲ب: پنل روزانه در سطح تک‌تک آیتم منو (شعبه × کد کالا)

آیتم‌ها بر اساس تراکم فروش به دو گروه تقسیم می‌شوند:
  A) پرتردد  -> مدل ML
  B) کم‌تردد -> میانگین ساده روز هفته
"""
import pandas as pd, numpy as np, jdatetime
from config import PROC, WAR_START, WAR_END, USE_HIJRI, ITEM_MIN_ACTIVE_DAYS, ITEM_LOOKBACK
from external import add_hijri, add_fx, add_anomalies, HIJRI_FEATS
from clean_menu import build_menu_map, apply_menu_map

def main():
    df = pd.read_parquet(PROC / "clean.parquet")
    df["rev"] = df.qty * df.price

    # ---- تمیزکاری نام منو و ادغام تکراری‌ها ----
    last = df.date.max()
    mp = build_menu_map(df[df.date > last - pd.Timedelta(days=ITEM_LOOKBACK)])
    df = apply_menu_map(df, mp)
    df["item"] = df.menu_item

    last = df.date.max()
    rec = df[df.date > last - pd.Timedelta(days=ITEM_LOOKBACK)]
    act = rec.groupby(["branch", "item"], observed=True).date.nunique()
    live = act.index  # فقط آیتم‌هایی که اخیراً فروش داشته‌اند

    tier_a = set(act[act >= ITEM_MIN_ACTIVE_DAYS].index)
    print(f"  آیتم فعال (اخیر): {len(live):,}")
    print(f"  گروه A (مدل ML، >={ITEM_MIN_ACTIVE_DAYS} روز فعال): {len(tier_a):,}")
    print(f"  گروه B (میانگین روز هفته): {len(live)-len(tier_a):,}")

    d = (df.groupby(["branch", "item", "date"], observed=True)
           .agg(qty=("qty", "sum"), rev=("rev", "sum")).reset_index())
    d = d[d.set_index(["branch", "item"]).index.isin(live)]

    # گرید کامل روی بازه فعالیت هر آیتم
    alive = d.groupby(["branch", "item"]).date.agg(["min", "max"]).reset_index()
    rows = []
    for _, r in alive.iterrows():
        for dt in pd.date_range(r["min"], r["max"]):
            rows.append((r["branch"], r["item"], dt))
    g = pd.DataFrame(rows, columns=["branch", "item", "date"])
    g = g.merge(d, on=["branch", "item", "date"], how="left")
    g["qty"] = g.qty.fillna(0); g["rev"] = g.rev.fillna(0)

    g["tier"] = np.where(
        pd.MultiIndex.from_arrays([g.branch, g.item]).isin(tier_a), "A", "B")

    # ---- تقویم شمسی ----
    jj = g.date.dt.date.map(lambda x: jdatetime.date.fromgregorian(date=x))
    g["jy"] = jj.map(lambda x: x.year); g["jm"] = jj.map(lambda x: x.month)
    g["jd"] = jj.map(lambda x: x.day)
    g["dow"] = (g.date.dt.dayofweek + 2) % 7
    g["is_thu"] = (g.dow == 5).astype(int); g["is_fri"] = (g.dow == 6).astype(int)
    g["is_weekend"] = g.dow.isin([5, 6]).astype(int)
    g["is_nowruz"] = ((g.jm == 1) & (g.jd <= 13)).astype(int)
    g["is_month_end"] = (g.jd >= 27).astype(int)
    g["is_month_start"] = (g.jd <= 3).astype(int)
    g["jstr"] = (g.jy.astype(str) + "/" + g.jm.astype(str).str.zfill(2)
                 + "/" + g.jd.astype(str).str.zfill(2))
    g["is_war"] = ((g.jstr >= WAR_START) & (g.jstr <= WAR_END)).astype(int)

    if USE_HIJRI:
        g = add_hijri(g)
    else:
        for c in HIJRI_FEATS: g[c] = 0.0
    g, _ = add_fx(g)
    g, n_anom = add_anomalies(g)
    g["exclude"] = ((g.is_war == 1) | (g.is_anomaly == 1)).astype(int)

    g = g.sort_values(["branch", "item", "date"]).reset_index(drop=True)
    g.to_parquet(PROC / "daily_item.parquet", index=False)
    print(f"✅ {len(g):,} ردیف | {g.groupby(['branch','item']).ngroups:,} سری")
    print(f"   حذف‌شده از آموزش: {g.exclude.sum():,}")

if __name__ == "__main__":
    main()
