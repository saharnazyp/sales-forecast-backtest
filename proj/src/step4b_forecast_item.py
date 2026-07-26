# -*- coding: utf-8 -*-
"""مرحله ۴ب: پیش‌بینی ۲۸ روزه در سطح تک‌تک آیتم منو

گروه A (پرتردد)  -> مدل LightGBM چندافقی
گروه B (کم‌تردد) -> میانگین روز هفته ۸ هفته اخیر
"""
import pandas as pd, numpy as np, lightgbm as lgb, jdatetime
from config import PROC, OUT, HORIZONS, FORECAST_DAYS, LGB_PARAMS, BLEND_WEIGHT, USE_HIJRI
from features import FUTURE_KNOWN, wmape
from external import HIJRI_FEATS, FX_FEATS

GRP = ["branch", "item"]
LAGS = [1, 2, 3, 7, 14, 21, 28]
ROLLS = [7, 14, 28]
DOW_FA = {0:"شنبه",1:"یکشنبه",2:"دوشنبه",3:"سه‌شنبه",4:"چهارشنبه",5:"پنجشنبه",6:"جمعه"}

STATE = (["log_scale","r_dow","dow_ratio","trend","nz_28"]
         + [f"r_lag_{L}" for L in LAGS]
         + [f"r_rm_{R}" for R in ROLLS] + [f"r_rs_{R}" for R in ROLLS])
FEATURES = ["branch_c"] + FUTURE_KNOWN + STATE


def add_feats(g):
    g = g.sort_values(GRP + ["date"]).copy()
    s1 = g.groupby(GRP).qty.shift(1)
    grp = [g.branch, g.item]
    g["scale"] = (s1.groupby(grp).rolling(28, min_periods=7).mean()
                    .reset_index(level=[0,1], drop=True))
    for L in LAGS: g[f"lag_{L}"] = g.groupby(GRP).qty.shift(L)
    for R in ROLLS:
        g[f"rm_{R}"] = (s1.groupby(grp).rolling(R, min_periods=3).mean()
                          .reset_index(level=[0,1], drop=True))
        g[f"rs_{R}"] = (s1.groupby(grp).rolling(R, min_periods=3).std()
                          .reset_index(level=[0,1], drop=True))
    # نسبت روزهای غیرصفر — برای آیتم‌های پراکنده مهم است
    g["nz_28"] = ((s1 > 0).groupby(grp).rolling(28, min_periods=7).mean()
                    .reset_index(level=[0,1], drop=True))
    g["dow_mean_4"] = (g.groupby(GRP + ["dow"]).qty.shift(1)
                         .groupby([g.branch, g.item, g.dow])
                         .rolling(4, min_periods=2).mean()
                         .reset_index(level=[0,1,2], drop=True))
    g["trend"] = g.rm_7 / g.rm_28.replace(0, np.nan)
    return g


def to_ratio(g):
    g = g[g.scale.notna() & (g.scale > 0) & g.lag_28.notna()
          & g.dow_mean_4.notna()].copy()
    for L in LAGS: g[f"r_lag_{L}"] = g[f"lag_{L}"] / g.scale
    for R in ROLLS:
        g[f"r_rm_{R}"] = g[f"rm_{R}"] / g.scale
        g[f"r_rs_{R}"] = g[f"rs_{R}"] / g.scale
    g["r_dow"] = g.dow_mean_4 / g.scale
    g["dow_ratio"] = g.dow_mean_4 / g.rm_28.replace(0, np.nan)
    g["log_scale"] = np.log1p(g.scale)
    g["branch_c"] = g.branch.astype("category")
    return g


def build_h(base, g, h):
    keep = GRP + ["date", "scale"] + [c for c in FEATURES
                                       if c not in (["branch_c"] + FUTURE_KNOWN)]
    s = base[keep].copy()
    tgt = g[GRP + ["date", "qty", "exclude"] + FUTURE_KNOWN].copy()
    tgt["anchor"] = tgt.date - pd.Timedelta(days=h)
    d = s.merge(tgt, left_on=GRP + ["date"], right_on=GRP + ["anchor"],
                suffixes=("_a", ""))
    d = d[d.exclude == 0]
    bad = g[g.exclude == 1][GRP + ["date"]].assign(w=1)
    d = d.merge(bad, left_on=GRP + ["date_a"], right_on=GRP + ["date"],
                how="left", suffixes=("", "_w"))
    d = d[d.w.isna()].copy()
    d["y"] = d.qty / d.scale
    d["branch_c"] = d.branch.astype("category")
    return d


def main():
    g = pd.read_parquet(PROC / "daily_item.parquet")
    ga = g[g.tier == "A"].copy()
    g = add_feats(g)
    ga = add_feats(ga)
    base = to_ratio(ga)
    print(f"  گروه A: {base.groupby(GRP).ngroups:,} سری قابل مدل‌سازی")

    # ---------- اعتبارسنجی ----------
    print(f"\n{'افق':>5} {'تست':>8} {'مدل':>8} {'baseline':>9} {'ترکیب':>8}")
    print("-" * 44)
    for h in [7, 14, 28]:
        d = build_h(base, ga, h)
        cut = d.date.max() - pd.Timedelta(days=28)
        tr, te = d[d.date <= cut], d[d.date > cut]
        m = lgb.LGBMRegressor(**LGB_PARAMS)
        m.fit(tr[FEATURES], tr.y, categorical_feature=["branch_c"])
        p = np.clip(m.predict(te[FEATURES]), 0, None) * te.scale
        a = te.qty.values; b = (te.r_dow * te.scale).values
        bl = BLEND_WEIGHT * p.values + (1 - BLEND_WEIGHT) * b
        print(f"{h:>5} {len(te):>8,} {wmape(a,p):>7.1f}% {wmape(a,b):>8.1f}% "
              f"{wmape(a,bl):>7.1f}%")

    # ---------- آموزش نهایی ----------
    print("\nآموزش مدل نهایی...")
    models = {}
    for h in HORIZONS:
        d = build_h(base, ga, h)
        m = lgb.LGBMRegressor(**LGB_PARAMS)
        m.fit(d[FEATURES], d.y, categorical_feature=["branch_c"])
        models[h] = m

    anchor = base.date.max()
    cur = base[base.date == anchor].copy()
    ja = jdatetime.date.fromgregorian(date=anchor.date())
    print(f"مبنا: {anchor.date()} = {ja.year}/{ja.month:02d}/{ja.day:02d}")

    fut = pd.DataFrame({"date": [anchor + pd.Timedelta(days=k)
                                 for k in range(1, FORECAST_DAYS + 1)]})
    if USE_HIJRI:
        from external import add_hijri
        fut = add_hijri(fut)
    else:
        for c in HIJRI_FEATS: fut[c] = 0.0
    for c in FX_FEATS:
        fut[c] = float(g[c].iloc[-1]) if c in g.columns and len(g) else 0.0
    fut = fut.set_index("date")

    out = []
    for step in range(1, FORECAST_DAYS + 1):
        td = anchor + pd.Timedelta(days=step)
        jd_ = jdatetime.date.fromgregorian(date=td.date())
        h = min([x for x in HORIZONS if x >= step], default=max(HORIZONS))
        r = cur.copy()
        dw = (td.dayofweek + 2) % 7
        r["dow"] = dw; r["is_thu"] = int(dw == 5); r["is_fri"] = int(dw == 6)
        r["is_weekend"] = int(dw in (5, 6)); r["jd"] = jd_.day
        r["is_nowruz"] = int(jd_.month == 1 and jd_.day <= 13)
        r["is_month_end"] = int(jd_.day >= 27); r["is_month_start"] = int(jd_.day <= 3)
        for c in HIJRI_FEATS + FX_FEATS: r[c] = fut.loc[td, c]
        r["branch_c"] = r.branch.astype("category")
        p = np.clip(models[h].predict(r[FEATURES]), 0, None) * r.scale
        nb = (cur.r_dow * cur.scale).values
        fc = BLEND_WEIGHT * p.values + (1 - BLEND_WEIGHT) * nb
        out.append(pd.DataFrame({
            "branch": r.branch.values, "item": r["item"].values, "date": td,
            "jdate": f"{jd_.year}/{jd_.month:02d}/{jd_.day:02d}",
            "dow": dw, "tier": "A", "forecast_qty": np.round(fc, 2)}))
    F = pd.concat(out, ignore_index=True)

    # ---------- گروه B: میانگین روز هفته ----------
    got = set(map(tuple, F[GRP].drop_duplicates().values))
    recent = g[g.date > anchor - pd.Timedelta(days=56)]
    alive = recent.groupby(GRP).qty.sum()
    alive = alive[alive > 0].index
    miss = [k for k in alive if k not in got]
    if miss:
        dowf = recent.groupby(GRP + ["dow"]).qty.mean()
        allm = recent.groupby(GRP).qty.mean()
        add = []
        for step in range(1, FORECAST_DAYS + 1):
            td = anchor + pd.Timedelta(days=step)
            jd_ = jdatetime.date.fromgregorian(date=td.date())
            dw = (td.dayofweek + 2) % 7
            for br, ic in miss:
                v = dowf.get((br, ic, dw), np.nan)
                if pd.isna(v): v = allm.get((br, ic), 0.0)
                add.append(dict(branch=br, item=ic, date=td,
                    jdate=f"{jd_.year}/{jd_.month:02d}/{jd_.day:02d}",
                    dow=dw, tier="B", forecast_qty=round(float(np.nan_to_num(v)), 2)))
        F = pd.concat([F, pd.DataFrame(add)], ignore_index=True)
        print(f"  گروه B: {len(miss):,} آیتم با میانگین روز هفته")

    F["wk"] = ((F.date - anchor).dt.days - 1) // 7 + 1
    F["روز"] = F.dow.map(DOW_FA)
    F = F.sort_values(GRP + ["date"]).reset_index(drop=True)
    F.to_parquet(PROC / "forecast_item.parquet", index=False)

    path = OUT / "forecast_28d_item.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        d1 = F[["branch","item","jdate","روز","tier","forecast_qty"]]
        d1.columns = ["شعبه","آیتم منو","تاریخ","روز","گروه","پیش‌بینی"]
        d1.to_excel(w, sheet_name="روزانه", index=False)

        wk = F.groupby(["branch","item","wk"], as_index=False).forecast_qty.sum()
        p = wk.pivot_table(index=["branch","item"],
                           columns="wk", values="forecast_qty").round(1)
        p.columns = [f"هفته {c}" for c in p.columns]
        p["جمع ۲۸ روز"] = p.sum(axis=1).round(1)
        (p.reset_index().rename(columns={"branch":"شعبه","item":"آیتم منو"})
          .sort_values("جمع ۲۸ روز", ascending=False)
          .to_excel(w, sheet_name="هفتگی", index=False))

        tot = (F.groupby(["branch","item","tier"], as_index=False)
                 .forecast_qty.sum().round(1)
                 .sort_values("forecast_qty", ascending=False))
        tot.columns = ["شعبه","آیتم منو","گروه","جمع ۲۸ روز"]
        tot.to_excel(w, sheet_name="جمع آیتم", index=False)

        br = F.groupby(["branch","jdate","روز"], as_index=False).forecast_qty.sum()
        br.columns = ["شعبه","تاریخ","روز","پیش‌بینی"]
        br.to_excel(w, sheet_name="شعبه-روزانه", index=False)

        # خلاصه گروه B با بازه اطمینان
        try:
            from tier_b_summary import build as build_tier_b
            tb = build_tier_b(g)
            if len(tb):
                tb.to_excel(w, sheet_name="گروه B - جمع دوره", index=False)
                print(f"  شیت گروه B: {len(tb)} آیتم")
                warn = tb[tb["وضعیت"].str.startswith("⚠")]
                if len(warn):
                    print(f"    ⚠ {len(warn)} آیتم نیاز به بررسی "
                          f"({(tb['وضعیت']=='⚠ راکد — بیش از ۳ هفته بی‌فروش').sum()} راکد، "
                          f"{(tb['وضعیت']=='⚠ جدید — سابقه کوتاه').sum()} جدید)")
        except Exception as e:
            print(f"  ⚠ شیت گروه B ساخته نشد: {e}")

        # شیت بازبینی: چه نام‌هایی با هم ادغام شدند
        try:
            from clean_menu import build_menu_map
            raw = pd.read_parquet(PROC / "clean.parquet")
            lastd = raw.date.max()
            mp = build_menu_map(raw[raw.date > lastd - pd.Timedelta(days=90)],
                                verbose=False)
            aud = (pd.DataFrame({"raw": list(mp.keys()), "menu": list(mp.values())})
                     .groupby("menu").raw
                     .agg(n="nunique", names=lambda x: " | ".join(sorted(set(x))))
                     .reset_index())
            aud = aud[aud.n > 1].sort_values("n", ascending=False)
            aud.columns = ["آیتم منو", "تعداد املا", "نام‌های خام ادغام‌شده"]
            aud.to_excel(w, sheet_name="بازبینی ادغام", index=False)
        except Exception as e:
            print(f"  ⚠ شیت بازبینی ساخته نشد: {e}")

    print(f"\n✅ {len(F):,} ردیف | {F.groupby(GRP).ngroups:,} آیتم "
          f"| جمع {F.forecast_qty.sum():,.0f} پرس")
    print(f"   ذخیره شد: outputs/forecast_28d_item.xlsx")

if __name__ == "__main__":
    main()
