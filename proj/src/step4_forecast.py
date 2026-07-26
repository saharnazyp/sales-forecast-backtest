# -*- coding: utf-8 -*-
"""مرحله ۴: آموزش نهایی روی همه داده + پیش‌بینی ۲۸ روز آینده -> اکسل"""
import pandas as pd, numpy as np, lightgbm as lgb, jdatetime
from config import PROC, OUT, HORIZONS, FORECAST_DAYS, LGB_PARAMS, BLEND_WEIGHT
from features import add_raw_features, to_ratio, FEATURES, FUTURE_KNOWN, GRP
from config import USE_HIJRI
from external import HIJRI_FEATS, FX_FEATS
from step3_validate import build_horizon

DOW_FA = {0:"شنبه",1:"یکشنبه",2:"دوشنبه",3:"سه‌شنبه",4:"چهارشنبه",5:"پنجشنبه",6:"جمعه"}

def main():
    g = pd.read_parquet(PROC / "daily.parquet")
    g = add_raw_features(g)
    base = to_ratio(g)

    print("آموزش مدل‌ها...")
    models = {}
    for h in HORIZONS:
        d = build_horizon(base, g, h)
        m = lgb.LGBMRegressor(**LGB_PARAMS)
        m.fit(d[FEATURES], d.y, categorical_feature=["branch_c","cat_c"])
        models[h] = m
        print(f"  افق {h:>2} روز — {len(d):,} نمونه")

    anchor = base.date.max()
    cur = base[base.date == anchor].copy()
    ja = jdatetime.date.fromgregorian(date=anchor.date())
    print(f"\nمبنا: {anchor.date()} = {ja.year}/{ja.month:02d}/{ja.day:02d}")

    # ---- فیچرهای بیرونی برای روزهای آینده ----
    fut = pd.DataFrame({"date": [anchor + pd.Timedelta(days=k)
                                 for k in range(1, FORECAST_DAYS + 1)]})
    if USE_HIJRI:
        from external import add_hijri
        fut = add_hijri(fut)
        nr = int(fut.is_ramadan.sum())
        if nr: print(f"  رمضان در افق پیش‌بینی: {nr} روز")
    else:
        for c in HIJRI_FEATS: fut[c] = 0.0
    # ارز آینده نامعلوم است -> آخرین مقدار شناخته‌شده ثابت نگه داشته می‌شود
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
        r["dow"] = dw
        r["is_thu"] = int(dw == 5); r["is_fri"] = int(dw == 6)
        r["is_weekend"] = int(dw in (5, 6))
        r["jd"] = jd_.day
        r["is_nowruz"] = int(jd_.month == 1 and jd_.day <= 13)
        r["is_month_end"] = int(jd_.day >= 27)
        r["is_month_start"] = int(jd_.day <= 3)
        for c in HIJRI_FEATS + FX_FEATS:
            r[c] = fut.loc[td, c]
        r["branch_c"] = r.branch.astype("category")
        r["cat_c"] = r.cat.astype("category")
        p = np.clip(models[h].predict(r[FEATURES]), 0, None) * r.scale
        nb = (cur.r_dow * cur.scale).values
        fc = BLEND_WEIGHT * p.values + (1 - BLEND_WEIGHT) * nb
        out.append(pd.DataFrame({
            "branch": r.branch.values, "cat": r.cat.values, "date": td,
            "jdate": f"{jd_.year}/{jd_.month:02d}/{jd_.day:02d}",
            "dow": dw, "forecast_qty": np.round(fc, 1)}))
    F = pd.concat(out, ignore_index=True)

    # سری‌های کم‌تراکم که فیچر کافی ندارند
    recent = g[g.date > anchor - pd.Timedelta(days=28)]
    alive = recent.groupby(GRP).qty.sum()
    alive = alive[alive > 0].index
    got = set(map(tuple, F[GRP].drop_duplicates().values))
    miss = [k for k in alive if k not in got]
    if miss:
        dowf = recent.groupby(GRP + ["dow"]).qty.mean()
        add = []
        for step in range(1, FORECAST_DAYS + 1):
            td = anchor + pd.Timedelta(days=step)
            jd_ = jdatetime.date.fromgregorian(date=td.date())
            dw = (td.dayofweek + 2) % 7
            for br, ct in miss:
                v = dowf.get((br, ct, dw), np.nan)
                if pd.isna(v):
                    sub = recent[(recent.branch == br) & (recent.cat == ct)]
                    v = sub.qty.mean() if len(sub) else 0
                add.append(dict(branch=br, cat=ct, date=td,
                    jdate=f"{jd_.year}/{jd_.month:02d}/{jd_.day:02d}",
                    dow=dw, forecast_qty=round(float(np.nan_to_num(v)), 1)))
        F = pd.concat([F, pd.DataFrame(add)], ignore_index=True)
        print(f"  fallback برای {len(miss)} سری کم‌تراکم")

    F["wk"] = ((F.date - anchor).dt.days - 1) // 7 + 1
    F["روز"] = F.dow.map(DOW_FA)
    F = F.sort_values(GRP + ["date"]).reset_index(drop=True)
    F.to_parquet(PROC / "forecast.parquet", index=False)

    path = OUT / "forecast_28d.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        d1 = F[["branch","cat","jdate","روز","forecast_qty"]]
        d1.columns = ["شعبه","دسته","تاریخ","روز","پیش‌بینی"]
        d1.to_excel(w, sheet_name="روزانه", index=False)

        d2 = F.groupby(["branch","jdate","روز"], as_index=False).forecast_qty.sum()
        d2.columns = ["شعبه","تاریخ","روز","پیش‌بینی"]
        d2.to_excel(w, sheet_name="شعبه-روزانه", index=False)

        wk = F.groupby(["branch","cat","wk"], as_index=False).forecast_qty.sum()
        p = wk.pivot_table(index=["branch","cat"], columns="wk",
                           values="forecast_qty").round(0)
        p.columns = [f"هفته {c}" for c in p.columns]
        p["جمع"] = p.sum(axis=1)
        (p.reset_index().rename(columns={"branch":"شعبه","cat":"دسته"})
          .to_excel(w, sheet_name="هفتگی", index=False))

        dw_ = F.groupby(["branch","روز","dow"], as_index=False).forecast_qty.mean()
        dp = dw_.pivot(index="branch", columns="روز", values="forecast_qty").round(0)
        dp = dp[[DOW_FA[i] for i in range(7) if DOW_FA[i] in dp.columns]]
        (dp.reset_index().rename(columns={"branch":"شعبه"})
          .to_excel(w, sheet_name="الگوی هفته", index=False))

    print(f"\n✅ {len(F):,} ردیف | {F.groupby(GRP).ngroups} سری | جمع {F.forecast_qty.sum():,.0f} پرس")
    print(f"   ذخیره شد: outputs/forecast_28d.xlsx")

if __name__ == "__main__":
    main()
