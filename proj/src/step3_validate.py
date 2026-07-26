# -*- coding: utf-8 -*-
"""مرحله ۳: اعتبارسنجی walk-forward — مقایسه مدل با baseline"""
import pandas as pd, numpy as np, lightgbm as lgb
from config import PROC, OUT, HORIZONS, LGB_PARAMS, BLEND_WEIGHT
from features import add_raw_features, to_ratio, FEATURES, FUTURE_KNOWN, GRP, wmape

def build_horizon(base, g, h):
    """ردیف لنگر در روز t، هدف در روز t+h ؛ فیچر تقویمی از t+h"""
    keep = ["branch","cat","date","scale"] + [c for c in FEATURES
            if c not in (["branch_c","cat_c"] + FUTURE_KNOWN)]
    s = base[keep].copy()
    tgt = g[["branch","cat","date","qty","exclude"] + FUTURE_KNOWN].copy()
    tgt["anchor"] = tgt.date - pd.Timedelta(days=h)
    d = s.merge(tgt, left_on=["branch","cat","date"],
                right_on=["branch","cat","anchor"], suffixes=("_a",""))
    d = d[d.exclude == 0]
    bad = g[g.exclude == 1][["branch","cat","date"]].assign(w=1)
    d = d.merge(bad, left_on=["branch","cat","date_a"],
                right_on=["branch","cat","date"], how="left", suffixes=("","_w"))
    d = d[d.w.isna()].copy()
    d["y"] = d.qty / d.scale
    d["branch_c"] = d.branch.astype("category")
    d["cat_c"] = d.cat.astype("category")
    return d

def main():
    g = pd.read_parquet(PROC / "daily.parquet")
    g = add_raw_features(g)
    base = to_ratio(g)

    print(f"{'افق':>5} {'تست':>7} {'مدل':>8} {'baseline':>9} {'ترکیب':>8} {'سوگیری':>8}")
    print("-" * 50)
    rows = []
    for h in HORIZONS:
        d = build_horizon(base, g, h)
        cut = d.date.max() - pd.Timedelta(days=28)
        tr, te = d[d.date <= cut], d[d.date > cut]
        m = lgb.LGBMRegressor(**LGB_PARAMS)
        m.fit(tr[FEATURES], tr.y, categorical_feature=["branch_c","cat_c"])
        p = np.clip(m.predict(te[FEATURES]), 0, None) * te.scale
        a = te.qty.values
        b = (te.r_dow * te.scale).values
        bl = BLEND_WEIGHT * p.values + (1 - BLEND_WEIGHT) * b
        r = dict(h=h, n=len(te), model=wmape(a, p), base=wmape(a, b),
                 blend=wmape(a, bl), bias=100*(p.sum()-a.sum())/a.sum())
        rows.append(r)
        print(f"{h:>5} {r['n']:>7,} {r['model']:>7.1f}% {r['base']:>8.1f}% "
              f"{r['blend']:>7.1f}% {r['bias']:>7.1f}%")

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "validation.csv", index=False)
    print(f"\n✅ ذخیره شد: outputs/validation.csv")
    print("   wMAPE = میانگین خطای وزنی (کمتر بهتر)")

if __name__ == "__main__":
    main()
