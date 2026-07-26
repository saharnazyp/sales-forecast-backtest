# -*- coding: utf-8 -*-
"""ساخت فیچرهای مدل - مشترک بین آموزش و پیش‌بینی"""
import pandas as pd, numpy as np

GRP = ["branch", "cat"]
LAGS = [1, 2, 3, 7, 14, 21, 28]
ROLLS = [7, 14, 28]

CAL_FEATS = ["dow", "is_thu", "is_fri", "is_weekend", "jd",
             "is_nowruz", "is_month_end", "is_month_start"]

# فیچرهای بیرونی — در step2 ساخته می‌شوند
from external import HIJRI_FEATS, FX_FEATS
EXT_FEATS = HIJRI_FEATS + FX_FEATS

# فیچرهایی که مقدارشان در روز هدف (t+h) معلوم است، نه روز لنگر
FUTURE_KNOWN = CAL_FEATS + EXT_FEATS

STATE_FEATS = (["log_scale", "r_dow", "dow_ratio", "trend"]
               + [f"r_lag_{L}" for L in LAGS]
               + [f"r_rm_{R}" for R in ROLLS]
               + [f"r_rs_{R}" for R in ROLLS])

FEATURES = ["branch_c", "cat_c"] + FUTURE_KNOWN + STATE_FEATS


def add_raw_features(g):
    """lag / rolling خام روی پنل روزانه"""
    g = g.sort_values(GRP + ["date"]).copy()
    s1 = g.groupby(GRP).qty.shift(1)

    g["scale"] = (s1.groupby([g.branch, g.cat])
                    .rolling(28, min_periods=7).mean()
                    .reset_index(level=[0, 1], drop=True))
    for L in LAGS:
        g[f"lag_{L}"] = g.groupby(GRP).qty.shift(L)
    for R in ROLLS:
        g[f"rm_{R}"] = (s1.groupby([g.branch, g.cat])
                          .rolling(R, min_periods=3).mean()
                          .reset_index(level=[0, 1], drop=True))
        g[f"rs_{R}"] = (s1.groupby([g.branch, g.cat])
                          .rolling(R, min_periods=3).std()
                          .reset_index(level=[0, 1], drop=True))
    # میانگین ۴ هفته اخیرِ همان روز هفته  (baseline قوی)
    g["dow_mean_4"] = (g.groupby(GRP + ["dow"]).qty.shift(1)
                         .groupby([g.branch, g.cat, g.dow])
                         .rolling(4, min_periods=2).mean()
                         .reset_index(level=[0, 1, 2], drop=True))
    g["trend"] = g.rm_7 / g.rm_28.replace(0, np.nan)
    return g


def to_ratio(g):
    """نرمال‌سازی نسبت به سطح خود سری — کلید کار با سری‌های بزرگ و کوچک همزمان"""
    g = g[g.scale.notna() & (g.scale > 0)
          & g.lag_28.notna() & g.dow_mean_4.notna()].copy()
    for L in LAGS:
        g[f"r_lag_{L}"] = g[f"lag_{L}"] / g.scale
    for R in ROLLS:
        g[f"r_rm_{R}"] = g[f"rm_{R}"] / g.scale
        g[f"r_rs_{R}"] = g[f"rs_{R}"] / g.scale
    g["r_dow"] = g.dow_mean_4 / g.scale
    g["dow_ratio"] = g.dow_mean_4 / g.rm_28.replace(0, np.nan)
    g["log_scale"] = np.log1p(g.scale)
    g["branch_c"] = g.branch.astype("category")
    g["cat_c"] = g.cat.astype("category")
    return g


def wmape(actual, pred):
    actual = np.asarray(actual, float); pred = np.asarray(pred, float)
    return 100 * np.abs(actual - pred).sum() / max(np.abs(actual).sum(), 1)
