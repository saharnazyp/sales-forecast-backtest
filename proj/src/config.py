# -*- coding: utf-8 -*-
"""تنظیمات پروژه - همه مسیرها و پارامترها اینجا"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
for p in (RAW, PROC, OUT): p.mkdir(parents=True, exist_ok=True)

# ---- فایل ورودی: اسم فایل اکسل فروش را اینجا بگذار ----
SALES_FILE = RAW / "Sales_Report-Ver01.xlsb"
SHEETS = ["1404-1", "1404-2", "1405"]   # شیت‌های تراکنش

# ستون‌های موردنیاز (شماره ستون در فایل خام، از صفر)
COL_MAP = {2:"jdate", 7:"branch", 8:"item_code", 9:"item_name",
           10:"qty", 11:"price", 15:"discount", 19:"channel"}

# ---- بازه جنگ (از آموزش حذف می‌شود) ----
WAR_START = "1404/12/09"
WAR_END   = "1405/01/19"

# ---- فایل‌های اختیاری ----
# نرخ ارز: CSV یا Excel با ستون تاریخ (شمسی یا میلادی) و ستون قیمت
#   نام ستون تاریخ: date / تاریخ / jdate      نام ستون قیمت: price / قیمت / نرخ / close
FX_FILE = RAW / "usd_rate.csv"

# بازه‌های غیرعادی: CSV با ستون‌های start,end,branch,reason
#   branch = ALL یعنی همه شعبه‌ها
ANOMALY_FILE = RAW / "anomalies.csv"

# ---- سطح آیتم منو ----
# حداقل روز فعال در ITEM_LOOKBACK روز اخیر تا آیتم وارد مدل ML شود
ITEM_MIN_ACTIVE_DAYS = 30
ITEM_LOOKBACK = 90

# استفاده از فیچرهای قمری (رمضان، عید فطر، محرم)
USE_HIJRI = False   # پیش‌فرض خاموش — با ۱۵ ماه داده ضرر می‌زند (README را ببین)

# ---- پارامترهای مدل ----
HORIZONS = [1, 3, 7, 14, 21, 28]
FORECAST_DAYS = 28
BLEND_WEIGHT = 0.5      # وزن مدل در ترکیب با baseline

LGB_PARAMS = dict(
    n_estimators=400, learning_rate=0.03, num_leaves=15,
    min_child_samples=80, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, random_state=42,
)
