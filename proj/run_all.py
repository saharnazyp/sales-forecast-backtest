# -*- coding: utf-8 -*-
"""اجرای کل خط لوله - همین فایل را Run کن"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import step1_extract, step2_build, step3_validate, step4_forecast

STEPS = [("۱/۴ خواندن اکسل", step1_extract),
         ("۲/۴ ساخت پنل روزانه", step2_build),
         ("۳/۴ اعتبارسنجی", step3_validate),
         ("۴/۴ پیش‌بینی", step4_forecast)]

if __name__ == "__main__":
    t0 = time.time()
    for name, mod in STEPS:
        print(f"\n{'='*55}\n{name}\n{'='*55}")
        mod.main()
    print(f"\n{'='*55}\n🎉 تمام شد در {time.time()-t0:.0f} ثانیه")
    print("   خروجی: outputs/forecast_28d.xlsx")
