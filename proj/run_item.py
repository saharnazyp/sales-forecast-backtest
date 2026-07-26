# -*- coding: utf-8 -*-
"""پیش‌بینی در سطح تک‌تک آیتم‌های منو

اگر قبلاً run_all.py را زده‌ای، مرحله ۱ لازم نیست دوباره اجرا شود.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import PROC

import step1_extract, step2b_build_item, step4b_forecast_item

if __name__ == "__main__":
    t0 = time.time()
    if not (PROC / "clean.parquet").exists():
        print("=" * 55); print("۱/۳ خواندن اکسل"); print("=" * 55)
        step1_extract.main()
    else:
        print("ℹ clean.parquet موجود است — مرحله ۱ رد شد")

    print("\n" + "=" * 55); print("۲/۳ ساخت پنل آیتم"); print("=" * 55)
    step2b_build_item.main()

    print("\n" + "=" * 55); print("۳/۳ پیش‌بینی آیتم"); print("=" * 55)
    step4b_forecast_item.main()

    print(f"\n{'='*55}\n🎉 تمام شد در {time.time()-t0:.0f} ثانیه")
    print("   خروجی: outputs/forecast_28d_item.xlsx")
