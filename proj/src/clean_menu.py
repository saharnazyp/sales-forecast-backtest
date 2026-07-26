# -*- coding: utf-8 -*-
"""تمیزکاری نام آیتم‌های منو و ادغام تکراری‌ها

مشکلاتی که حل می‌کند:
  ۱. حروف عربی/فارسی مخلوط  (ي/ی ، ك/ک)
  ۲. ستاره و فاصله اضافه     ("چيز برگر*" = "چيز برگر")
  ۳. نیم‌فاصله و اعراب
  ۴. جابه‌جایی کلمات          ("قوطي كوكا" = "كوكا قوطي")
  ۵. اقلام غیرمنو             (پرسنلی، بسته‌بندی، هزینه سرویس)

توجه مهم: کد کالا در داده قابل اعتماد نیست — یک کد گاهی به چند کالای
کاملاً متفاوت اختصاص یافته. برای همین گروه‌بندی بر اساس *نام نرمال‌شده*
انجام می‌شود نه item_code.
"""
import pandas as pd, numpy as np, re

# ---------------------------------------------------------------- نرمال‌سازی
# برای نمایش: آ حفظ می‌شود
_AR2FA = str.maketrans({"ي":"ی","ك":"ک","ة":"ه","ۀ":"ه","ؤ":"و","ئ":"ی","ء":""})
# فقط برای تطبیق: همه شکل‌های الف یکی می‌شوند
_FOLD = str.maketrans({"آ":"ا","أ":"ا","إ":"ا"})
_DIAC = re.compile(r"[\u064B-\u0652\u0670]")
_PUNCT = re.compile(r"[*#()\[\]{}\-_/\\.,،؛:؟!\"'`~]+")
_DIGITS = re.compile(r"\d+")
_SPACE = re.compile(r"\s+")

# کلمات بی‌اثر در تشخیص یکسانی
_STOP = {"عدد","عددي","عددی","پرس","سرو","جديد","جدید"}

# اعدادی که اندازه/تعداد را مشخص می‌کنند و باید حفظ شوند
# (سوخاری ۳تکه با سوخاری ۹تکه یکی نیست)
_SIZE_CTX = ["تکه","تيکه","تیکه","نفره","عددي","عددی","سي سي","سی سی",
             "ميلي","میلی","گرم","کيلو","کیلو","ليتر","لیتر","اينچ","اینچ",
             "لايه","لایه","اسکوپ","قلو"]

def normalize(s):
    """نرمال‌سازی پایه — برای نمایش"""
    if not isinstance(s, str):
        return ""
    s = s.replace("\u200c", " ").translate(_AR2FA)
    s = _DIAC.sub("", s)
    s = _PUNCT.sub(" ", s)
    s = _SPACE.sub(" ", s).strip()
    return s

def canonical_key(s):
    """کلید یکسانی — برای ادغام. اعداد و کلمات بی‌اثر حذف، واژه‌ها مرتب می‌شوند
    تا 'قوطي كوكا' و 'كوكا قوطي' یکی شوند."""
    s = normalize(s).translate(_FOLD)
    # عدد فقط وقتی حذف می‌شود که کنارش واحد اندازه نباشد
    has_size = any(k in s for k in _SIZE_CTX)
    if not has_size:
        s = _DIGITS.sub(" ", s)
    else:
        # عدد چسبیده به واحد را جدا کن تا "3تکه" و "3 تکه" یکی شوند
        s = re.sub(r"(\d+)\s*", r"\1 ", s)
    words = [w for w in s.split() if w not in _STOP and len(w) > 1 or w.isdigit()]
    return " ".join(sorted(words))

# ---------------------------------------------------------------- اقلام غیرمنو
NON_MENU_KEYWORDS = [
    "پرسنلي","پرسنلی","پرسنل",
    "تست کالا","تست كالا","ضايعات","ضایعات","مهمان","تعارف",
    "بسته بندي","بسته بندی","بيرون بر","بیرون بر","نايلون","نایلون",
    "کيسه","کیسه","جعبه","ظرف يکبار","ظرف یکبار",
    "هزينه","هزینه","سرويس","سرویس","ماليات","مالیات","تخفيف","تخفیف",
    "کارد","چنگال","قاشق","دستمال","شمع",
]

def is_non_menu(name):
    n = normalize(name)
    return any(k in n for k in NON_MENU_KEYWORDS)

# ---------------------------------------------------------------- ساخت نگاشت
def build_menu_map(df, qty_col="qty", name_col="item_name",
                   drop_non_menu=True, verbose=True):
    """نگاشت نام خام -> نام تمیز (منو).

    برای هر گروه، پرفروش‌ترین املا به عنوان نام رسمی انتخاب می‌شود.
    """
    t = (df.groupby(name_col, observed=True)[qty_col].sum()
           .reset_index().rename(columns={name_col: "raw", qty_col: "q"}))
    t["raw"] = t.raw.astype(str)
    t["key"] = t.raw.map(canonical_key)
    t = t[t.key.str.len() > 0]

    # نام رسمی = پرفروش‌ترین املا در هر گروه، پس از نرمال‌سازی ظاهری
    best = (t.sort_values("q", ascending=False)
              .groupby("key").raw.first().map(normalize))
    t["menu_item"] = t.key.map(best)
    t["non_menu"] = t.raw.map(is_non_menu)

    if verbose:
        merged = t.groupby("key").raw.nunique()
        print(f"  نام خام: {len(t):,}  ->  آیتم منو: {t.menu_item.nunique():,}")
        print(f"  گروه‌های ادغام‌شده: {(merged > 1).sum():,}")
        if drop_non_menu:
            nq = t.loc[t.non_menu, "q"].sum(); tq = t.q.sum()
            print(f"  اقلام غیرمنو حذف‌شده: {t.non_menu.sum():,} نام "
                  f"({100*nq/max(tq,1):.1f}% حجم)")

    if drop_non_menu:
        t = t[~t.non_menu]
    return t.set_index("raw").menu_item.to_dict()


def clean_branch(s):
    """نام شعبه: فقط یکسان‌سازی حروف عربی/فارسی و فاصله"""
    return normalize(s)


def apply_menu_map(df, mapping, name_col="item_name", keep_unmapped=False):
    """نام تمیز را به دیتافریم اضافه می‌کند و اقلام غیرمنو را حذف می‌کند"""
    df = df.copy()
    df["menu_item"] = df[name_col].astype(str).map(mapping)
    if not keep_unmapped:
        df = df[df.menu_item.notna()]
    if "branch" in df.columns:
        df["branch"] = df.branch.astype(str).map(clean_branch)
    return df


if __name__ == "__main__":
    import sys
    from config import PROC
    d = pd.read_parquet(PROC / "clean.parquet")
    last = d.date.max()
    rec = d[d.date > last - pd.Timedelta(days=90)]
    m = build_menu_map(rec)
    out = apply_menu_map(rec, m)
    top = out.groupby("menu_item").qty.sum().sort_values(ascending=False)
    print(f"\nپرفروش‌ترین ۲۵ آیتم منو (۹۰ روز اخیر):")
    print(top.head(25).round(0).to_string())
