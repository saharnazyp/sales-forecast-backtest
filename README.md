# sales-forecast-backtest
Time series sales forecasting — model training, evaluation, and validation against actual sales
# Restaurant Demand Forecasting

Production forecasting pipeline for a 13-branch restaurant chain. Ingests 2.2M transaction lines, reconstructs a clean menu taxonomy, and produces 28-day demand forecasts at two granularities for staffing and procurement planning.

Built around the Persian (Jalali) calendar, with explicit handling for a 41-day operational disruption in the training window.

```
2.2M transactions  →  clean panel  →  direct multi-horizon LightGBM  →  28-day forecast
   457 days              1,150 series        6 models per level          Excel output
```

---

## Results

| Horizon | Category level | Item level | Naive baseline |
|---|---|---|---|
| 7 days | **21.0%** | 33.2% | 22.4% |
| 14 days | **21.6%** | 34.2% | 22.9% |
| 28 days | **24.8%** | 37.1% | 27.9% |

wMAPE on a held-out 28-day window; lower is better. Baseline is a same-weekday trailing mean, which is a strong reference for this domain.

**Forecast bias is under 1% at short horizons** and −8% at 28 days, so aggregate planning numbers are usable directly rather than needing a correction factor.

### Findings that changed the modeling approach

Four results from the exploratory phase materially shaped the design:

**Revenue growth was entirely price, not volume.** Comparing matched months year-over-year: units +10.2%, nominal revenue +103%, implied unit price +84.6%. Real revenue was flat. Any model trained on nominal figures would have projected phantom growth.

**A menu-level price index can be derived from the transaction data itself.** Implied unit price rose 81% over 15 months, monthly resolution, no external source needed — more precise than a national CPI for this purpose since it measures the actual basket.

**Decline was traffic loss, not basket shrinkage.** Decomposing affected branches into visits versus basket composition: items-per-basket held flat and real basket value *rose* 15–21%. Price-sensitive customers churned while high-spend customers stayed. This is a customer-acquisition problem, not a menu or pricing problem — a distinction the raw revenue numbers obscured.

**Product codes are unreliable.** The same `item_code` maps to unrelated products (`21060052` is both a bottled malt drink and a Caesar dressing). Grouping had to move to normalized names.

---

## Architecture

### Pipeline

```
                    Sales_Report.xlsb  (150 MB, 3 sheets)
                              │
                    ┌─────────▼─────────┐
      step1_extract │  chunked parse    │  200k-row chunks, per-sheet
                    │  Jalali→Gregorian │  bounded memory footprint
                    └─────────┬─────────┘
                              │  clean.parquet  (2.2M rows)
                    ┌─────────┴─────────┐
                    │                   │
         ┌──────────▼──────────┐  ┌─────▼──────────────────┐
         │ step2_build         │  │ step2b_build_item      │
         │ branch × category   │  │ clean_menu → dedupe    │
         │ 21 rule-based cats  │  │ branch × menu item     │
         └──────────┬──────────┘  └─────┬──────────────────┘
                    │                   │
                    │   calendar · lags · rolling stats
                    │   disruption windows excluded
                    │                   │
         ┌──────────▼──────────┐  ┌─────▼──────────────────┐
         │ step3_validate      │  │ tier split A / B       │
         │ walk-forward        │  │ A → ML   B → dow mean  │
         └──────────┬──────────┘  └─────┬──────────────────┘
                    │                   │
         ┌──────────▼──────────┐  ┌─────▼──────────────────┐
         │ step4_forecast      │  │ step4b_forecast_item   │
         │ 6 horizon models    │  │ 6 horizon models       │
         │ blend with baseline │  │ + sparse fallback      │
         └──────────┬──────────┘  └─────┬──────────────────┘
                    │                   │
              forecast_28d.xlsx   forecast_28d_item.xlsx
```

### Module responsibilities

| Module | Role |
|---|---|
| `config.py` | Single source of truth: paths, disruption windows, hyperparameters, tier thresholds |
| `clean_menu.py` | Name normalization, canonical-key deduplication, non-menu filtering |
| `categorize.py` | Rule-based item → category assignment (21 categories, 21 keyword groups) |
| `features.py` | Feature construction shared by training and inference — prevents train/serve skew |
| `external.py` | Optional signals: Hijri calendar, FX rates, user-declared anomaly windows |
| `step*.py` | Pipeline stages, each independently runnable and idempotent |

Training and inference import the same `FEATURES` list and the same transform functions. There is no separate serving path to drift out of sync.

### Feature design

Three groups, 31 features at category level:

```
identity     branch, category                       categorical
calendar     dow, thu/fri/weekend flags, jday,      known at forecast time
             nowruz, month start/end
state        log_scale, 7 lag ratios, 3 rolling     derived from anchor date
             means, 3 rolling stds, dow ratio,
             trend, nonzero ratio (item level)
```

Calendar features are computed for the **target** date; state features come from the **anchor** date. This separation is what makes direct multi-horizon prediction correct — the model always sees exactly what would be available at inference time.

---

## Modeling decisions

### Ratio target

Series span three orders of magnitude — 3,630 units/day down to under 10. Training on raw counts lets large series dominate the loss and small series contribute nothing.

The target is `qty ÷ trailing-28-day mean`, with predictions rescaled afterwards. This normalizes every series onto a comparable scale and lets a single global model serve all of them.

It also absorbs inflation for free: a trailing mean tracks the price level, so nominal drift never enters the target.

### Direct multi-horizon

A separate model per horizon (1, 3, 7, 14, 21, 28 days), each mapping anchor-date state directly to a target `h` days out.

The alternative — recursive prediction, feeding forecasts back as history — compounds error multiplicatively. Direct models degrade linearly instead: 21% at h=1 to 25% at h=28, with no discontinuity.

### Excluding trending features

An early iteration included a price index and day-of-year sin/cos terms. Both became top features by importance and drove **wMAPE to 108%** at the 56-day horizon.

The cause is structural: gradient-boosted trees partition on thresholds and cannot extrapolate past their training range. A monotonically increasing feature guarantees that every out-of-sample value falls in the terminal bin, collapsing the model to a constant.

Removing them restored ~20%. The lesson generalizes — any feature with an unbounded trend is unsafe in a tree ensemble unless differenced first.

### Disruption handling

A 41-day war window (9 Esfand 1404 – 19 Farvardin 1405) removed approximately 232,000 units of demand. Branch-level impact ranged from −58% (delivery-oriented format) to −100% (mall locations that closed entirely).

These days are **excluded** from training and evaluation rather than flagged. A single occurrence provides no learnable signal; including it teaches the model that a particular calendar month is catastrophic. Both target rows and anchor rows are filtered, so no contaminated feature vector reaches training.

Additional windows are user-declarable via `anomalies.csv` without touching code.

### Two-tier item handling

At item granularity, demand is highly intermittent. Items are split by activity:

- **Tier A** — ≥30 active days in the trailing 90 → LightGBM. ~650 items, **86% of volume**
- **Tier B** — everything else → day-of-week mean over 8 weeks. ~540 items

Machine learning adds nothing for series that are mostly zeros. The tier threshold is configurable, and the output labels every row with its tier so downstream consumers know the provenance.

---

## Menu reconstruction

Raw item names fragment the same product across multiple spellings:

| Issue | Example |
|---|---|
| Arabic/Persian glyph variants | `نيويورك فرايز` vs `نيويورک فرايز` |
| Trailing markers and whitespace | `چيز برگر*` vs `چيز برگر ` |
| Token order | `قوطي كوكا` vs `كوكا قوطي` |
| Non-menu lines | staff meals, packaging, service charges |

`clean_menu.py` resolves these with a canonical-key strategy: normalize glyphs, strip punctuation and diacritics, drop semantically empty tokens, then sort the remaining tokens. Order-independent matching falls out naturally.

**1,163 raw names → 753 menu items**, 339 merge groups, 2.1% of volume classified as non-menu and dropped. The canonical label for each group is its highest-volume spelling.

Size-bearing numerals are preserved when adjacent to a unit keyword (`تکه`, `سی سی`, `نفره`, `گرم`), so `سوخاری 3تکه` and `سوخاری 9تکه` stay distinct while `آب پرتقال` and `آب پرتقال*` merge. An earlier revision stripped all digits and incorrectly collapsed five portion sizes into one.

Every merge group is exported to a **Merge audit** sheet for human review.

---

## Negative results

Two external signals were implemented, measured, and disabled. Documenting them prevents re-litigation.

### Hijri calendar — disabled

| Horizon | Without | With |
|---|---|---|
| 7d | 21.7% | 22.0% |
| 14d | **22.6%** | 25.8% |
| 28d | **25.6%** | 30.2% |

The underlying effect is real and large: during Ramadan, weekday volume rises **+37%** while Friday falls **−7%**, flattening the weekly profile. Eid al-Fitr runs +46%.

It fails for a data-availability reason, not a domain reason. The window contains two partial Ramadans; one lies entirely inside the excluded disruption period, leaving 19 usable days and **zero Ramadan days in the evaluation window**. Six features that are inactive at evaluation time consume capacity without contributing signal.

Enable via `USE_HIJRI = True` once 2+ years of history exist, or when a forecast horizon actually intersects Ramadan.

### FX rates — disabled

7d: 21.7% → 23.1%. 28d: 25.6% → 28.6%.

Rates enter as relative changes (`fx_chg_7`, `fx_chg_30`, `fx_vol_14`) rather than levels, avoiding the extrapolation failure described above. They still degrade accuracy, because the macro signal they carry is already captured by the trailing-mean denominator.

### Anomaly windows — enabled

The one external input that reliably helps, because it removes contamination rather than adding signal.

---

## Usage

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# place sales export at data/raw/Sales_Report-Ver01.xlsb

python run_all.py      # category level  → outputs/forecast_28d.xlsx
python run_item.py     # item level      → outputs/forecast_28d_item.xlsx
```

Stage 1 is shared and cached; `run_item.py` skips it when `clean.parquet` exists.

### Configuration

```python
# src/config.py
SALES_FILE = RAW / "Sales_Report-Ver01.xlsb"
WAR_START, WAR_END = "1404/12/09", "1405/01/19"
HORIZONS = [1, 3, 7, 14, 21, 28]
FORECAST_DAYS = 28
BLEND_WEIGHT = 0.5              # model vs baseline
ITEM_MIN_ACTIVE_DAYS = 30       # tier A threshold
USE_HIJRI = False
```

### Optional inputs

`data/raw/anomalies.csv` — periods to exclude:

```csv
start,end,branch,reason
1404/09/01,1404/09/30,Branch Name,permanent closure
1405/02/10,1405/02/12,ALL,nationwide power outage
```

Jalali dates. `ALL` applies chain-wide.

`data/raw/usd_rate.csv` — daily FX. Date column accepts Jalali or Gregorian; headers `date`/`تاریخ`/`jdate` and `price`/`قیمت`/`نرخ`/`close`.

### Output

| Sheet | Contents |
|---|---|
| Daily | branch × item/category × day |
| Branch-daily | per-branch totals — staffing input |
| Weekly | 4-week pivot with totals |
| Item totals | ranked 28-day totals (item level) |
| Merge audit | raw-name merge groups (item level) |

---

## Repository structure

```
├── data/
│   ├── raw/                     inputs (gitignored)
│   └── processed/               intermediate parquet (gitignored)
├── src/
│   ├── config.py                configuration
│   ├── clean_menu.py            name normalization & dedup
│   ├── categorize.py            category assignment
│   ├── features.py              shared feature construction
│   ├── external.py              optional external signals
│   ├── step1_extract.py         Excel → parquet
│   ├── step2_build.py           category panel
│   ├── step2b_build_item.py     item panel
│   ├── step3_validate.py        walk-forward validation
│   ├── step4_forecast.py        category forecast
│   └── step4b_forecast_item.py  item forecast
├── outputs/
├── run_all.py
└── run_item.py
```

---

## Operational notes

**Memory.** Stage 1 peaks near 6 GB parsing a 150 MB `.xlsb`. Reduce `CHUNK` in `step1_extract.py` if constrained. Later stages stay under 2 GB.

**Runtime.** Stage 1 ~10 minutes; stages 2–4 under 2 minutes combined.

**Retraining.** Drop a newer export into `data/raw/` and rerun. No manual state.

**Extending horizons.** `FORECAST_DAYS` in config. Accuracy degrades beyond 28 days — with 15 months of history there is no repeated season from which annual structure can be learned.

**Adding features.** Run `step3_validate.py` and compare against the results table before keeping anything. On a dataset this size, marginal features usually cost more than they contribute.

---

## Limitations

**No repeated seasons.** Fifteen months means Nowruz and summer appear exactly once. Annual patterns are memorized, not generalized. This is the binding constraint on forecast quality and cannot be engineered around.

**Granularity trades against accuracy.** Item-level error is ~1.5× category-level. An item selling 2 units/day is inherently noisier than a category selling 300. Totals reconcile within 3–7% at branch level, so aggregate figures remain reliable.

**Day-of-week dominates.** Friday runs 2.35× Saturday. Any method capturing this reaches most of the baseline; the ML contribution is a few points on top. That contribution is real and consistent, but the honest framing is refinement, not transformation.

**Rule-based categorization.** ~1.4% of items fall through to a catch-all. Extend the keyword lists in `categorize.py` to close gaps.

**Single-tenant assumptions.** Column positions, sheet names, and branch semantics are specific to this chain's export format. Adapting to another source means rewriting `step1_extract.py` and `config.COL_MAP`.

---

## Stack

```
pandas · numpy · pyarrow · lightgbm · scikit-learn · jdatetime · pyxlsb · openpyxl
```

Optional: `hijridate` for Hijri calendar features.
