# The Measurable Tradeoffs Behind The 4% Rule

*Date: May 14, 2026*

## Executive Summary

This report starts with a realistic retiree rather than an abstract withdrawal rate, then turns the usual conservative-versus-aggressive portfolio debate into measurable risk and quality-of-life tradeoffs.

Imagine someone retiring with a portfolio of a few million dollars. Some spending is mandatory: food, basic housing, utilities, insurance, taxes, health care, and the baseline costs of staying independent. Other spending is discretionary: vacations, nicer housing, gifts, upgrades, and the parts of retirement that make life feel abundant.

That budget naturally creates two retirement objectives:

- **Target spending:** the preferred lifestyle, including mandatory and discretionary spending.
- **Floor spending:** the minimum acceptable lifestyle, focused on mandatory spending.

This distinction matters because the generic permanent **60/40 portfolio plus 4% rule** treats "not running out of money" as the main objective. That is too narrow. A retiree can avoid ruin and still spend too many years below the lifestyle target.

The first contribution of this report is to replace vague labels like **conservative** and **aggressive** with measurable tradeoffs: odds of ruin, years below the target lifestyle, floor-breach risk, shortfall depth, and real ending wealth. The second contribution is to show why the 4% rule is conservative in this model, not as a slogan, but as a point on a risk-versus-quality-of-life curve.

With zero cash and zero bonds, a 4% target has **0.23% ruin**, **19.40% target-shortfall path-years**, and **4.27x real median ending wealth**. Moving to a 5% target raises current lifestyle but also raises risk: **1.21% ruin**, **24.73% target-shortfall path-years**, and **3.32x real median ending wealth**. That is the kind of tradeoff the report is designed to make explicit.

This report stress-tests the generic permanent 60/40 allocation against a target-and-floor spending model. The finding is direct: under this flexible spending objective, the generic 60/40 answer solves the easy metric, literal ruin, while weakening the harder objective, maintaining the target lifestyle.

The setup is now expressed as a percentage rule, not as one hardcoded starting portfolio:

- Target annual spending: **5% of the initial portfolio**, inflation-adjusted.
- Minimum spending floor: **half of target spending**, or **2.5% of the initial portfolio**, inflation-adjusted.
- Spending cap during the simulation: **5% of the current portfolio value**.
- Horizon: **30 years**.
- Inflation: **historical CPI-U inflation**, sampled from the same calendar years as returns.
- Cash return: **historical T-bill returns**, sampled from the same calendar years as stocks, bonds, and inflation. Cash means a T-bill-like reserve, not an assumed 0% real inflation-matched asset.
- Simulations: **20,000 Monte Carlo paths**.
- Market model: **5-year historical block bootstrap**.
- Return source: **Damodaran annual S&P 500 total returns with dividends reinvested, T-bill returns, and bond total returns**.
- Inflation source: **FRED CPIAUCNS**, calculated December-to-December.
- Main lookback window: **75 years**, excluding the Great Depression while retaining major modern drawdowns.
- Ending wealth: **real CPI-adjusted wealth**, shown in starting-year purchasing power.

The percentage rule maps to these concrete examples:

| Starting portfolio | Target spending (5%) | Minimum floor (2.5%) |
|---:|---:|---:|
| $2M | $100,000 | $50,000 |
| $3M | $150,000 | $75,000 |
| $4M | $200,000 | $100,000 |
| $5M | $250,000 | $125,000 |
| $6M | $300,000 | $150,000 |
| $10M | $500,000 | $250,000 |

Because the model has no taxes, fees, fixed-dollar expenses, or account limits, the simulation is scale-invariant. A result shown as `4.0x starting portfolio` means `$8M` of real ending wealth for a `$2M` starting portfolio, `$24M` for a `$6M` starting portfolio, and `$40M` for a `$10M` starting portfolio.

The main result: under this flexible spending rule, large permanent cash and bond allocations did not improve the chosen objective. They mostly traded a reduction in ruin and floor-breach risk for more target shortfall and lower median ending wealth. A 60% bond portfolio had low ruin in this run, but target shortfall rose to 51.71% of simulated path-years and median ending wealth fell below the starting portfolio.

The direct traditional benchmark makes the problem clearer:

| Benchmark | Spending rule | Bonds | Ruin | Shortfall path-years | Real final median |
|---|---|---:|---:|---:|---:|
| 4% stock-only | 4% target / 2% floor, flexible | 0% | 0.23% (46 paths) | 19.40% | 4.27x |
| 4% 60/40 | 4% target / 2% floor, flexible | 40% | 0.01% (1 path) | 26.01% | 2.06x |
| Classic fixed-real 4% 60/40 | Fixed 4%, no spending flexibility | 40% | 4.69% (938 paths) | 0.82% | 1.96x |
| 5% stock-only baseline | 5% target / 2.5% floor, flexible | 0% | 1.21% (242 paths) | 24.73% | 3.32x |
| 5% 60/40 | 5% target / 2.5% floor, flexible | 40% | 0.31% (62 paths) | 36.13% | 1.41x |

The flexible 4% target / 2% floor 60/40 plan looks safe if the only question is ruin. But it has more target-shortfall years and less than half the real median ending wealth of the 4% stock-only flexible plan. The true fixed-real 4% 60/40 benchmark behaves differently: it does not show many shortfall years because it refuses to cut spending until assets are depleted, which raises ruin to 4.69%.

The safe-withdrawal sweep gives useful context for the 5% baseline. With zero cash and zero bonds, a 4% target is robust. A 5% target still keeps floor breach low, but historical inflation makes it less forgiving than the fixed-inflation version. At 6%, measurable failure risk starts to appear. From 7% upward, the target becomes increasingly aggressive, and by 9-10% the plan is no longer close to safe under this model.

The tables report two decimal places for reproducibility, not because the second decimal place is economically meaningful. The larger uncertainty is model uncertainty: historical sample, block size, inflation treatment, asset data source, bond duration, and future regime.

## What The Simulator Measures

The simulator separates three outcomes that are easy to blur together:

- **Ruin:** percent of simulated paths that hit zero before year 30.
- **Target shortfall:** percent of all simulated path-years in which withdrawal is below the real target.
- **Floor breach:** percent of all simulated path-years in which the portfolio cannot fund the real floor.
- **Final p10 / median:** real ending wealth in starting-year purchasing power, divided by starting wealth.

That distinction is central. Ruin is catastrophic. Floor breach means the plan cannot support even the reduced lifestyle. Target shortfall is softer: the household is still spending above the floor, but below the preferred lifestyle target.

In plain household terms, target shortfall means the discretionary layer gets cut: fewer trips, less housing flexibility, fewer upgrades, fewer gifts, or delayed optional spending. Floor breach means the mandatory layer is no longer fully funded.

The spending rule itself creates target shortfall before the portfolio is ruined. If the portfolio falls below its starting value, a 5% current-portfolio cap allows less than the original 5% target.

For the 5% baseline, the rule is:

```text
target = 5.0% of initial portfolio, inflation adjusted
floor = 2.5% of initial portfolio, inflation adjusted
cap = 5.0% of current portfolio after that year's market return

withdrawal = min(current portfolio, max(floor, min(target, cap)))
```

The floor can override the cap. If the remaining portfolio cannot fund the floor, the withdrawal is limited to the remaining portfolio, and that year is counted as a floor breach.

The main table metric called **target shortfall** is the percentage of all simulated path-years below the target. It is not the same as the probability that a retiree ever sees a shortfall. The companion shortfall tables add three additional views: probability of ever missing the target, median number of shortfall years among affected paths, and average shortfall depth.

![Spending rule: 5% target and 2.5% floor](/Users/sergeyzelvenskiy/retirement_planning/notes/assets/bond_report_spending_cap.png)

The spending-rule chart separates four lived regimes: floor breach, mandatory spending only, discretionary cuts, and full target lifestyle. The left threshold, `0.025x`, means one year of floor spending remains. For a `$6M` starting portfolio, that is `$150,000`.

The report figures use the same visual language throughout:

- **Red** marks target spending or target shortfall.
- **Green** marks floor-related outcomes.
- **Gray** means ruin.
- **Blue** means ending wealth.
- Shaded regions mark the part of the chart where the strategy becomes either durable, aggressive, or fragile.

For charts with two panels, the top panel shows spending reliability and failure risk. The bottom panel shows ending wealth as a multiple of the starting portfolio. This is intentional: a strategy can reduce ruin while still making the target lifestyle less reliable.

## Block Bootstrap Method

The engine is a **historical block-bootstrap Monte Carlo**. It still generates 20,000 random 30-year retirement paths, but the random draw is an actual contiguous 5-year historical block, not an independently sampled normal-distribution return.

The method works like this:

1. Start with actual annual historical return and inflation records.
2. Randomly select a contiguous 5-year block from history, with replacement.
3. Append that block to a simulated path.
4. Repeat until the path reaches the 30-year retirement horizon, which is six 5-year blocks.
5. Run that process 20,000 times.

![Historical block bootstrap method](/Users/sergeyzelvenskiy/retirement_planning/notes/assets/bond_report_bootstrap_method.png)

The reason for using blocks is that market returns and inflation are not independent year to year. Real markets have clusters: crashes, recoveries, inflationary periods, high-return stretches, and sideways periods. Sampling 5-year blocks preserves some of that sequence structure.

Independent annual sampling can match average return and volatility, but it tends to break sequence risk and macro co-movement by making each year independent. The block-bootstrap method is better aligned with this question because retirement risk is driven by the order of returns, inflation, and withdrawals, not only by the average return.

Each simulated year follows this chronology:

1. Sample the next historical year from the current 5-year block.
2. Apply sampled stock, bond, and T-bill returns, then deflate by that year's CPI inflation.
3. Compute the spending cap from the post-return real portfolio value.
4. Withdraw at year-end under the target/cap/floor rule.
5. Replenish cash when the strategy calls for it, rebalance bonds annually, and record outcomes.

Because the model deflates returns each year, all ending-wealth multiples in the tables and charts are real, CPI-adjusted multiples of the starting portfolio.

For stock-only tests, the bootstrap samples Damodaran's annual S&P 500 total-return series, including dividends reinvested, joined by year to T-bill returns and CPI inflation. For stock/bond tests, the model uses **paired block bootstrap**: stocks, bonds, T-bills, and inflation are sampled from the same historical years in the same 5-year blocks. That matters because bonds, cash, and inflation should not be modeled as independent random streams. Their value depends on the same macro regime that produced the stock return.

The stock/bond data comes from Aswath Damodaran's annual return dataset, using S&P 500 total returns with dividends reinvested and 10-year Treasury bond total returns:

- https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histretSP.html

Inflation comes from FRED CPIAUCNS, calculated as December-to-December CPI-U inflation:

- https://fred.stlouisfed.org/series/CPIAUCNS

For the main comparison, the paired history covers **1951-2025**. All report experiments use this same Damodaran S&P 500 total-return source unless a table explicitly changes the lookback window. That is why the 5% zero-cash stock-only baseline now matches the 0% bond row in the bond experiment.

The chart assets are generated from `notes/assets/bond_report_results.json` by `notes/generate_report_assets.py`.

## Experiment 1: Cash Buffer

This experiment varies the cash buffer from 0 to 5 years of target spending. A buffer is measured in years of the target withdrawal. For example, with a $5M portfolio and a 4% target, target spending is $200,000, so a 5-year cash buffer is $1M held in a T-bill-like cash reserve. The reserve uses historical T-bill returns sampled from the same years as stock returns and inflation, so its real return can be positive or negative depending on the sampled year.

The remaining portfolio is invested in the Damodaran S&P 500 total-return series, using the 75-year lookback window with aligned historical T-bill returns and inflation.

| Cash buffer | Ruin | Target shortfall | Floor breach | Real final p10 | Real final median |
|---:|---:|---:|---:|---:|---:|
| 0 years | 1.21% | 24.73% | 0.20% | 0.60x | 3.32x |
| 1 year | 1.09% | 25.25% | 0.17% | 0.59x | 3.13x |
| 2 years | 1.00% | 26.43% | 0.14% | 0.56x | 2.87x |
| 3 years | 0.92% | 28.00% | 0.12% | 0.53x | 2.57x |
| 5 years | 0.74% | 32.68% | 0.08% | 0.43x | 1.92x |

![T-bill cash buffer tradeoff](/Users/sergeyzelvenskiy/retirement_planning/notes/assets/bond_report_cash_buffer.png)

The plot separates target shortfall from tiny ruin/floor probabilities so both sides of the tradeoff are visible. T-bill cash reduces the already-small floor risk only slightly, while target shortfall rises steadily and real ending wealth falls.

**Conclusion:** T-bill cash still does what cash is supposed to do: it slightly lowers ruin and floor-breach risk. But it remains expensive. A 5-year buffer lowers ruin from 1.21% to 0.74%, while target shortfall rises from 24.73% to 32.68% and median ending wealth falls from 3.32x to 1.92x. That is why the rest of the report uses zero cash buffer as the baseline.

## Experiment 2: Historical Window Sensitivity

This experiment keeps the same 5% target and 2.5% floor, with no cash buffer, and changes the S&P 500 total-return lookback window.

| History window | Ruin | Target shortfall | Floor breach | Real final p10 | Real final median |
|---:|---:|---:|---:|---:|---:|
| 50 years (1976-2025) | 0.33% | 19.11% | 0.04% | 0.88x | 4.47x |
| 75 years (1951-2025) | 1.21% | 24.73% | 0.20% | 0.60x | 3.32x |
| 98 years (1928-2025) | 2.25% | 29.85% | 0.41% | 0.44x | 2.64x |

![History sensitivity](/Users/sergeyzelvenskiy/retirement_planning/notes/assets/bond_report_history_sensitivity.png)

The Depression-inclusive 98-year case is useful as a stress test, but it changes the modeled world. It is not just a longer sample; it pulls in a different economic regime.

The 98-year window is a tail-regime stress test. It includes economic conditions that may not represent the central planning case, but it is useful for understanding sensitivity to rare historical sequences.

The 75-year window keeps difficult nominal-return regimes: the 1970s, 2000-2002, 2008, 2022, and other modern drawdowns. It excludes the Depression-era return path because the goal here is to test a contemporary baseline, not the most severe possible historical regime.

**Conclusion:** the 75-year window remains the main baseline because it keeps modern severe sequences, including the 1970s inflation problem, while excluding the Great Depression and World War II regime from the central case. The 98-year result should still be read as a legitimate stress test, not dismissed.

## Experiment 3: Safe Withdrawal Search

After the cash-buffer experiment, the working assumption is zero cash buffer. This experiment then searches target spending rates from 4% to 10% of the initial portfolio. Each row uses:

- zero cash buffer,
- zero bonds,
- 75-year Damodaran S&P 500 total-return block bootstrap,
- target spending equal to the listed cap,
- minimum floor equal to 50% of that target.

| Target spending rate | Floor | Ruin | Target shortfall | Floor breach | Real final p10 | Real final median |
|---:|---:|---:|---:|---:|---:|---:|
| 4% | 2.0% | 0.23% | 19.40% | 0.03% | 0.86x | 4.27x |
| 5% | 2.5% | 1.21% | 24.73% | 0.20% | 0.60x | 3.32x |
| 6% | 3.0% | 3.25% | 30.93% | 0.64% | 0.36x | 2.40x |
| 7% | 3.5% | 7.53% | 37.91% | 1.67% | 0.11x | 1.58x |
| 8% | 4.0% | 14.54% | 45.53% | 3.53% | 0.00x | 0.95x |
| 9% | 4.5% | 23.89% | 53.37% | 6.45% | 0.00x | 0.60x |
| 10% | 5.0% | 35.38% | 60.95% | 10.42% | 0.00x | 0.33x |

![Safe withdrawal search](/Users/sergeyzelvenskiy/retirement_planning/notes/assets/bond_report_safe_withdrawal_search.png)

The shaded regions summarize the curve: 4-5% remains the durable range, 6-7% is aggressive, and 8-10% becomes fragile quickly. These zone labels are judgmental summaries based on rising ruin, floor breach, target shortfall, and declining real p10 wealth.

The useful range is not a single magic number. The 4% target is conservative. The 5% target looks reasonable for a flexible-spending household with a 2.5% floor. The 6% target is more aggressive but not absurd. Above that, the risk profile changes quickly: target shortfall becomes common, floor breach becomes visible, and ending wealth erodes sharply.

This is why the bond experiment uses 5% as the main baseline. It is meaningfully higher than the traditional 4% rule, but still inside the range where ruin and floor-breach risk remain low in the 75-year block-bootstrap model.

The next table explains what "target shortfall" means in lived-path terms. The first column repeats the path-year metric. The second column is the probability that a path misses the target at least once.

| Target spending rate | Shortfall path-years | Ever miss target | Median shortfall years if any | Avg shortfall-year spending | Avg target gap | Integrated target loss |
|---:|---:|---:|---:|---:|---:|---:|
| 4% | 19.40% | 61.05% | 6 | 3.00% | 25.1% | 4.87% |
| 5% | 24.73% | 64.46% | 9 | 3.63% | 27.5% | 6.80% |
| 6% | 30.93% | 70.06% | 11 | 4.18% | 30.3% | 9.38% |
| 7% | 37.91% | 73.24% | 16 | 4.64% | 33.6% | 12.76% |
| 8% | 45.53% | 77.60% | 20 | 5.01% | 37.3% | 17.00% |
| 9% | 53.37% | 81.99% | 23 | 5.28% | 41.4% | 22.08% |
| 10% | 60.95% | 85.65% | 25 | 5.43% | 45.7% | 27.83% |

For example, at the 5% target, 24.73% of all simulated path-years are below target. Across paths, 64.46% experience at least one target miss. Among those affected paths, the median path has 9 shortfall years, and the average shortfall-year withdrawal is 3.63% of the initial portfolio instead of 5%. The integrated target-loss metric combines frequency and severity; at 5%, the average loss is 6.80% of target-year spending across the full retirement horizon.

**Conclusion:** the search turns the withdrawal rate into a quantifiable risk choice rather than a single magic number. Both 4% and 5% have low ruin risk in this model, but 4% makes that risk much lower: 0.23% ruin versus 1.21% at 5%. The tradeoff is lifestyle. A retiree can choose how much target shortfall, floor risk, and ending-wealth uncertainty they are willing to accept; above 6%, the plan starts depending too heavily on favorable sequences.

## Experiment 4: Block Size Sensitivity

This experiment keeps the 5% target, 2.5% floor, zero cash, zero bonds, and 75-year Damodaran S&P 500 total-return history. It changes only the bootstrap block size.

| Block size | Ruin | Target shortfall | Floor breach | Real final p10 | Real final median |
|---:|---:|---:|---:|---:|---:|
| 1 year | 1.47% | 23.46% | 0.32% | 0.62x | 3.82x |
| 3 years | 1.16% | 22.89% | 0.23% | 0.65x | 3.46x |
| 5 years | 1.21% | 24.73% | 0.20% | 0.60x | 3.32x |
| 10 years | 2.37% | 27.44% | 0.33% | 0.44x | 2.87x |

**Conclusion:** block size is a modeling assumption, not something to optimize until it best fits the past. Very short blocks break historical sequencing, but very long blocks can overfit to old macro regimes by replaying them too literally. A 5-year block is the working compromise: it preserves multi-year drawdowns and recoveries while recognizing that markets, policy response, technology, and inflation dynamics now change faster than they did in many older historical regimes.

## Experiment 5: Bonds

This experiment removes the cash buffer and varies the bond allocation from 0% to 60% of the non-cash portfolio. Bonds are modeled as 10-year Treasury total returns, paired with S&P 500 total returns from the same historical blocks.

Two flexible spending targets are shown. The 4% table tests a target/floor version of the traditional withdrawal-rate baseline. The 5% table tests the higher flexible-spending target used elsewhere in the report. The true fixed-real 4% withdrawal benchmark is shown in the executive summary because it is a different rule, not another target/floor case.

### 4% Target / 2% Floor

| Bond allocation | Ruin | Target shortfall | Floor breach | Real final p10 | Real final median |
|---:|---:|---:|---:|---:|---:|
| 0% bonds | 0.23% | 19.40% | 0.03% | 0.86x | 4.27x |
| 10% bonds | 0.10% | 19.79% | 0.01% | 0.85x | 3.68x |
| 20% bonds | 0.05% | 21.02% | 0.01% | 0.82x | 3.09x |
| 40% bonds | 0.01% (1 path) | 26.01% | 0.00% (3 path-years) | 0.70x | 2.06x |
| 60% bonds | 0.01% (1 path) | 37.93% | 0.00% (1 path-year) | 0.55x | 1.24x |

### 5% Target / 2.5% Floor

Because this experiment uses the same Damodaran S&P 500 total-return series as the stock-only experiments, the 0% bond row matches the 5% zero-cash baseline above.

| Bond allocation | Ruin | Target shortfall | Floor breach | Real final p10 | Real final median |
|---:|---:|---:|---:|---:|---:|
| 0% bonds | 1.21% | 24.73% | 0.20% | 0.60x | 3.32x |
| 10% bonds | 0.78% | 25.98% | 0.11% | 0.59x | 2.80x |
| 20% bonds | 0.48% | 28.07% | 0.06% | 0.57x | 2.29x |
| 40% bonds | 0.31% | 36.13% | 0.03% | 0.49x | 1.41x |
| 60% bonds | 0.45% | 51.71% | 0.04% | 0.37x | 0.83x |

![Bond allocation tradeoff](/Users/sergeyzelvenskiy/retirement_planning/notes/assets/bond_report_bond_allocation.png)

The plot separates target shortfall from tiny ruin/floor probabilities. More bonds drive ruin and floor breach toward zero, but target shortfall rises and real ending wealth falls.

Bonds do what they are supposed to do in a narrow risk-control sense: they reduce volatility, ruin, and floor breach.

But the cost is material. Higher bond allocations reduce the growth engine that supports the target lifestyle. In this run, going from 0% bonds to 60% bonds reduced median ending wealth from 3.32x starting portfolio to 0.83x starting portfolio. Target shortfall rose from 24.73% to 51.71%.

The target-shortfall experience also changes. The 60% bond allocation does not just create slightly more misses; it makes target misses much more common across paths.

| Bond allocation | Shortfall path-years | Ever miss target | Median shortfall years if any | Avg shortfall-year spending | Avg target gap | Integrated target loss |
|---:|---:|---:|---:|---:|---:|---:|
| 0% bonds | 24.73% | 64.46% | 9 | 3.63% | 27.5% | 6.80% |
| 10% bonds | 25.98% | 64.25% | 10 | 3.70% | 26.1% | 6.77% |
| 20% bonds | 28.07% | 68.02% | 10 | 3.76% | 24.9% | 6.99% |
| 40% bonds | 36.13% | 72.44% | 15 | 3.81% | 23.8% | 8.61% |
| 60% bonds | 51.71% | 84.02% | 21 | 3.74% | 25.1% | 12.99% |

This companion table is why the target-shortfall metric should not be read as a path failure rate. At 60% bonds, 51.71% is the share of simulated years below target; 84.02% is the share of paths that experience at least one target miss; and affected paths have a median of 21 target-shortfall years.

The severity story is more nuanced than the frequency story. The 40% bond row has more shortfall path-years than the 0% bond row, but a smaller average target gap in each shortfall year: 23.8% versus 27.5%. Bonds made shortfalls more frequent, not deeper in every comparison. The integrated target-loss metric combines the two effects, and it is still worse for 40% bonds: 8.61% of target-year spending versus 6.80% for stock-only.

![Objective tradeoff](/Users/sergeyzelvenskiy/retirement_planning/notes/assets/bond_report_objective_tradeoff_4_5.png)

The objective chart isolates the bond decision for both the 4% and 5% targets: each step toward more bonds moves down and to the right, toward lower real median ending wealth and higher target shortfall.

Ending wealth is not just inheritance or excess. It is longevity cushion. A 30-year simulation is a modeling horizon, not a known lifespan. If the retiree lives 35 or 40 years, the difference between ending with 3.32x and 1.41x is not cosmetic. It is the reserve that protects against extra years, late-life care, bad post-year-30 returns, family needs, and inflation surprises.

**Conclusion:** bonds reduce ruin and floor-breach risk, but they do not improve the target/floor objective. At 4%, stock-only already has a 0.23% ruin rate; 60/40 cuts that to 0.01%, but target shortfall rises from 19.40% to 26.01% and real median ending wealth falls from 4.27x to 2.06x. At 5%, 60/40 cuts ruin from 1.21% to 0.31%, but shortfall becomes more frequent. The average gap in a shortfall year is smaller with 40% bonds, but the integrated target loss is still worse: 8.61% of target-year spending versus 6.80% for stock-only.

## Interpretation

The traditional allocation argument says bonds reduce sequence risk. That is true, but incomplete.

This plan already has risk controls:

1. Spending is capped as a percentage of current portfolio value.
2. Spending can fall from the target to the floor.
3. The floor is only half of the target.

Those controls absorb much of the sequence risk before the portfolio is exhausted. Once that is true, the value of extra defensive allocation falls in this model. The plan does not need a large bond sleeve to prevent ruin in most paths. It needs enough growth to keep spending near the target.

That is why cash and bonds look weak here. They reduce small catastrophic risks, but they increase a more frequent lifestyle risk: spending below the target.

## What This Says About 60/40 And 40/60 Advice

Terminology matters:

- **60/40** usually means 60% stocks and 40% bonds.
- **40/60** means 40% stocks and 60% bonds.

This report tests 0%, 10%, 20%, 40%, and 60% bond allocations. The 40% bond case is the traditional 60/40 portfolio. The strongest evidence is against the 60% bond case, but the 40% bond case is also worse than 0% bonds on target shortfall and median wealth in this model.

The conclusion is sharper than the prior version of this report: the generic permanent 60/40 plus 4% recommendation breaks under this flexible-spending objective. It protects against an already-small ruin risk while giving up too much target lifestyle reliability, compounding, and longevity cushion.

The defensible conclusion is still narrower than "bonds are bad." Under this flexible spending rule, large permanent cash and bond allocations did not improve the chosen objective. They mainly traded a small reduction in already-low ruin/floor risk for more target shortfall and lower long-term wealth.

It also weakens several common defenses of bond-heavy advice.

### Required Spending

One common argument is that bonds are useful when spending cannot flex. In this setup, that is backward.

If spending truly cannot flex, then target shortfall is not a soft lifestyle downgrade. It is the failure metric. On that metric, the 60% bond portfolio is the worst tested allocation: it produced target shortfall in 51.71% of simulated path-years and ended with median wealth below the starting portfolio.

So the right conclusion is not "use more bonds when spending is required." The right conclusion is that required spending needs one of these:

- a lower required spending level,
- a larger starting portfolio,
- outside income such as Social Security, pensions, or annuities,
- explicit short-term liability matching for known near-term expenses,
- or a spending policy that can actually flex.

### Near-Term Liabilities

Near-term liability matching is real, but it is not the same as a permanent bond-heavy strategy.

If a known bill must be paid this year or next year, the matching asset should be cash, T-bills, or short-duration Treasuries sized to that specific liability. That is an expense reserve. It does not imply that 40-60% of the whole long-term portfolio should sit in bonds indefinitely.

### Drawdown Tolerance

"The investor cannot tolerate equity drawdowns" is also not a portfolio-efficiency argument. It is a behavioral constraint.

If someone will abandon an equity strategy after a 30-50% drawdown, then a lower-equity allocation may be better than a strategy they cannot actually hold. But that is not evidence that 40/60 is financially superior. It means the investor is paying an expected-return and lifestyle-reliability cost to reduce the chance of panic behavior.

That tradeoff should be stated plainly.

### Low Volatility

Low volatility can be an objective, but it is a different objective from funding a high real lifestyle over decades.

If the goal is "make the account balance move less," bonds help. If the goal is "spend near the target, keep the floor safe, and preserve long-term wealth," this experiment says a high bond allocation is not helping.

### Taxes, Fees, And Account Structure

The tax argument is not modeled here, so it should not be used as evidence for a 40/60 portfolio in this report. Equity and bond tax treatment depends on account type, holding period, asset location, and the investor's tax situation. Trading costs for broad ETFs are also usually too small to justify a large bond allocation by themselves.

There can be account-specific reasons to avoid selling a particular equity position, such as a concentrated low-basis holding or an illiquid asset. But that is a special tax-planning problem, not a general argument for a permanent bond-heavy portfolio.

The remaining defensible cases for bonds are narrow: explicit short-term liability matching, a bond ladder or annuity-like structure for a non-negotiable floor, or a behavioral compromise for someone who cannot hold a stock-heavy portfolio. Those are not the same claim as "60/40 or 40/60 is the right default retirement allocation."

For a retiree who can downshift spending from 5% of initial wealth to a 2.5% floor, the high-bond portfolio is protecting against an already-small risk while making the preferred lifestyle less likely.

The sharper conclusion is that the generic permanent 60/40 plus 4% recommendation breaks under this flexible-spending objective. The report's larger contribution is replacing abstract labels like "conservative" and "aggressive" with a quantified risk-versus-quality-of-life tradeoff. A retiree can see the actual odds of ruin, the odds and depth of target shortfall, the chance of breaching the spending floor, and the size of the longevity cushion left at year 30.

A portfolio is only conservative if it conserves what the retiree actually values. In this model, a generic permanent 60/40 allocation buys a small reduction in already-low ruin risk by accepting more years below the desired lifestyle and a smaller reserve for long life, late-life care, family needs, bad late returns, and inflation surprises. That is not automatically safer. It is a different quality-of-life choice with measurable costs.

## Practical Takeaways

1. **Do not optimize for ruin alone.** A portfolio can make ruin nearly impossible by accepting a much lower lifestyle. That is not necessarily a better retirement outcome.

2. **Track target shortfall separately from floor breach.** Missing the target is not the same as violating the minimum floor.

3. **Cash and bonds are not free safety.** They reduce drawdown exposure, but they also reduce the compounding needed to support spending.

4. **Small bond allocations are a compromise, not a free improvement.** A 10-20% bond sleeve reduced floor risk, but target shortfall still increased and median wealth fell.

5. **Large permanent bond allocations did not improve this objective.** At 40-60% bonds, target shortfall and median-wealth drag become the dominant effects.

## Limitations

Important limitations:

- Taxes are ignored.
- Fees are ignored.
- Returns are annual, not monthly.
- Withdrawals happen after annual returns; beginning-of-year or monthly withdrawals may be harsher.
- Inflation is annual CPI-U, not monthly household-specific inflation.
- Cash earns historical T-bill returns sampled from the same calendar years as stocks, bonds, and inflation.
- Bonds are 10-year Treasury annual total returns, not a live bond fund.
- Cash is replenished mechanically when the strategy is at a market peak; a different reserve policy could change results.
- Spending behavior is mechanical.
- The bootstrap assumes historical blocks are a reasonable proxy for future regimes.
- The model does not include Social Security, pensions, mortgages, health shocks, or estate goals.

These limitations matter. They are the reason this report should not be read as a universal anti-bond claim.

The most important follow-up tests are:

- alternative inflation sources and spending baskets,
- TIPS or inflation-linked bond proxies,
- T-bills, 5-year Treasuries, 10-year Treasuries, and short Treasury ladders,
- explicit floor-funding with a bond ladder or annuity-like income stream,
- Social Security or pension income layered into the spending rule,
- monthly returns and monthly withdrawal timing,
- beginning-of-year versus end-of-year withdrawals,
- 35- and 40-year longevity horizons,
- 25%, 50%, and 75% floor-to-target ratios,
- probability of 5+ and 10+ consecutive years below target,
- dynamic bond glidepaths rather than fixed allocations,
- Social Security or pension income layered into the floor,
- valuation-aware equity assumptions,
- and additional return datasets where available.

Until those are tested, the result should be framed narrowly: under this specific flexible spending rule, with annual returns, sampled historical CPI inflation, and permanent cash/bond allocations, the high defensive allocations did not improve the selected objective.

## Bottom Line

The experiment supports a skeptical view of large permanent bond allocations for this kind of flexible-spending retiree.

The primary objective is not to stay fully invested for its own sake. The objective is to live well, downshift when needed, and avoid running out of money.

Under this flexible spending rule, the main risk is not literal ruin. The main risk is spending below the target lifestyle. Because the rule already cuts spending when wealth falls, large permanent cash and bond allocations mostly reduce an already-small catastrophic risk while increasing target shortfall and lowering long-term wealth.

The report does not replace one abstract label with another. It replaces "conservative" and "aggressive" with measurable odds of ruin, measurable lifestyle shortfall, floor-breach risk, and real longevity cushion. Under this model, a generic permanent 60/40 allocation is not automatically safer. It is a different quality-of-life tradeoff with visible costs.
