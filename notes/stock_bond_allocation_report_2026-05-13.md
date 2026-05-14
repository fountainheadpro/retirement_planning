# A Retirement Plan That Breaks The 60/40 Story

*Date: May 13, 2026*

## Executive Summary

This report starts with a realistic retiree rather than an abstract withdrawal rate.

Imagine someone retiring with a portfolio of a few million dollars. Some spending is mandatory: food, basic housing, utilities, insurance, taxes, health care, and the baseline costs of staying independent. Other spending is discretionary: vacations, nicer housing, gifts, upgrades, and the parts of retirement that make life feel abundant.

That budget naturally creates two retirement objectives:

- **Target spending:** the preferred lifestyle, including mandatory and discretionary spending.
- **Floor spending:** the minimum acceptable lifestyle, focused on mandatory spending.

This distinction matters because the traditional advice of a **60/40 portfolio and the 4% rule** treats "not running out of money" as the main objective. That is too narrow. A retiree can avoid ruin and still spend too many years below the lifestyle target.

This report stress-tests the traditional 60/40 plus 4% framing against a target-and-floor spending model. The finding is direct: under this flexible spending objective, the generic 60/40 answer solves the easy metric, literal ruin, while weakening the harder objective, maintaining the target lifestyle.

The setup is now expressed as a percentage rule, not as one hardcoded starting portfolio:

- Target annual spending: **5% of the initial portfolio**, inflation-adjusted.
- Minimum spending floor: **half of target spending**, or **2.5% of the initial portfolio**, inflation-adjusted.
- Spending cap during the simulation: **5% of the current portfolio value**.
- Horizon: **30 years**.
- Inflation: **3% fixed annual inflation**.
- Cash return: **3% nominal**, so cash earns roughly 0% real return.
- Simulations: **20,000 Monte Carlo paths**.
- Market model: **5-year historical block bootstrap**.
- Return source: **Damodaran annual S&P 500 total returns with dividends reinvested, T-bill returns, and bond total returns**.
- Main lookback window: **75 years**, excluding the Great Depression while retaining major modern drawdowns.

The percentage rule maps to these concrete examples:

| Starting portfolio | Target spending (5%) | Minimum floor (2.5%) |
|---:|---:|---:|
| $2M | $100,000 | $50,000 |
| $3M | $150,000 | $75,000 |
| $4M | $200,000 | $100,000 |
| $5M | $250,000 | $125,000 |
| $6M | $300,000 | $150,000 |
| $10M | $500,000 | $250,000 |

Because the model has no taxes, fees, fixed-dollar expenses, or account limits, the simulation is scale-invariant. A result shown as `4.0x starting portfolio` means `$8M` ending wealth for a `$2M` starting portfolio, `$24M` for a `$6M` starting portfolio, and `$40M` for a `$10M` starting portfolio.

The main result: under this flexible spending rule, large permanent cash and bond allocations did not improve the chosen objective. They mostly traded a small reduction in ruin and floor-breach risk for more target shortfall and lower median ending wealth. A 60% bond portfolio eliminated ruin in this run, but target shortfall rose to 47.84% of simulated path-years and median ending wealth fell below the starting portfolio.

The direct traditional benchmark makes the problem clearer:

| Benchmark | Target / floor | Bonds | Ruin | Target shortfall | Final median |
|---|---:|---:|---:|---:|---:|
| 4% stock-only | 4% / 2% | 0% | 0.04% | 14.83% | 5.19x |
| Traditional 4% 60/40 | 4% / 2% | 40% | 0.00% | 18.51% | 2.47x |
| 5% stock-only flexible baseline | 5% / 2.5% | 0% | 0.40% | 19.60% | 4.11x |
| 5% 60/40 | 5% / 2.5% | 40% | 0.00% | 29.27% | 1.75x |

The classic 4% 60/40 plan looks safe if the only question is ruin. But it has more target-shortfall years and less than half the median ending wealth of the 4% stock-only plan. It also delivers a lower target lifestyle than the 5% stock-only baseline while ending with much less wealth.

The safe-withdrawal sweep gives useful context for the 5% baseline. With zero cash and zero bonds, a 4% target is very robust. A 5% target still keeps ruin and floor breach low. At 6%, measurable failure risk starts to appear. From 7% upward, the target becomes increasingly aggressive, and by 9-10% the plan is no longer close to safe under this model.

The tables report two decimal places for reproducibility, not because the second decimal place is economically meaningful. The larger uncertainty is model uncertainty: historical sample, block size, inflation treatment, asset data source, bond duration, and future regime.

## What The Simulator Measures

The simulator separates three outcomes that are easy to blur together:

- **Ruin:** the portfolio runs out of money.
- **Target shortfall:** a simulated year has withdrawals below the target.
- **Floor breach:** a simulated year has withdrawals below the minimum floor.

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

The report figures use the same visual language throughout:

- **Red** means target shortfall.
- **Green** means floor breach.
- **Gray** means ruin.
- **Blue** means ending wealth.
- Shaded regions mark the part of the chart where the strategy becomes either durable, aggressive, or fragile.

For charts with two panels, the top panel shows spending reliability and failure risk. The bottom panel shows ending wealth as a multiple of the starting portfolio. This is intentional: a strategy can reduce ruin while still making the target lifestyle less reliable.

## Block Bootstrap Method

The simulations use historical block bootstrap rather than a normal distribution or a simple average-return assumption.

The method works like this:

1. Start with actual annual historical returns.
2. Randomly select a contiguous 5-year block from history.
3. Append that block to a simulated path.
4. Repeat until the path reaches the 30-year retirement horizon.
5. Run that process 20,000 times.

The reason for using blocks is that market returns are not independent year to year. Real markets have clusters: crashes, recoveries, inflationary periods, high-return stretches, and sideways periods. Sampling 5-year blocks preserves some of that sequence structure.

For stock-only tests, the bootstrap samples Damodaran's annual S&P 500 total-return series, including dividends reinvested. For stock/bond tests, the model uses **paired block bootstrap**: stocks and bonds are sampled from the same historical years in the same 5-year blocks. That matters because bonds should not be modeled as an independent random return stream. Their value depends on the same macro regime that produced the stock return.

The stock/bond data comes from Aswath Damodaran's annual return dataset, using S&P 500 total returns with dividends reinvested and 10-year Treasury bond total returns:

- https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histretSP.html

For the main comparison, the paired history covers **1951-2025**. All report experiments use this same Damodaran S&P 500 total-return source unless a table explicitly changes the lookback window. That is why the 5% zero-cash stock-only baseline now matches the 0% bond row in the bond experiment.

The chart assets are generated from `notes/assets/bond_report_results.json` by `notes/generate_report_assets.py`.

## Experiment 1: Cash Buffer

This experiment varies the cash buffer from 0 to 5 years of target spending. The portfolio is otherwise invested in the Damodaran S&P 500 total-return series, which includes dividends reinvested, using the 75-year lookback window.

| Cash buffer | Ruin | Target shortfall | Floor breach | Final p10 | Final median |
|---:|---:|---:|---:|---:|---:|
| 0 years | 0.40% | 19.60% | 0.06% | 0.82x | 4.11x |
| 1 year | 0.39% | 20.15% | 0.06% | 0.79x | 3.88x |
| 2 years | 0.34% | 21.33% | 0.05% | 0.75x | 3.53x |
| 3 years | 0.29% | 22.96% | 0.04% | 0.69x | 3.15x |
| 5 years | 0.21% | 28.17% | 0.02% | 0.56x | 2.33x |

![Cash buffer tradeoff](/Users/sergeyzelvenskiy/retirement_planning/notes/assets/bond_report_cash_buffer.png)

The top panel shows the key problem: cash reduces the already-small floor risk only slightly, while target shortfall rises steadily. The bottom panel shows the cost in lower ending wealth.

The cash buffer slightly reduces ruin and floor-breach risk. But those risks are already small without the buffer. The visible cost is more target shortfall and lower median ending wealth.

That is the reason the cash buffer looks weak in this model. It is not solving the binding problem. The binding problem is sustaining the target lifestyle while preserving enough long-term growth.

Cash can still have behavioral value. It may help someone avoid panic-selling during a drawdown. But as a mechanical portfolio improvement, the modeled buffer is expensive relative to the risk reduction it buys.

This is the key setup decision for the rest of the report: use zero cash buffer as the baseline, then ask what spending cap is reasonable under that assumption.

## Experiment 2: Historical Window Sensitivity

This experiment keeps the same 5% target and 2.5% floor, with no cash buffer, and changes the S&P 500 total-return lookback window.

| History window | Ruin | Target shortfall | Floor breach | Final p10 | Final median |
|---:|---:|---:|---:|---:|---:|
| 50 years (1976-2025) | 0.34% | 16.03% | 0.05% | 1.04x | 5.56x |
| 75 years (1951-2025) | 0.40% | 19.60% | 0.06% | 0.82x | 4.11x |
| 98 years (1928-2025) | 3.74% | 27.47% | 0.95% | 0.41x | 2.99x |

![History sensitivity](/Users/sergeyzelvenskiy/retirement_planning/notes/assets/bond_report_history_sensitivity.png)

The Depression-inclusive 98-year case is useful as a stress test, but it changes the modeled world. It is not just a longer sample; it pulls in a different economic regime.

The 98-year window is much harsher because it includes the Great Depression. That may be useful as a stress test, but it is a different modeling choice from a modern-retirement baseline.

The 75-year window keeps difficult nominal-return regimes: the 1970s, 2000-2002, 2008, 2022, and other modern drawdowns. It excludes the Depression-era return path because the goal here is to test a contemporary baseline, not the most severe possible historical regime.

Important caveat: the model uses fixed 3% inflation. That means it includes 1970s nominal asset returns but does **not** fully include 1970s spending-pressure inflation. A stronger version of this analysis should test historical inflation or real returns directly.

## Experiment 3: Safe Withdrawal Search

After the cash-buffer experiment, the working assumption is zero cash buffer. This experiment then searches target spending rates from 4% to 10% of the initial portfolio. Each row uses:

- zero cash buffer,
- zero bonds,
- 75-year Damodaran S&P 500 total-return block bootstrap,
- target spending equal to the listed cap,
- minimum floor equal to 50% of that target.

| Target cap | Floor | Ruin | Target shortfall | Floor breach | Final p10 | Final median |
|---:|---:|---:|---:|---:|---:|---:|
| 4% | 2.0% | 0.04% | 14.83% | 0.00% | 1.20x | 5.19x |
| 5% | 2.5% | 0.40% | 19.60% | 0.06% | 0.82x | 4.11x |
| 6% | 3.0% | 1.42% | 25.56% | 0.26% | 0.56x | 3.09x |
| 7% | 3.5% | 4.04% | 32.73% | 0.79% | 0.32x | 2.13x |
| 8% | 4.0% | 9.26% | 40.56% | 2.03% | 0.03x | 1.27x |
| 9% | 4.5% | 17.52% | 49.12% | 4.28% | 0.00x | 0.75x |
| 10% | 5.0% | 28.44% | 57.56% | 7.66% | 0.00x | 0.47x |

![Safe withdrawal search](/Users/sergeyzelvenskiy/retirement_planning/notes/assets/bond_report_safe_withdrawal_search.png)

The shaded regions summarize the curve: 4-5% is durable in this model, 6-7% is aggressive, and 8-10% becomes fragile quickly.

The useful range is not a single magic number. The 4% target is conservative. The 5% target looks reasonable for a flexible-spending household with a 2.5% floor. The 6% target is more aggressive but not absurd. Above that, the risk profile changes quickly: target shortfall becomes common, floor breach becomes visible, and ending wealth erodes sharply.

This is why the bond experiment uses 5% as the main baseline. It is meaningfully higher than the traditional 4% rule, but still inside the range where ruin and floor-breach risk remain low in the 75-year block-bootstrap model.

The next table explains what "target shortfall" means in lived-path terms. The first column repeats the path-year metric. The second column is the probability that a path misses the target at least once.

| Target cap | Shortfall path-years | Ever miss target | Median shortfall years if any | Avg shortfall-year spending | Avg target gap |
|---:|---:|---:|---:|---:|---:|
| 4% | 14.83% | 55.17% | 5 | 3.10% | 22.4% |
| 5% | 19.60% | 59.84% | 7 | 3.77% | 24.7% |
| 6% | 25.56% | 65.12% | 9 | 4.36% | 27.3% |
| 7% | 32.73% | 68.99% | 14 | 4.88% | 30.3% |
| 8% | 40.56% | 73.10% | 19 | 5.28% | 34.0% |
| 9% | 49.12% | 77.50% | 22 | 5.59% | 37.9% |
| 10% | 57.56% | 83.70% | 24 | 5.77% | 42.3% |

For example, at the 5% target, 19.60% of all simulated path-years are below target. Across paths, 59.84% experience at least one target miss. Among those affected paths, the median path has 7 shortfall years, and the average shortfall-year withdrawal is 3.77% of the initial portfolio instead of 5%.

## Experiment 4: Block Size Sensitivity

This experiment keeps the 5% target, 2.5% floor, zero cash, zero bonds, and 75-year Damodaran S&P 500 total-return history. It changes only the bootstrap block size.

| Block size | Ruin | Target shortfall | Floor breach | Final p10 | Final median |
|---:|---:|---:|---:|---:|---:|
| 1 year | 0.84% | 20.03% | 0.17% | 0.79x | 4.61x |
| 3 years | 0.49% | 18.23% | 0.11% | 0.86x | 4.31x |
| 5 years | 0.40% | 19.60% | 0.06% | 0.82x | 4.11x |
| 10 years | 0.68% | 22.79% | 0.09% | 0.69x | 3.59x |

The direction of the result is not dependent on the exact 5-year block choice, but the numbers move enough to treat the second decimal place as false precision. Block size is a modeling assumption, not a law of nature.

## Experiment 5: Bonds

This experiment removes the cash buffer and varies the bond allocation from 0% to 60% of the non-cash portfolio. Bonds are modeled as 10-year Treasury total returns, paired with S&P 500 total returns from the same historical blocks.

Because this experiment uses the same Damodaran S&P 500 total-return series as the stock-only experiments, the 0% bond row matches the 5% zero-cash baseline above.

| Bond allocation | Ruin | Target shortfall | Floor breach | Final p10 | Final median |
|---:|---:|---:|---:|---:|---:|
| 0% bonds | 0.40% | 19.60% | 0.06% | 0.82x | 4.11x |
| 10% bonds | 0.22% | 20.38% | 0.02% | 0.82x | 3.46x |
| 20% bonds | 0.04% | 21.92% | 0.00% | 0.78x | 2.85x |
| 40% bonds | 0.00% | 29.27% | 0.00% | 0.67x | 1.75x |
| 60% bonds | 0.00% | 47.84% | 0.00% | 0.52x | 0.96x |

![Bond allocation tradeoff](/Users/sergeyzelvenskiy/retirement_planning/notes/assets/bond_report_bond_allocation.png)

The shaded high-bond region shows the central tradeoff. More bonds drive ruin and floor breach toward zero, but target shortfall rises and ending wealth falls.

Bonds do what they are supposed to do in a narrow risk-control sense: they reduce volatility, ruin, and floor breach.

But the cost is material. Higher bond allocations reduce the growth engine that supports the target lifestyle. In this run, going from 0% bonds to 60% bonds reduced median ending wealth from 4.11x starting portfolio to 0.96x starting portfolio. Target shortfall rose from 19.60% to 47.84%.

The target-shortfall experience also changes. The 60% bond allocation does not just create slightly more misses; it makes target misses much more common across paths.

| Bond allocation | Shortfall path-years | Ever miss target | Median shortfall years if any | Avg shortfall-year spending | Avg target gap |
|---:|---:|---:|---:|---:|---:|
| 0% bonds | 19.60% | 59.84% | 7 | 3.77% | 24.7% |
| 10% bonds | 20.38% | 61.36% | 7 | 3.87% | 22.5% |
| 20% bonds | 21.92% | 63.03% | 7 | 3.97% | 20.7% |
| 40% bonds | 29.27% | 69.70% | 10 | 4.08% | 18.4% |
| 60% bonds | 47.84% | 81.85% | 20 | 4.02% | 19.5% |

This companion table is why the target-shortfall metric should not be read as a path failure rate. At 60% bonds, 47.84% is the share of simulated years below target; 81.85% is the share of paths that experience at least one target miss; and affected paths have a median of 20 target-shortfall years.

![Objective tradeoff](/Users/sergeyzelvenskiy/retirement_planning/notes/assets/bond_report_objective_tradeoff.png)

The objective chart isolates the bond decision: each step toward more bonds moves down and to the right, toward lower median ending wealth and higher target shortfall.

The tradeoff is not subtle. The 60% bond portfolio is safer if the only metric is avoiding zero. It is worse if the objective is maintaining the target lifestyle while keeping the floor breach risk very low.

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

The conclusion is sharper than the prior version of this report: for this flexible-spending retiree, the generic 60/40 plus 4% recommendation breaks. It protects against an already-small ruin risk while giving up too much target lifestyle reliability and compounding.

The defensible conclusion is still narrower than "bonds are bad." Under this flexible spending rule, large permanent cash and bond allocations did not improve the chosen objective. They mainly traded a small reduction in already-low ruin/floor risk for more target shortfall and lower long-term wealth.

It also weakens several common defenses of bond-heavy advice.

### Required Spending

One common argument is that bonds are useful when spending cannot flex. In this setup, that is backward.

If spending truly cannot flex, then target shortfall is not a soft lifestyle downgrade. It is the failure metric. On that metric, the 60% bond portfolio is the worst tested allocation: it produced target shortfall in 47.84% of simulated path-years and ended with median wealth below the starting portfolio.

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

The sharper conclusion is:

> For this flexible-spending retiree, the generic 60/40 plus 4% recommendation breaks: it protects against an already-small ruin risk while giving up too much target lifestyle reliability and compounding.

## Practical Takeaways

1. **Do not optimize for ruin alone.** A portfolio can make ruin nearly impossible by accepting a much lower lifestyle. That is not necessarily a better retirement outcome.

2. **Track target shortfall separately from floor breach.** Missing the target is not the same as violating the minimum floor.

3. **Cash and bonds are not free safety.** They reduce drawdown exposure, but they also reduce the compounding needed to support spending.

4. **Small bond allocations are a compromise, not a free improvement.** A 10-20% bond sleeve reduced floor risk, but target shortfall still increased and median wealth fell.

5. **Large permanent bond allocations did not improve this objective.** At 40-60% bonds, target shortfall and median-wealth drag become the dominant effects.

## Limitations

This is a model result, not personalized financial advice. Important limitations:

- Taxes are ignored.
- Fees are ignored.
- Returns are annual, not monthly.
- Inflation is fixed at 3%, not sampled historically.
- Cash earns a fixed 3% nominal return.
- Bonds are 10-year Treasury annual total returns, not a live bond fund.
- Spending behavior is mechanical.
- The bootstrap assumes historical blocks are a reasonable proxy for future regimes.
- The model does not include Social Security, pensions, mortgages, health shocks, or estate goals.

These limitations matter. They are the reason this report should not be read as a universal anti-bond claim.

The most important follow-up tests are:

- historical inflation or real returns instead of fixed 3% inflation,
- TIPS or inflation-linked bond proxies,
- T-bills, 5-year Treasuries, 10-year Treasuries, and short Treasury ladders,
- explicit floor-funding with a bond ladder or annuity-like income stream,
- Social Security or pension income layered into the spending rule,
- monthly returns and monthly withdrawal timing,
- beginning-of-year versus end-of-year withdrawals,
- dynamic bond glidepaths rather than fixed allocations,
- valuation-aware equity assumptions,
- and additional return datasets where available.

Until those are tested, the result should be framed narrowly: under this specific flexible spending rule, with annual returns, fixed inflation, and permanent cash/bond allocations, the high defensive allocations did not improve the selected objective.

## Bottom Line

The experiment supports a skeptical view of large permanent bond allocations for this kind of flexible-spending retiree.

The primary objective is not to stay fully invested for its own sake. The objective is to live well, downshift when needed, and avoid running out of money.

Under this flexible spending rule, the main risk is not literal ruin. The main risk is spending below the target lifestyle. Because the rule already cuts spending when wealth falls, large permanent cash and bond allocations mostly reduce an already-small catastrophic risk while increasing target shortfall and lowering long-term wealth.

This does not prove that bonds are useless. It says that for a retiree who can flex from a 5% target to a 2.5% floor, broad equity exposure is hard to beat in this model unless bonds are being used for a specific liability, behavioral need, or guaranteed-income strategy.
