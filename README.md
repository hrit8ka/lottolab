# Loto Lab

**Can you beat the Mauritius Loto with data science?**
Short answer: no. Long answer: this repo, where I try very hard, measure everything honestly, and end up with a null result I can actually defend.

I scraped every Loto 6/40 draw I could find (932 of them, February 2016 to November 2026), ran statistical tests on them, built five "prediction" strategies, and backtested all of them. Then I wrapped it in a dashboard, because a dashboard is more fun than a CSV.

---

## The result, up front

| Question | Answer |
|---|---|
| Are some numbers drawn more than others? | Not beyond what chance produces (p = 0.26) |
| Do pairs of numbers show up together unusually often? | No (p = 0.35) |
| Does one draw predict the next? | No (p = 0.38) |
| Did any strategy beat random on ~830 backtested draws? | No. None survived correction for multiple comparisons |
| Can my pipeline find a bias if one exists? | Yes, on synthetic lotteries (see below) |

So the lottery looks fair, which is what a regulated lottery should be. I'm treating that as the finding, not a failure. If I had "found" a winning method, my first move would be to assume I had a bug.

The app still generates "predictions", because that's the fun part. It also shows, right next to them, that they do no better than random picks. That's deliberate.

---

## Why build this at all

A lottery is a clean test case for a messy question: when does a model find a real signal, and when is it fitting noise? Real-world data usually can't answer that, because you never know the ground truth. Here I do know it (it *should* be random), so I can check whether my methods behave correctly. Getting a pipeline to say "nothing here" and be right is harder than it sounds, and most beginner "lottery predictors" skip it entirely.

It's also a decent excuse to practise the full chain in one project: messy web data, validation, storage, statistics, leak-free evaluation, and an app on top.

---

## The game and the maths

Mauritius Loto 6/40: six distinct numbers from 1 to 40, drawn on Wednesdays and Saturdays.

- Number of possible tickets: C(40, 6) = **3,838,380**
- Chance that a given number appears in a draw: 6/40 = 0.15
- If you pick six numbers, the expected matches are 6 × 0.15 = **0.9**
- The full distribution of matches is hypergeometric, with a standard deviation of about **0.82**

That 0.9 is the benchmark for everything in this repo. A strategy only means something if it beats 0.9 by more than luck would allow.

---

## Where the data comes from, and what went wrong

I wanted fully automated collection, no typing results by hand. I looked at three sources:

- **Lottotech** (the official site). It sits behind a Cloudflare waiting-room queue. I'm not going to try to dodge that, so I only use it for manual spot-checks.
- **my.t** (Mauritius Telecom). It only lists the last 20 draws. Fine for cross-checking recent results, useless as a history.
- **Yelo**. It has a month filter going back to February 2016. This became the main source.

Yelo's page mixes three games in one table: Loto (6 numbers), Loto Plus (a separate draw) and Loto Vert (7 numbers, drawn daily, often repeating). My parser keeps only rows labelled "Loto Result" with exactly six numbers. My first parser did not do this, and I only noticed because I tested it against what the page really looked like. Lesson learned.

**Why a browser (Playwright) and not just `requests`?** The month filter is a dropdown on the page, so I let a real browser click it, save each month's HTML, and parse that offline. Cached pages mean I never hit their server twice for the same month, and there's a 3-second pause between requests so I'm not hammering anyone.

**The bug I'm weirdly proud of.** My first run printed "9 Loto draws" for every month. It looked fine until I noticed November 2026 can only have one draw so far. The month filter wasn't applying, and each file was just the default "last 30 draws" page again. Now the downloader checks that every draw falls inside the requested month, and stops with a diagnostic dump if not. A script that fails loudly beats one that quietly saves wrong data.

I don't redistribute the data. `data/raw` and the database are git-ignored; you rebuild them by running the downloader yourself.

### What the data looks like

- 932 draws, 2016-02-06 to 2026-11-04, 126 monthly pages downloaded
- Every draw valid: six distinct numbers in 1..40, no conflicts
- **Two eras.** Until August 2018 there was one draw a week (124 draws). From September 2018 it's twice a week (808 draws)
- Gaps from missing months on Yelo (January 2017 and 2018, April and May 2020). The 2020 gap lines up with the COVID lockdown, and March/April 2021 look thin for the same reason
- Three draws on odd weekdays (one each on a Sunday, Monday and Thursday), most likely holiday shifts

Missing months don't bias anything here, because each draw is independent. They do reduce the sample, so I mention them.

**Why SQLite and not PostgreSQL?** About a thousand rows. Postgres would be a server to run for no benefit. SQLite is one file, and swapping later is easy.

---

## How the evaluation works (the part I care about most)

### 1. No peeking at the future
Every prediction is made using only draws that happened *before* the one being predicted. This is called walk-forward testing: step through history in order, pick, reveal the real draw, score, move on. I never shuffle draws into random train/test splits, because that leaks future information into the past.

There are unit tests for this. One changes the last draw and checks that nothing before it changes. Another wipes the future and checks that features stay identical.

### 2. Test for signal *before* modelling
It's much cheaper to ask "is there anything non-random in here?" than to train models and hope. Three direct tests:

- **Number uniformity** (chi-squared): is each number drawn about equally often?
- **Pair independence**: do certain pairs co-occur too often?
- **Draw-to-draw overlap**: do consecutive draws share numbers more than the expected 0.9?

For all three I use Monte Carlo p-values: simulate thousands of genuinely fair lotteries of the same size and see where the real data falls. I chose that over textbook formulas because draws are without replacement, so the standard approximations are slightly off.

### 3. Power: how small an edge could I even detect?
With about 830 scored draws, the smallest real improvement I'd reliably notice is around **0.08 hits per draw**. Anything smaller is invisible to this dataset. So "no edge found" really means "no edge larger than about 0.08 found". I'd rather say that than claim the lottery is provably perfect.

### 4. Comparing many strategies = many chances to get lucky
Test enough strategies and one will look good by accident. I fixed the protocol in advance (one main metric, one family of four strategies) and apply a Holm correction to the p-values. The random strategy is kept out of that family and used as a sanity check.

### 5. Synthetic lotteries, to check my own pipeline
Before trusting any real result I built two fake lotteries:

- **Fair:** every number equally likely. Nothing should be "discovered". It wasn't.
- **Biased:** six numbers made 1.5× more likely. Bayesian/Thompson and "hot" caught it in all three runs. (1.5× is a big bias, deliberately obvious, to prove the machinery works.)

If the pipeline can find a planted bias and stays quiet on a fair lottery, then a quiet result on the real data means something.

---

## The strategies

| Strategy | What it does | Why it's here |
|---|---|---|
| **Random** | Picks six numbers uniformly | The control. Everything else has to beat this |
| **Hot** | The six most frequent numbers in the last 50 draws | The classic "hot numbers" belief |
| **Overdue** | The six numbers absent the longest | The gambler's fallacy, tested properly |
| **Bayesian / Thompson** | Keeps a probability estimate for each number, samples from it, picks the top six | A principled statistical learner. If any real signal exists, this is the one most likely to find it |
| **Low-overlap** | Leans towards numbers above 31 and away from "lucky" ones | See below |

**About low-overlap.** It does *not* make you more likely to win. Many players choose birthdays, so low numbers get picked a lot. If you ever won, you'd share the jackpot with more people. Playing less popular numbers can raise your expected payout *given a win*, without changing your odds. It's the only strategy with a plausible real-world edge. One honest caveat: the popularity weights are my own guesses, not measured from ticket data.

**Why no deep reinforcement learning (yet)?** A lottery draw isn't affected by what you pick, so this isn't a true sequential RL problem. It's closer to a one-step contextual bandit, and the best any learner can do is estimate each number's probability, which the Bayesian model already does. PPO is on my list as a "does extra complexity help?" comparison, and I'd bet money (not lottery money) that it lands at chance too.

---

## Real results

832 scored draws (the first 100 are warm-up), average hits per draw vs the 0.9 benchmark:

| Strategy | Mean hits | Avg over 5 seeds | p-value | p (Holm) |
|---|---|---|---|---|
| random (control) | 0.887 | 0.909 | 0.66 | n/a |
| hot | 0.894 | 0.907 | 0.84 | 1.00 |
| overdue | 0.917 | 0.921 | 0.54 | 1.00 |
| bayes_thompson | 0.834 | 0.886 | 0.018 | 0.07 |
| low_overlap | 0.894 | 0.877 | 0.84 | 1.00 |

Bayesian/Thompson's raw p-value of 0.018 looks tempting until you look closer. The edge is *negative* (it did worse than chance), the corrected p-value is 0.07, and averaged over five random seeds it sits at 0.886, close to the 0.9 benchmark. The random control itself moved between 0.887 and 0.909 across seeds. That wobble is the noise level, and Thompson is inside it.

Most drawn number: 21 and 34 (161 times each). Least drawn: 40 (114 times). A fair lottery with 40 numbers and 932 draws expects about 140 each, give or take 11, so those extremes are ordinary.

---

## Forward test: the version nobody can argue with

Backtests are open to accusations of subtle cheating. So I also log each strategy's picks *before* a draw and score them after. The logger:

- seeds picks from the draw date, so they're reproducible
- refuses to re-roll picks for a draw that's already logged
- refuses to log picks for a draw already in the database
- stores a UTC timestamp and the latest draw the model had seen

First picks were logged on 6 November 2026 for the draw on 7 November. Results will go here as they come in.

<!-- TODO: paste the forward-test summary table here after a few weeks of draws -->

---

## The dashboard

Run it with `streamlit run dashboard/app.py`.

- **Dashboard:** overview, frequency chart, and a button that runs the randomness tests
- **Number explorer:** frequencies over different windows, time since each number last appeared, pair heatmap
- **Prediction lab:** generate combinations from any strategy, with a plain-English description of how it picked
- **Scoreboard:** the walk-forward backtest, hit distribution against theory, cumulative hits above expectation
- **Forward test:** logged picks and how they scored

If real data isn't loaded, the app falls back to a clearly labelled synthetic demo dataset so it still runs.

**Why Streamlit?** It turns a Python script into an app without a front-end project, which suits a data project. I'd pick something else for a production site.

---

## Run it yourself

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium

python -m ingestion.yelo            # download every Loto month (about 7 minutes, be polite)
python -m ingestion.run_pipeline    # validate and load into SQLite
python -m experiments.real_report   # tests + backtest, printed as one report
python -m experiments.synthetic     # fair vs biased sanity check
streamlit run dashboard/app.py      # the dashboard
pytest                              # 17 tests
```

Before a draw: `python -m experiments.forward_test picks --date YYYY-MM-DD`
After a draw: re-run the downloader and pipeline, then `python -m experiments.forward_test score`.

Tested on Python 3.9 and 3.12 (macOS and Linux).

---

## Project layout

```
ingestion/     probe, Yelo downloader and parser, validation, SQLite storage
features/      leak-free frequency, recency and co-occurrence features
agents/        the five strategies
evaluation/    walk-forward backtester, metrics, power, Holm correction, Monte Carlo tests
experiments/   synthetic lotteries, real-data report, forward test
dashboard/     Streamlit app
tests/         17 tests (validation, no-lookahead, agents, parsing, forward log)
scripts/       setup helpers
```

---

## Honest limitations

- About 830 scored draws means I can only detect edges of roughly 0.08 hits or larger. A subtler bias could be hiding.
- Yelo is a third-party site, not the official source. I haven't yet hand-checked a sample of draws against Lottotech or my.t, though the draw dates I compared line up.
- Missing months on Yelo leave small gaps in the history.
- The low-overlap popularity weights are assumptions.
- No deep RL agent yet.
- "No pattern found" is not "proven random". It's "nothing detectable, with this much data".

## Ideas for next

- Cross-check a sample of draws against the official source
- A PPO agent, purely as a "complexity doesn't help" comparison
- Let the forward test run for a few months and report it
- Look at Loto Plus as a second, independent dataset

---

## Data and fair play

Draw data comes from public result pages. Check each site's terms before collecting it, keep your request rate low, and don't redistribute it. This project is for learning. Lotteries have a negative expected value, so if you play, treat it as entertainment and spend only what you'd happily spend on a night out. No method here, or anywhere, improves your odds of winning.
