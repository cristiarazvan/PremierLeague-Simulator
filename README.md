# Premier League Monte Carlo Simulator

This project implements a **Monte Carlo Simulation** to predict the outcomes of the English Premier League. By leveraging player statistics (Expected Goals, Expected Assists, Progression stats) and team performance metrics, the simulator models individual match probabilities and aggregates them to forecast the entire league season.

## 📌 Project Overview

### 1. The Problem
We aim to estimate the probability distribution of the final league standings and individual match outcomes. Football is a stochastic process where "skill" dictates the mean performance, but "luck" (variance) plays a significant role.
- **Mathematical Formulation:** We model the number of goals scored by a team in a match as a **Poisson Process**, where the rate $\lambda$ is a function of the attacking team's power vs. the defending team's resilience, adjusted for home advantage and league averages.

### 2. Monte Carlo Algorithm
The simulation follows these steps:
1.  **Model Calibration:** Convert raw player stats into "Attack" and "Defense" power ratings for each team.
2.  **Noise Injection:** For every match simulation, we introduce random Gaussian noise ($\mathcal{N}(0, \sigma)$) to team powers to simulate form/luck fluctuations.
3.  **Poisson Sampling:** Goals are generated using $Goals \sim \text{Poisson}(\lambda_{adjusted})$.
4.  **Aggregation:** We repeat this process $N$ times. By the Law of Large Numbers (LLN), the average of these simulations converges to the theoretical expected probability of each outcome.

---

## 📐 Mathematical Proof & Convergence

### Why 10,000 Simulations?

We need to determine the number of simulations $N$ required to achieve a **95% Confidence Level** with a **Margin of Error (\epsilon) of $\approx 1%$**.

We rely on the **Central Limit Theorem (CLT)**. The CLT states that the distribution of sample means approximates a normal distribution as the sample size becomes larger.

For estimating a probability $p$ (e.g., "The probability that Arsenal finishes 1st"), the margin of error $\epsilon$ at a $(1-\alpha)$ confidence level is given by:

$$ \epsilon = Z_{\alpha/2} \sqrt{\frac{p(1-p)}{N}} $$

Where:
*   $Z_{\alpha/2}$ is the Z-score for the confidence interval. For 95%, $\alpha = 0.05$ and $Z_{0.025} \approx 1.96$.
*   $p$ is the true probability. The variance is maximized when $p=0.5$ (worst-case scenario), so $\sqrt{p(1-p)} = 0.5$.
*   $N$ is the number of simulations.

### Derivation for N

Rearranging the formula to solve for $N$ given a desired error of $\epsilon = 0.01$ (1%):

$$ N \geq \left( \frac{Z_{\alpha/2} \cdot \sqrt{p(1-p)}}{\epsilon} \right)^2 $$

$$ N \geq \left( \frac{1.96 \cdot 0.5}{0.01} \right)^2 $$

$$ N \geq \left( \frac{0.98}{0.01} \right)^2 = (98)^2 = 9604 $$

**Conclusion:** We require approximately **9,604 simulations** to guarantee that our estimated probabilities for league standings are within a 1% margin of error of the true model probabilities.

### Implementation
We have set our simulation count to:
$$ N = 10,000 $$

Recalculating the precise error for $N=10,000$:

$$ \epsilon = \frac{1.96 \cdot 0.5}{\sqrt{10000}} = \frac{0.98}{100} = 0.0098 \approx \mathbf{0.98\% } $$

This rigorous approach ensures that the "League Heatmaps" and "Win Probabilities" presented in the tool are statistically significant and not artifacts of random noise.

---

## 📊 Visualizations & Analysis tools

We have implemented two specific scripts to visualize the data and prove convergence:

### 1. Interactive Visualizer (`interactive_session.py`)
After running a full league simulation (Option 2) or a match simulation (Option 1), the tool offers visualization options. **All plots are automatically saved to the `plots/` directory.**

#### A. League Position Heatmap
A color-coded matrix showing the probability of every team finishing in every position (1-20). This provides a much richer understanding than just "Average Points."

![League Heatmap Example](plots/league_heatmap.png)
*(Run Option 2 in the CLI to generate this image)*

#### B. Points Distribution Histogram
A histogram showing the range of points a specific team scored across the 10,000 simulations, including the 95% Confidence Interval.

![Points Distribution Example](plots/Liverpool_points_dist.png)
*(Run Option 2, then Select Team in the CLI to generate this image)*

#### C. Match Convergence Plot
When simulating a specific match, you can generate a convergence plot to see how the win probability stabilizes as $N \rightarrow 10,000$.

![Convergence Plot Example](plots/Liverpool_convergence.png)
*(Run Option 1 in the CLI to generate this image)*

### 2. Convergence Proof Script (`convergence_check.py`)
This standalone script runs a specific experiment to demonstrate the Law of Large Numbers. It simulates a single match 10,000 times and plots the running average.

---

## 📂 Project Structure

```
├── main.py                 # Core script (CLI entry point)
├── interactive_session.py  # Advanced CLI with Visualizations
├── convergence_check.py    # Math proof script (generates plots)
├── plots/                  # Generated images directory
├── data/
│   └── raw/                # CSV Stats from FBref
├── src/
│   ├── visualizer.py       # Plotting library (Matplotlib/Seaborn)
│   ├── data_loader.py      # Parses CSVs into Objects
│   ├── league.py           # League Logic & Simulation Engine
│   ├── models.py           # Player & Team Classes
│   └── utils.py            # Math helpers & Stat calculations
```

## 🚀 How to Use

### 1. Interactive Mode (Recommended)
This is the primary way to interact with the project.
```bash
python interactive_session.py
```
*   **Simulate League:** Runs 10k seasons. Wait for it to finish (~20-30s), then choose "Show Position Heatmap" to see the full probability spread.
*   **Simulate Match:** Choose "Simulate Single Match", enter two teams, and then press 'V' to see the Monte Carlo convergence graph.
*   **Manage Teams:** Edit starting lineups to see how benching a star player affects the odds.

### 2. Run Convergence Analysis
To generate the convergence plot for your report without the CLI menu:
```bash
python convergence_check.py
```

## 📚 References
1.  **FBref:** Source of raw player and team statistics.
2.  **Central Limit Theorem (CLT):** Used to derive the required number of Monte Carlo iterations.
3.  **Poisson Distribution:** Mathematical basis for modeling football scores.
