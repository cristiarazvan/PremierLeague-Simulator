# Premier League Monte Carlo Simulator

This project implements a **Monte Carlo Simulation** to predict the outcomes of the English Premier League. By leveraging player statistics (Expected Goals, Expected Assists, Progression stats) and team performance metrics, the simulator models individual match probabilities and aggregates them to forecast the entire league season.

## 📌 Project Overview (Conform Tasks.md)

### 1. The Problem
We aim to estimate the probability distribution of the final league standings and individual match outcomes. Football is a stochastic process where "skill" dictates the mean performance, but "luck" (variance) plays a significant role.
- **Mathematical Formulation:** We model the number of goals scored by a team in a match as a **Poisson Process**, where the rate $\lambda$ is a function of the attacking team's power vs. the defending team's resilience, adjusted for home advantage and league averages.

### 2. Monte Carlo Algorithm
The simulation follows these steps:
1.  **Model Calibration:** Convert raw player stats into "Attack" and "Defense" power ratings for each team.
2.  **Noise Injection:** For every match simulation, we introduce random Gaussian noise ($\mathcal{N}(0, \sigma)$) to team powers to simulate form/luck fluctuations.
3.  **Poisson Sampling:** Goals are generated using $Goals \sim \text{Poisson}(\lambda_{adjusted})$.
4.  **Aggregation:** We repeat this process $N$ times. By the Law of Large Numbers, the average of these simulations converges to the theoretical expected probability of each outcome.

---

## 📐 Theoretical Demonstration: Number of Simulations

We aim for a **95% Confidence Level** with a **Margin of Error (\epsilon) of roughly 1%**.

We determine the required number of simulations $N$ using the **Central Limit Theorem (CLT)**.

### Calculation of N
For estimating a probability $p$ (e.g., "Probability Liverpool wins"), the standard error is maximized when $p=0.5$.
The formula for the margin of error is:
$$\epsilon = \frac{Z_{\alpha/2} \cdot \sqrt{p(1-p)}}{\sqrt{N}}$$

Where:
*   **Confidence Level:** $95\% \Rightarrow \alpha = 0.05 \Rightarrow Z_{0.025} \approx 1.96$.
*   **Worst-case Variance:** $p=0.5 \Rightarrow \sqrt{p(1-p)} = 0.5$.
*   **Desired Error (\epsilon):** $\approx 0.01$ (1%).

Rearranging for $N$:
$$N \geq \left( \frac{1.96 \cdot 0.5}{0.01} \right)^2$$
$$N \geq \left( \frac{0.98}{0.01} \right)^2 = (98)^2 = 9604$$

**Conclusion:** We require approximately **9,604 simulations** to achieve a $\le 1\%$ margin of error with 95% confidence.

### Implementation Decision
We have set our simulation count to:
$$N = 10,000$$

Recalculating the precise error for $N=10,000$:
$$\epsilon = \frac{1.96 \cdot 0.5}{\sqrt{10000}} = \frac{0.98}{100} = 0.0098 \approx \mathbf{0.98\%}$$

This confirms that running 10,000 iterations guarantees (under the assumptions of the model) that our predicted probabilities are within ~1% of the "true" model values.

---

## 📂 Project Structure

```
├── main.py                 # Core script (runs standard simulation)
├── interactive_session.py  # Interactive CLI for Custom Teams/Lineups
├── data/
│   └── raw/                # CSV Stats from FBref
├── src/
│   ├── data_loader.py      # Parses CSVs into Objects
│   ├── league.py           # League Logic & Simulation Engine
│   ├── models.py           # Player & Team Classes
│   └── utils.py            # Math helpers & Stat calculations
```

## 🚀 How to Use

### 1. Interactive Mode (Recommended)
This is the primary way to interact with the project. It allows you to:
- **Simulate specific matches** (e.g., Liverpool vs Man City).
- **Run the full league season** (10,000 times) to see the predicted table.
- **Manage Teams:** Edit starting lineups to see how benching a star player affects the odds.
- **Create Custom Teams:** Build your own "Dream Team" from the database and add them to the league.

**To Run:**
```bash
python interactive_session.py
```

### 2. Standard Simulation
Runs the default setup defined in `main.py`, which typically executes a full league simulation and prints the results.
```bash
python main.py
```

## 📊 Evaluation & Analysis
*   **Originality:** We moved beyond simple FIFA ratings by calculating "Moment Power" which includes stochastic form deviations ($\sigma=0.15$). We also allow for dynamic lineup changes which directly impact the Poisson $\lambda$ values.
*   **Rigour:** We utilize the Poisson distribution for scorelines, acknowledging that goals are rare, independent events. The choice of $N=10,000$ is mathematically justified to ensure statistical significance.
*   **Analysis:** The tool outputs Win/Draw/Loss percentages which can be compared against betting odds or real-world results to validate the model's accuracy.

## 📚 References
1.  **FBref:** Source of raw player and team statistics.
2.  **Central Limit Theorem:** Used to derive the required number of Monte Carlo iterations.
3.  **Poisson Distribution:** Mathematical basis for modeling football scores.
