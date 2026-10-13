# MLB Pitch Type Prediction

Predicts the next pitch type an MLB pitcher will throw, using only information available *before* the pitch: count, outs, runners, score, handedness, rest days, recent pitch history, and batter tendencies. Trained on real Statcast data.

## Results

| Model | Test Accuracy |
|---|---|
| Random guess | 7.1% |
| Always predict fastball | 31.7% |
| Logistic regression | 40.0% |
| Gradient boosting (base features) | 39.4% |
| + batter data, rare-class consolidation, tuning | **42.2%** |

The final model beats the best naive baseline by 10.5 points (33% relative). Its **top-2 accuracy is 68.1% and top-3 is 83.2%**, so the true pitch is usually among the model's first few guesses. This is in line with published work on the same problem, which reports 7-10 point gains over baseline (e.g., 59.5% to 70%).

## Data

- **Source:** MLB Statcast via `pybaseball`: 50 pitchers (starters and relievers with deliberately varied arsenals), Jan 2025 to Jul 2026, about 151K pitches.
- **Cleaning:** removed spring training and exhibition games (9% of rows). Fixed accented-name lookup failures. Caught two silent duplicate-name bugs (a "Luis Castillo" and a "Charlie Morton" whose lookups matched the wrong player, one last active in 1885) because they returned zero pitches without raising an error.
- **Split:** chronological train/validation/test, so the model is only evaluated on games after the ones it trained on.
- **Batter data:** a separate pull of the 150 most frequently faced batters, used to compute each batter's on-base rate against each pitch type.

## Features (about 22 core, plus batter stats)

- **Game state:** count, outs, inning, runners, score differential, handedness, times through the order
- **Rolling in-game features:** previous 1-3 pitch types, rolling velocity, in-game pitch mix, in-game whiff rate by pitch type
- **Season aggregates:** pitcher pitch mix, starter/reliever role, batter on-base rate by pitch type

Any column describing the *current* pitch (its velocity, spin, location, or outcome) was excluded to avoid leakage. All rolling features use only prior pitches.

## Key Findings

- **Systematic fastball bias:** the model identifies fastballs well (96% recall at top-3) but defaults to "fastball" when unsure. Curveballs were mislabeled as fastballs more often (463) than correctly identified (164).
- **Rare-class consolidation was the main driver of improvement.** Merging three classes (0.20% of rows) into "OTHER" added about 1-2 points, confirmed with a 2x2 ablation against tuning and batter volume.
- **The ceiling is a signal limit, not an undertrained model.** A deliberately overfit model reached 63% train accuracy but only 41.6% test accuracy.

**Tested, no meaningful gain:** more batter coverage (571 vs. 150 batters), pitcher-vs-batter matchup history (too sparse), runner sprint speed (pitchouts too rare), and cross-timezone travel (effects too mild). A simplified fastball/breaking/offspeed model reached 55.4% against a 53.0% baseline, showing the same fastball bias.

## Limitations

- Pitch-type labels are Statcast's own algorithmic classification, which is imperfect for similar pitches like cutters and sliders.
- Personalization only covers the 50 pitchers in training; raw pitcher ID does not generalize to new pitchers.
- Information like catcher calls, in-game feel, and injuries isn't in any available dataset.

## Future Work

Pitch location and outcome prediction, defensive-quality features, weather, and testing XGBoost (it was not used here; scikit-learn's `HistGradientBoostingClassifier` was used instead).

## Repository Structure and Reproduction

```
├── features.py                  # Feature-engineering pipeline and splits
├── 01_data_exploration.ipynb    # Data pulls, EDA, modeling, experiments
├── requirements.txt
└── README.md
```

```bash
git clone https://github.com/aaravmehta111/baseball-pitch-predictor.git
cd baseball-pitch-predictor
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Then run `01_data_exploration.ipynb` top to bottom. The 50-pitcher pull takes about 15-20 minutes. Data files are not committed because they are regenerable.

**Stack:** Python, pandas, NumPy, scikit-learn, `pybaseball`, Jupyter, Git.
