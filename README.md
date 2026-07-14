# HR Employee Attrition Analysis

Why do employees leave, and which factors actually predict it? This project digs into the IBM HR Analytics dataset to find out, using EDA and two classification models, then turns the findings into actual HR recommendations instead of just a pile of charts.

## Dataset

[IBM HR Analytics Employee Attrition Dataset](https://www.kaggle.com/datasets/pavansubhash/ibm-hr-analytics-attrition-dataset) (Kaggle) — 1,470 employees, 35 features covering demographics, compensation, satisfaction scores, and tenure.

## What I found

- **Overtime is the strongest driver.** Employees working overtime leave at roughly 31%, versus 10% for everyone else — about 3x higher. This shows up as the top feature in both the correlation heatmap and the Random Forest model, not just one method.
- **Income matters, but less than expected.** Employees who left had a lower median monthly income (~₹3,100 vs ~₹5,200), ranking second in feature importance — real, but not the main story.
- **Tenure points to an early-career problem.** Median tenure for leavers is ~3 years, versus ~6 for those who stayed. This looks like a retention gap in the first few years, not a company-wide issue.
- **Raw counts by job role are misleading.** Sales Executives had the most departures in absolute numbers, but that's mostly because it's the largest role. Looking at rate instead of count, Sales Representatives and Laboratory Technicians actually lose a higher share of people.
- **The better-looking model isn't the more useful one.** Random Forest hits 86% accuracy but catches only 21% of actual leavers. Logistic Regression looks worse on paper (72% accuracy) but catches 56% of them. For a tool meant to flag at-risk employees, the second number is the one that matters.

Full write-up with reasoning: see `Results.pdf` in this repo.

## Approach

1. **Cleaning** — dropped constant-value columns, checked for nulls and duplicates, encoded the target variable
2. **EDA** — attrition rate by job role, overtime, income, tenure, commute distance, and a correlation heatmap
3. **Modeling** — Logistic Regression and Random Forest, both with `class_weight='balanced'` to handle the 84/16 class imbalance
4. **Evaluation** — precision/recall/F1 per class, not just accuracy, since accuracy alone hides how well a model catches actual leavers

## Repo contents

- `HR_Employee_Attrition.csv` — dataset
- `HR_Employee_Attrition_Analysis.py` — full analysis script (cleaning, EDA, modeling)
- `Results.pdf` — all plots and model outputs
- `README.md` — this file

## Tools

Python, pandas, numpy, matplotlib, seaborn, scikit-learn

## Running it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python HR_Employee_Attrition_Analysis.py
```

Keep `HR_Employee_Attrition.csv` in the same folder as the script.

## What I'd want next

Exit interview text, manager notes, or engagement survey scores would probably explain more than anything in this dataset. The numeric HR fields tell you who's leaving. They're weaker at telling you why.
