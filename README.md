# Customer Churn Prediction & Model Optimization

**Name:** AARON BENNY PHILIP

**MUID:** aaronbennyphilip@mulearn

---

## 📌 Project Summary
Customer retention is one of the key metrics for subscription-based business models. Acquiring new customers is often significantly more expensive than retaining existing ones. 

This project builds, optimizes, and evaluates Machine Learning classification models to predict customer churn using the **Customer Churn Dataset** (~505,207 total entries). By identifying high-risk customers early and understanding the key drivers behind churn, business teams can proactively deploy retention campaigns and minimize revenue loss.

---

## 🛠️ Optimization Approach & Workflow

The machine learning workflow follows a structured, end-to-end pipeline:

1. **Data Ingestion & Aggregation:**
   - Merged the training dataset (`440,833` rows) and testing dataset (`64,374` rows) into a single consolidated dataset (`505,207` rows).
   - Handled missing values (`dropna()`) and removed non-predictive identifiers (`CustomerID`).

2. **Feature Engineering & Preprocessing:**
   - Encoded categorical variables (`Gender`, `Subscription Type`, `Contract Length`) using label encoding.
   - Standardized numeric features using `StandardScaler` to ensure zero mean and unit variance.
   - Stratified dataset splitting (`80% Train` / `20% Test`) to preserve original churn class distribution.

3. **Baseline Model:**
   - Trained a standard **Logistic Regression** model as a linear baseline.
   - Evaluated using cross-industry metrics: **Accuracy, Precision, Recall, F1 Score, and ROC-AUC**.

4. **Model Optimization Strategy:**
   - Transitioned to an ensemble model: **Random Forest Classifier**.
   - **Optimization Technique:** Executed **`RandomizedSearchCV`** over 10 parameter combinations with 3-fold Stratified Cross-Validation (`StratifiedKFold`).
   - **Efficiency Sampling Strategy:** Sampled a stratified subset of 30,000 samples (`X_tune`) during hyperparameter search to optimize compute time and prevent notebook timeouts while maintaining distribution validity.
   - **Handling Class Imbalance:** Set `class_weight="balanced"` to ensure equal importance for churned and retained cases.
   - Fitted the optimized hyperparameter configuration back onto the complete training set.

---

## 📊 Key Findings & Model Improvements

### Metric Comparison Matrix

| Metric | Baseline (Logistic Regression) | Optimized (Random Forest) | Net Improvement |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 81.96% | **93.57%** | **+11.61%** |
| **Precision** | 84.48% | **89.84%** | **+5.36%** |
| **Recall** | 82.71% | **99.69%** | **+16.98%** |
| **F1 Score** | 0.8359 | **0.9451** | **+0.1093** |
| **ROC AUC** | 0.8874 | **0.9531** | **+0.0657** |

---

## 🔍 Important Observations & Feature Importance

Based on the `feature_importances_` extracted from the Optimized Random Forest model, the **top drivers of customer churn** are:

1. **Support Calls:** The #1 strongest predictor of churn. High call volume signals unresolved customer pain points or technical frustration.
2. **Total Spend:** Customers with lower historical spend or declining lifetime value show a significantly higher likelihood to churn.
3. **Payment Delay:** Frequent payment delays serve as an immediate early indicator of financial disengagement or upcoming cancellation.
4. **Age:** Specific age demographic segments exhibit varying retention rates, indicating potential product-market fit differences.
5. **Contract Length:** Short-term (e.g., monthly) contract holders churn at substantially higher rates compared to annual plan subscribers.

---

## 💡 Strategic Business Recommendations

Based on empirical evidence from the optimized model:

1. **Proactive Support Interventions:** Set up automated triggers when a customer reaches **3+ support calls within a month** to route them to a specialized customer success team.
2. **Early Warning System for Delays:** Implement automated payment reminders and engagement workflows when a customer's **Payment Delay exceeds threshold limits**.
3. **Contract Conversion Incentives:** Offer targeted discounts or value-added perks for converting short-term/monthly subscribers to annual contracts.
4. **High-Value Loyalty Programs:** Develop retention incentives specifically for high **Total Spend** customers to prevent revenue leakage.

---

## 🏁 Final Conclusions

- Transitioning from a linear model (**Logistic Regression**) to a non-linear ensemble model (**Random Forest**) with hyperparameter tuning yielded significant gains, bringing the **F1 Score to ~0.945** and **Recall to ~99.7%**.
- High recall ensures that nearly all customers at risk of churning are flagged in advance, enabling maximum retention coverage.
- Focusing customer success efforts on **Support Call resolution** and **Payment Delay monitoring** provides the highest ROI for customer retention campaigns.

---

### 📂 Repository Structure
```text
.
├── model_optimization.ipynb   # Full code implementation with outputs & plots
└── README.md                  # Project summary, optimization details & findings
