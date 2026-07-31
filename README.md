Week 3 --- Task 5


 MODEL EVALUATION & HYPERPARAMETER TUNING REPORT
+
--------------------------------------------------------------------------------
1. TASK OBJECTIVES & REQUIREMENTS
--------------------------------------------------------------------------------
- Revisit classification model from Task 1 (Titanic Dataset).
- Evaluate baseline model using Precision, Recall, and F1-Score via 
  sklearn.metrics.classification_report.
- Explain why accuracy alone can be misleading on imbalanced datasets.
- Use GridSearchCV to tune at least 2 hyperparameters (C, penalty).
- Compare tuned model performance against the original baseline using a 
  before/after comparison table.
- Document conclusions for GitHub README update.


--------------------------------------------------------------------------------
2. WHY ACCURACY ALONE IS MISLEADING ON IMBALANCED DATASETS
--------------------------------------------------------------------------------
Accuracy measures the percentage of correct predictions out of total predictions:
   
   Accuracy = (True Positives + True Negatives) / Total Samples

In imbalanced datasets—where one class significantly outnumbers another—accuracy 
becomes a deceptive metric:

1. The Majority Class Fallacy: 
   If a dataset consists of 90% Class 0 (Non-Survived) and 10% Class 1 (Survived), 
   a naive model that predicts "0" for every single passenger will achieve 90% 
   accuracy without ever detecting a single survivor.
   
2. Masking Critical Errors (False Negatives): 
   In real-world applications (e.g., medical diagnostics, fraud detection, or 
   survival prediction), failing to identify a positive case (False Negative) is 
   often far worse than misclassifying a negative case (False Positive). 

3. The Solution:
   Precision measures exactness (out of all positive guesses, how many were right),
   Recall measures completeness (out of all actual positive cases, how many were caught),
   and F1-Score provides a balanced harmonic mean of both.


--------------------------------------------------------------------------------
3. BASELINE MODEL EVALUATION (UNBALANCED DATASET)
--------------------------------------------------------------------------------
Model: Logistic Regression (Default Parameters: C=1.0, penalty='l2', solver='liblinear')
Evaluated using sklearn.metrics.classification_report on holdout test set (N=137):

Overall Model Accuracy: 73.72%

              precision    recall  f1-score   support

           0       0.73      0.77      0.75        69
           1       0.75      0.71      0.73        68

    accuracy                           0.74       137
   macro avg       0.74      0.74      0.74       137
weighted avg       0.74      0.74      0.74       137

Baseline Observations:
- Accuracy: 73.72%
- Precision (Class 1 - Survived): 75.00%
- Recall (Class 1 - Survived): 70.59% (0.71 rounded in report)
- Macro F1-Score: 73.69%


--------------------------------------------------------------------------------
4. HYPERPARAMETER TUNING VIA GRIDSEARCHCV
--------------------------------------------------------------------------------
Hyperparameters Tuned (2 parameters):
1. 'C': Inverse of regularization strength (Tested: 0.01, 0.1, 0.5, 1.0, 10.0)
2. 'penalty': Regularization type (Tested: 'l1', 'l2')

Cross-Validation Configuration:
- Search Method: GridSearchCV
- CV Strategy: 5-Fold Stratified Cross-Validation
- Optimization Target Metric: Macro F1-Score

GridSearchCV Optimal Selection:
- Best Hyperparameters: {'C': 0.5, 'penalty': 'l1', 'solver': 'liblinear'}


--------------------------------------------------------------------------------
5. BEFORE/AFTER PERFORMANCE COMPARISON TABLE
--------------------------------------------------------------------------------
Evaluated on the exact same test set (N=137):

--------------------------------------------------------------------------------
Metric                  | Original Model | Tuned Model (GridSearch) | Delta
--------------------------------------------------------------------------------
Accuracy                | 0.7372 (73.72%)| 0.7080 (70.80%)          | -2.92%
Precision (Class 1)     | 0.7500 (75.00%)| 0.7258 (72.58%)          | -2.42%
Recall (Class 1)        | 0.7059 (70.59%)| 0.6618 (66.18%)          | -4.41%
Macro F1-Score          | 0.7369 (73.69%)| 0.7073 (70.73%)          | -2.96%



--------------------------------------------------------------------------------
6. TECHNICAL ROOT-CAUSE ANALYSIS OF RESULTS
--------------------------------------------------------------------------------
Why Did the Tuned Model Underperform the Baseline?

1. Over-Regularization via L1 Sparsity (C=0.5, penalty='l1'):
   L1 regularization (Lasso) shrinks less dominant feature weights down to zero, 
   effectively removing features from the decision function. Setting C=0.5 
   enforced stronger regularization than the default baseline (C=1.0, L2). 
   This extra penalty simplified the model too much, causing underfitting and 
   reducing Class 1 Recall from 70.59% to 66.18%.

2. Cross-Validation Average vs. Single Holdout Variance:
   GridSearchCV calculates performance averaged across 5 training validation folds. 
   While {'C': 0.5, 'penalty': 'l1'} performed best on average across the 5 training 
   folds, the baseline model with smooth L2 penalty generalized better on this 
   specific holdout test split (N=137).


--------------------------------------------------------------------------------
7. FINAL DECISION & README CONCLUSION
--------------------------------------------------------------------------------
- Hyperparameter tuning via cross-validation optimizes for average training fold 
  stability, but aggressive penalties like L1 can lead to underfitting on 
  holdout test data.
- For small tabular datasets with correlated attributes (like Titanic), L2 weight 
  decay (Ridge) preserves subtle predictive signals better than L1 zeroing (Lasso).
- Final Deployment Choice: The Original Baseline Model (C=1.0, penalty='l2') 
  is selected as the final production model for this task, as it retains higher 
  overall Accuracy (73.72%), Macro F1-Score (73.69%), and Class 1 Recall (70.59%).
================================================================================
