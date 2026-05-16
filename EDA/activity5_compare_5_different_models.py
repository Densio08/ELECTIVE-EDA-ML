import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")


print("\n" + "=" * 60)
print("SECTION 1: DATA PREPARATION")
print("=" * 60)

# 1a. Load the dataset and drop Employee_ID
df = pd.read_csv('employee_attrition_dataset_10000.csv')
df = df.drop(columns=['Employee_ID'])

print(f"\nDataset loaded. Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

# 1b. Encode categorical columns using LabelEncoder
categorical_cols = df.select_dtypes(include='object').columns.tolist()
print(f"\nCategorical columns to encode: {categorical_cols}")

le = LabelEncoder()
df_encoded = df.copy()

for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

print("\nEncoding done. Sample of encoded data:")
print(df_encoded[categorical_cols].head())

# 1c. Separate features (X) and target (y)
X = df_encoded.drop(columns=['Attrition'])
y = df_encoded['Attrition']

print(f"\nFeature matrix X shape: {X.shape}")
print(f"Target vector y shape:  {y.shape}")
print(f"\nTarget distribution:\n{y.value_counts()}")
print(f"\n  0 = Stayed (No), 1 = Left (Yes)")

# 1d. Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeature scaling applied using StandardScaler.")

# 1e. Train-Test Split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain-Test Split (80/20):")
print(f"  Training set:  {X_train.shape[0]} samples")
print(f"  Testing set:   {X_test.shape[0]} samples")


print("\n" + "=" * 60)
print("SECTION 2: MODEL TRAINING AND EVALUATION")
print("=" * 60)


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train a model, print evaluation metrics, plot confusion matrix,
    and return a dictionary of scores."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n{'-' * 50}")
    print(f"Model: {name}")
    print(f"{'-' * 50}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Stayed (No)', 'Left (Yes)']))

    # Confusion matrix heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted: No', 'Predicted: Yes'],
                yticklabels=['Actual: No', 'Actual: Yes'])
    plt.title(f'Confusion Matrix - {name}')
    plt.tight_layout()
    plt.show()

    return {
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1
    }


# --- Train all 5 models ---
results = []

# Model 1 - Logistic Regression
lr_model = LogisticRegression(random_state=42, max_iter=1000)
results.append(evaluate_model("Logistic Regression", lr_model,
                              X_train, X_test, y_train, y_test))

# Model 2 - Decision Tree
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
results.append(evaluate_model("Decision Tree", dt_model,
                              X_train, X_test, y_train, y_test))

# Model 3 - K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)
results.append(evaluate_model("K-Nearest Neighbors", knn_model,
                              X_train, X_test, y_train, y_test))

# Model 4 - Naive Bayes
nb_model = GaussianNB()
results.append(evaluate_model("Naive Bayes", nb_model,
                              X_train, X_test, y_train, y_test))

# Model 5 - Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
results.append(evaluate_model("Random Forest", rf_model,
                              X_train, X_test, y_train, y_test))


print("\n" + "=" * 60)
print("SECTION 3: MODEL COMPARISON TABLE")
print("=" * 60)

# Build comparison DataFrame
comparison_data = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': []
}

for res in results:
    comparison_data['Model'].append(res['Model'])
    comparison_data['Accuracy'].append(round(res['Accuracy'], 4))
    comparison_data['Precision'].append(round(res['Precision'], 4))
    comparison_data['Recall'].append(round(res['Recall'], 4))
    comparison_data['F1 Score'].append(round(res['F1 Score'], 4))

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.set_index('Model')

print("\n")
print(comparison_df.to_string())

# Highlight the best value per metric
print("\n\nBest values per metric:")
for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
    best_model = comparison_df[metric].idxmax()
    best_val   = comparison_df[metric].max()
    print(f"   {metric:<12}: {best_model} ({best_val:.4f})")

# Bar chart comparison
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
colors  = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

for ax, metric, color in zip(axes, metrics, colors):
    bars = ax.bar(comparison_df.index, comparison_df[metric],
                  color=color, alpha=0.85, edgecolor='black')
    ax.set_title(metric, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(comparison_df.index, rotation=35, ha='right', fontsize=8)
    for bar, val in zip(bars, comparison_df[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Model Comparison - Employee Attrition Classification (5 Models)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


print("\n" + "=" * 60)
print("SECTION 4: ANALYSIS AND DISCUSSION")
print("=" * 60)

best_acc_model = comparison_df['Accuracy'].idxmax()
best_rec_model = comparison_df['Recall'].idxmax()
best_f1_model  = comparison_df['F1 Score'].idxmax()

best_acc_val = comparison_df['Accuracy'].max()
best_rec_val = comparison_df['Recall'].max()
best_f1_val  = comparison_df['F1 Score'].max()

print(f"""
WHICH MODEL HAS THE HIGHEST ACCURACY?
   -> {best_acc_model} with {best_acc_val*100:.2f}% accuracy.

WHICH MODEL HAS THE HIGHEST RECALL?
   -> {best_rec_model} with a Recall of {best_rec_val:.4f}.

WHICH MODEL HAS THE MOST BALANCED PERFORMANCE?
   -> {best_f1_model} with an F1 Score of {best_f1_val:.4f}.
""")


print("\n" + "=" * 60)
print("SECTION 5: MODELS SELECTED FOR TUNING")
print("=" * 60)

# Rank models by F1 Score (best overall balance of precision & recall)
ranked = comparison_df.sort_values('F1 Score', ascending=False)
top_two = ranked.head(2)

print("\nRanking by F1 Score (descending):\n")
print(ranked[['F1 Score', 'Accuracy', 'Precision', 'Recall']].to_string())

model_1 = top_two.index[0]
model_2 = top_two.index[1]
f1_1    = top_two['F1 Score'].iloc[0]
f1_2    = top_two['F1 Score'].iloc[1]

print(f"\n>> Models selected for the tuning stage:")
print(f"   1. {model_1}  (F1 Score = {f1_1:.4f})")
print(f"   2. {model_2}  (F1 Score = {f1_2:.4f})")


print("\n" + "=" * 60)
print("SECTION 6: BASIS FOR SELECTION (JUSTIFICATION)")
print("=" * 60)

print(f"""
Among the five classification models evaluated on the Employee
Attrition dataset, {model_1} and {model_2} were selected as the
two models to advance to the hyperparameter-tuning stage. The
selection was based primarily on the F1 Score, which balances
Precision and Recall -- a critical consideration for this imbalanced
dataset, where accurately identifying employees likely to leave
(the minority class) is more important than simply achieving high
overall accuracy. {model_1} achieved the highest F1 Score of
{f1_1:.4f}, demonstrating the strongest ability to detect true
attrition cases while maintaining reasonable precision.
{model_2} followed with an F1 Score of {f1_2:.4f}, also
outperforming the remaining models on this metric. Together,
these two models represent the best candidates for further
improvement through hyperparameter tuning, as they already show
meaningful predictive capability on the minority class and are
likely to benefit the most from fine-tuned parameters.
""")


print("\n" + "=" * 60)
print("SECTION 7: CONCLUSION")
print("=" * 60)

print(f"""
We trained and compared five classification models
on the Employee Attrition dataset using the same 80/20 stratified
train-test split and StandardScaler preprocessing:

   1. Logistic Regression
   2. Decision Tree Classifier
   3. K-Nearest Neighbors (KNN)
   4. Naive Bayes
   5. Random Forest

Comparison Table:
""")
print(comparison_df.to_string())

print(f"""

Key Findings:
   * {best_acc_model} achieved the highest Accuracy ({best_acc_val*100:.2f}%).
   * {best_rec_model} achieved the highest Recall ({best_rec_val:.4f}).
   * {best_f1_model} showed the most balanced performance (F1 = {best_f1_val:.4f}).

Models Selected for Tuning:
   -> {model_1} (F1 = {f1_1:.4f})
   -> {model_2} (F1 = {f1_2:.4f})
   
""")
