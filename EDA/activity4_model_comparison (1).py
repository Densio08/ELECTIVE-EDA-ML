import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

print("\n" + "="*60)
print("SECTION 1: DATA PREPARATION")
print("="*60)

df = pd.read_csv('employee_attrition_dataset_10000.csv')
df = df.drop(columns=['Employee_ID'])

print(f"\nDataset loaded. Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

categorical_cols = df.select_dtypes(include='object').columns.tolist()
print(f"\nCategorical columns to encode: {categorical_cols}")

le = LabelEncoder()
df_encoded = df.copy()

for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

print("\nEncoding done. Sample of encoded data:")
print(df_encoded[categorical_cols].head())

X = df_encoded.drop(columns=['Attrition'])
y = df_encoded['Attrition']

print(f"\nFeature matrix X shape: {X.shape}")
print(f"Target vector y shape:  {y.shape}")
print(f"\nTarget distribution:\n{y.value_counts()}")
print(f"\n  0 = Stayed (No), 1 = Left (Yes)")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeature scaling applied using StandardScaler.")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain-Test Split (80/20):")
print(f"  Training set:  {X_train.shape[0]} samples")
print(f"  Testing set:   {X_test.shape[0]} samples")


print("\n" + "="*60)
print("SECTION 2: MODEL TRAINING AND EVALUATION")
print("="*60)


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n{'─'*50}")
    print(f"Model: {name}")
    print(f"{'─'*50}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Stayed (No)', 'Left (Yes)']))

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted: No', 'Predicted: Yes'],
                yticklabels=['Actual: No', 'Actual: Yes'])
    plt.title(f'Confusion Matrix – {name}')
    plt.tight_layout()
    plt.show()

    return {'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1}


results = []

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_results = evaluate_model("Logistic Regression", lr_model, X_train, X_test, y_train, y_test)
results.append(lr_results)


dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_results = evaluate_model("Decision Tree", dt_model, X_train, X_test, y_train, y_test)
results.append(dt_results)


knn_model = KNeighborsClassifier(n_neighbors=5)
knn_results = evaluate_model("K-Nearest Neighbors (KNN)", knn_model, X_train, X_test, y_train, y_test)
results.append(knn_results)


nb_model = GaussianNB()
nb_results = evaluate_model("Naive Bayes", nb_model, X_train, X_test, y_train, y_test)
results.append(nb_results)


print("\n" + "="*60)
print("SECTION 3: MODEL COMPARISON TABLE")
print("="*60)

results_df = pd.DataFrame(results)
results_df = results_df.set_index('Model')
results_df = results_df.round(4)

print("\nSummary of All Model Performances:\n")
print(results_df.to_string())

print("\nBest values per metric:")
for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
    best_model = results_df[metric].idxmax()
    best_val   = results_df[metric].max()
    print(f"   {metric:<12}: {best_model} ({best_val:.4f})")

fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
colors  = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

for ax, metric, color in zip(axes, metrics, colors):
    bars = ax.bar(results_df.index, results_df[metric], color=color, alpha=0.85, edgecolor='black')
    ax.set_title(metric, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(results_df.index, rotation=30, ha='right', fontsize=9)
    for bar, val in zip(bars, results_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Model Comparison – Employee Attrition Classification', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()


print("\n" + "="*60)
print("SECTION 4: ANALYSIS AND DISCUSSION")
print("="*60)

best_acc_model = results_df['Accuracy'].idxmax()
best_rec_model = results_df['Recall'].idxmax()
best_f1_model  = results_df['F1 Score'].idxmax()

best_acc_val = results_df['Accuracy'].max()
best_rec_val = results_df['Recall'].max()
best_f1_val  = results_df['F1 Score'].max()

print(f"""
WHICH MODEL HAS THE HIGHEST ACCURACY?
   → {best_acc_model} with {best_acc_val*100:.2f}% accuracy.

WHICH MODEL HAS THE HIGHEST RECALL?
   → {best_rec_model} with a Recall of {best_rec_val:.4f}.

WHICH MODEL HAS THE MOST BALANCED PERFORMANCE?
   → {best_f1_model} with an F1 Score of {best_f1_val:.4f}.

WHICH MODEL SHOULD BE CARRIED FORWARD?
   → {best_f1_model} should be carried forward.
""")


print("\n" + "="*60)
print("SECTION 5: REFLECTION")
print("="*60)

print("""
"A model with the highest accuracy is not always the best model.
Explain this statement using a classification example related to
your PIT dataset."

─────────────────────────────────────────────────────────────

ANSWER:
High accuracy doesn’t always mean that the model is actually good, because accuracy only counts the correct
predictions overall without showing where the model is truly failing. When a dataset is imbalanced, a model can
score high accuracy simply by predicting the majority class every single time without learning anything that is
useful. This is clearly seen with Logistic Regression in the activity, which obtained the highest accuracy at
80.05% but had a Recall and F1 Score of 0.0000, which means that it failed to identify even one employee who
was actually going to leave.
""")


print("\n" + "="*60)
print("SECTION 6: CONCLUSION")
print("="*60)

print(f"""
In this activity, we trained and compared four classification models
on the Employee Attrition dataset:

  1. Logistic Regression
  2. Decision Tree Classifier
  3. K-Nearest Neighbors (KNN)
  4. Naive Bayes

  ✔ {best_acc_model} achieved the highest Accuracy ({best_acc_val*100:.2f}%).
  ✔ {best_rec_model} achieved the highest Recall ({best_rec_val:.4f}).
  ✔ {best_f1_model} showed the most balanced performance (F1 = {best_f1_val:.4f})
    and is recommended as the model to carry forward.
""")