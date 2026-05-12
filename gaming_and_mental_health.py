import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("Gaming and Mental Health.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Drop ID column — not a useful feature
df = df.drop(columns=["record_id"])

# Convert bool columns to int so StandardScaler accepts them
bool_cols = ["withdrawal_symptoms", "loss_of_other_interests",
             "continued_despite_problems", "eye_strain", "back_neck_pain"]
df[bool_cols] = df[bool_cols].astype(int)

X = df.drop("gaming_addiction_risk_level", axis=1)
Y = df["gaming_addiction_risk_level"]

X = pd.get_dummies(X, drop_first=True)

# Fill any NaNs that arise from encoding or missing values
X = X.fillna(X.median(numeric_only=True))

print("\nFeatures shape after encoding:", X.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, Y, test_size=0.25, random_state=42
)

print("\nTrain Size:", X_train.shape)
print("Test Size:", X_test.shape)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=1),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n{'='*40}")
    print(f"    {name}")
    print(f" Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

print("\n" + "="*40)
print(" Model Accuracy Summary")
print("="*40)
for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:<25} {acc:.2%}")

best = max(results, key=results.get)
print(f"\n Best Model: {best} ({results[best]:.2%})")

print("\n" + "="*40)
print("  DATASET INSIGHTS")
print("="*40)

print("\nAverage usage per Addiction Risk Level:")
print(df.groupby("gaming_addiction_risk_level")[[
    "daily_gaming_hours", "sleep_hours", "grades_gpa",
    "work_productivity_score", "social_isolation_score",
    "exercise_hours_weekly", "years_gaming", "monthly_game_spending_usd"
]].mean().round(2).to_string())

print("\nOverall Averages:")
print(f"  Daily Gaming Hours     : {df['daily_gaming_hours'].mean():.2f} hours/day")
print(f"  Sleep Hours            : {df['sleep_hours'].mean():.2f} hours/day")
print(f"  GPA / Grades           : {df['grades_gpa'].mean():.2f}")
print(f"  Work Productivity Score: {df['work_productivity_score'].mean():.2f}")
print(f"  Social Isolation Score : {df['social_isolation_score'].mean():.2f}")
print(f"  Exercise (weekly)      : {df['exercise_hours_weekly'].mean():.2f} hours")
print(f"  Monthly Game Spending  : ${df['monthly_game_spending_usd'].mean():.2f}")
print(f"  Average Age            : {df['age'].mean():.2f} years old")

print("\nGender Distribution:")
print(df["gender"].value_counts().to_string())

print("\nMost Common Game Genre:")
print(df["game_genre"].value_counts().head(3).to_string())

print("\nGaming Platform Distribution:")
print(df["gaming_platform"].value_counts().to_string())


plt.figure()
df["gaming_addiction_risk_level"].value_counts().plot(kind="bar")
plt.title("Gaming Addiction Risk Level Distribution")
plt.xlabel("Risk Level")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

plt.figure()
df["gender"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Gender Distribution")
plt.tight_layout()
plt.show()

plt.figure()
df["game_genre"].value_counts().head(5).plot(kind="bar")
plt.title("Top 5 Game Genres")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

plt.figure()
df[["daily_gaming_hours", "sleep_hours", "exercise_hours_weekly"]].mean().plot(kind="bar")
plt.title("Overall Average Hours (Gaming / Sleep / Exercise)")
plt.ylabel("Average Hours")
plt.tight_layout()
plt.show()