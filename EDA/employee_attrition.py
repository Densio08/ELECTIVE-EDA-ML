import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# Task 1: Load the dataset in pandas and display the first few rows.
print("=== Task 1: Load the dataset and display the first few rows ===")
df = pd.read_csv('employee_attrition_dataset_10000.csv')
if 'Employee_ID' in df.columns:
    df = df.drop(columns=['Employee_ID'])
print(df.head())
print("\n")

# Task 2: Show the dataset shape, column names, and data types.
print("=== Task 2: Dataset shape, column names, and data types ===")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nColumn Names:")
print(df.columns.tolist())
print("\nData Types:")
print(df.dtypes)
print("\n")

# Task 3: Check for missing values and duplicate rows.
print("=== Task 3: Check for missing values and duplicate rows ===")
print("Missing Values per column:")
print(df.isnull().sum())
print(f"\nTotal Duplicate Rows: {df.duplicated().sum()}")
print("\n")

# Task 4: Show the distribution of the target variable.
# Target variable is Attrition
print("=== Task 4: Distribution of the target variable (Attrition) ===")
print(df['Attrition'].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x='Attrition', data=df, palette='viridis')
plt.title('Task 4: Distribution of Employee Attrition')
plt.show()

print("=== Task 5: Generating 5 EDA Visualizations... ===")

# 1. Overall Attrition Rate
plt.figure(figsize=(6, 6))
df['Attrition'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#66b3ff','#ff9999'], startangle=90, explode=(0, 0.1))
plt.title('1. Overall Workforce Attrition Rate')
plt.ylabel('')
plt.show()

# 2. Workforce Distribution by Department
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Department', hue='Attrition', palette='viridis')
plt.title('2. Workforce Distribution by Department')
plt.ylabel('Number of Employees')
plt.show()

# 3. Workforce by Job Role
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Job_Role', hue='Attrition', palette='magma')
plt.title('3. Workforce Distribution by Job Role')
plt.ylabel('Number of Employees')
plt.show()

# 4. Monthly Income Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='Monthly_Income', hue='Attrition', bins=20, kde=True, multiple='stack', palette='Set2')
plt.title('4. Monthly Income Distribution')
plt.xlabel('Monthly Income')
plt.ylabel('Frequency')
plt.show()

# 5. Job Satisfaction Levels
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Job_Satisfaction', hue='Attrition', palette='coolwarm')
plt.title('5. Job Satisfaction Levels')
plt.xlabel('Job Satisfaction Level (1=Low, 5=High)')
plt.ylabel('Number of Employees')
plt.show()

"""
Task 6: Observations 

1. Attrition Rate: The dataset shows an imbalance and the company has an overall attrition rate of
~20%, meaning a solid ~80% of the workforce is retained. This 80/20 baseline is crucial for interpreting
the rest of the workforce data.

2. Departmental Uniformity: When looking at the departments, the ratio of retained ("No") to
departing ("Yes") employees is remarkably consistent. Turnover is a company-wide phenomenon, as
no single department shows a disproportionately high risk of attrition compared to its total size.

3. Job Role Consistency: Both retained and departing employees are evenly distributed across all
job roles. Whether an employee is an Assistant, Analyst, or Executive, the likelihood of them leaving
remains parallel to the overall 20% company average.

4. Income Independence: The income distribution is identical for both groups. Employees who stay
and those who leave span the exact same salary brackets and share the same median. This powerfully
indicates that low compensation is not the primary driver of attrition in this organization.

5. The Satisfaction Paradox: Surprisingly, the vast majority of the workforce reports high Job
Satisfaction (levels 4 and 5), and this trend holds true even for those who left. This suggests the
company maintains a highly positive environment, and departing employees are likely leaving for
external reasons (e.g., career pivots, relocation, or outside offers) rather than dissatisfaction with their
current jobs.

"""