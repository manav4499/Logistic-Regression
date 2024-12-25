import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
import seaborn as sns

# Load the data
studentdropout_Manav = pd.read_csv('studentsdropout.csv')

# Initial Exploration
print("\nFirst 5 records:")
print(studentdropout_Manav.head())

print("\nDataframe Info:")
studentdropout_Manav.info()

print("\nShape of the dataframe:")
print(studentdropout_Manav.shape)

print("\nUnique values in Educational special needs:")
print(studentdropout_Manav['Educational special needs'].unique())
print("\nUnique values in Displaced:")
print(studentdropout_Manav['Displaced'].unique())

# Data Visualization
plt.figure(figsize=(10, 6))
gender_success = pd.crosstab(studentdropout_Manav['Gender'], studentdropout_Manav['Academic_success '])
gender_success.plot(kind='bar')
plt.title('Gender vs Academic Success by Manav Chaudhary')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('gender_success.png')
plt.close()

plt.figure(figsize=(10, 6))
fees_success = pd.crosstab(studentdropout_Manav['Tuition fees up to date'],studentdropout_Manav['Academic_success '])
fees_success.plot(kind='bar')
plt.title('Tuition Fees Status vs Academic Success by Manav Chaudhary')
plt.xlabel('Tuition Fees Up to Date')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('fees_success.png')
plt.close()

# Scatter Matrix
numeric_cols = ['Admission grade', 'Age at enrollment',
                'Curricular units 1st sem (grade)',
                'Curricular units 2nd sem (grade)']
pd.plotting.scatter_matrix(studentdropout_Manav[numeric_cols], figsize=(12, 12))
plt.tight_layout()
plt.savefig('scatter_matrix.png')
plt.close()
#
# Data Transformation

# Get dummies for categorical variables
categorical_cols = ['Marital status', 'Course', 'Daytime/evening attendance',
                   'Displaced', 'Educational special needs', 'Tuition fees up to date',
                   'Gender', 'Scholarship holder', 'International', 'Academic_success ']

df_encoded = pd.get_dummies(studentdropout_Manav[categorical_cols],drop_first=True)

numeric_cols = ['Admission grade', 'Age at enrollment',
                'Curricular units 1st sem (grade)',
                'Curricular units 2nd sem (grade)']

# Combine numeric and encoded categorical columns
df_transformed = pd.concat([studentdropout_Manav[numeric_cols], df_encoded], axis=1)

# Replace missing values in Age
df_transformed['Age at enrollment'] = df_transformed['Age at enrollment'].fillna(df_transformed['Age at enrollment'].mean())

# Convert all to float
df_transformed = df_transformed.astype(float)
#
# # Normalization function
def normalize_dataframe(df):
    return (df - df.min()) / (df.max() - df.min())

df_normalized = normalize_dataframe(df_transformed)

print("\nFirst two records of normalized data:")
print(df_normalized.head(2))

# Generate histograms
df_normalized.hist(figsize=(9, 10))
plt.tight_layout()
plt.savefig('all_histograms.png')
plt.close()

# # Specific histograms for curricular units
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(studentdropout_Manav['Curricular units 1st sem (grade)'])
plt.title('1st Semester Grades Distribution')
plt.subplot(1, 2, 2)
plt.hist(studentdropout_Manav['Curricular units 2nd sem (grade)'])
plt.title('2nd Semester Grades Distribution')
plt.tight_layout()
plt.savefig('semester_grades.png')
plt.close()


# Prepare features and target
target_col = 'Academic_success _Graduate'

x_Manav = df_normalized.drop(['Academic_success _Graduate'], axis=1)
y_Manav = df_normalized[target_col]


# Split data
x_train_Manav, x_test_Manav, y_train_Manav, y_test_Manav = train_test_split(
    x_Manav, y_Manav, test_size=0.3, random_state=23)

# Build model
model = LogisticRegression(random_state=23)
model.fit(x_train_Manav, y_train_Manav)
#
# Display coefficients
coef_df = pd.DataFrame(
    zip(x_train_Manav.columns, np.transpose(model.coef_)),
    columns=['Feature', 'Coefficient']
)
print("\nModel Coefficients:")
print(coef_df)

# Cross validation with different splits
print("\nCross Validation Results:")
for i in np.arange(0.10, 0.55, 0.05):
    x_train, x_test, y_train, y_test = train_test_split(
        x_Manav, y_Manav, test_size=i, random_state=23)
    scores = cross_val_score(model, x_train, y_train, cv=10)
    print(f"\nTest size: {i:.2f}")
    print(f"Min accuracy: {scores.min():.4f}")
    print(f"Mean accuracy: {scores.mean():.4f}")
    print(f"Max accuracy: {scores.max():.4f}")

# Final model evaluation
y_pred_Manav = model.predict_proba(x_test_Manav)
y_pred_Manav_flag = y_pred_Manav[:, 1] > 0.5

print("\nModel Evaluation (threshold=0.5):")
print("\nAccuracy:", accuracy_score(y_test_Manav, y_pred_Manav_flag))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_Manav, y_pred_Manav_flag))
print("\nClassification Report:")
print(classification_report(y_test_Manav, y_pred_Manav_flag))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test_Manav, y_pred_Manav[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.close()

# Evaluation with threshold 0.60
y_pred_claude_flag_60 = y_pred_Manav[:, 1] > 0.60

print("\nModel Evaluation (threshold=0.60):")
print("\nAccuracy:", accuracy_score(y_test_Manav, y_pred_claude_flag_60))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_Manav, y_pred_claude_flag_60))
print("\nClassification Report:")
print(classification_report(y_test_Manav, y_pred_claude_flag_60))