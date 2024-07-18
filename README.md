# Logistic Regression Classification Task

This repository contains the solution to the classification problem using logistic regression. The quality of the model is evaluated by calculating the metrics TPR (True Positive Rate), FPR (False Positive Rate), and plotting the ROC (Receiver Operating Characteristic) and Precision-Recall curves.

## Task Description

The task requires building a logistic regression model to classify whether an athlete has won any medals. The following steps are performed:

1. **Load and preprocess the data**.
2. **Train the logistic regression model**.
3. **Evaluate the model** using ROC and Precision-Recall metrics.
4. **Visualize** the results with corresponding curves.

## Solution

### Data Loading and Preprocessing

1. Load the dataset containing athletes' information.
2. Create a target column indicating whether the athlete has won any medals.
3. Handle missing values and encode categorical variables.
4. Split the data into training and testing sets.

### Logistic Regression Model

1. Train a logistic regression model.
2. Predict probabilities for the test set.
3. Calculate TPR, FPR, and plot the ROC and Precision-Recall curves.

### Code

The code implementation is as follows:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('athletes.csv')

# Create target column
data['target'] = (data['gold'] + data['silver'] + data['bronze']) > 0
data['target'] = data['target'].astype(int)

# Drop unnecessary columns
data = data.drop(['id', 'name', 'dob'], axis=1)

# Handle missing values
data['height'].fillna(data['height'].mean(), inplace=True)
data['weight'].fillna(data['weight'].mean(), inplace=True)

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split data into features and target
X = data.drop('target', axis=1)
y = data['target']

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict probabilities
y_scores = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = roc_auc_score(y_test, y_scores)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Calculate metrics manually
thresholds_manual = sorted(set(y_scores), reverse=True)
tpr_manual = []
fpr_manual = []

for threshold in thresholds_manual:
    tp = fp = fn = tn = 0
    for i in range(len(y_test)):
        if y_scores[i] >= threshold:
            if y_test.iloc[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if y_test.iloc[i] == 1:
                fn += 1
            else:
                tn += 1
    tpr_manual.append(tp / (tp + fn))
    fpr_manual.append(fp / (fp + tn))

# Plot manual ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label='ROC curve (sklearn)')
plt.plot(fpr_manual, tpr_manual, label='ROC curve (manual)', linestyle='--')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Plot Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_scores)

plt.figure(figsize=(10, 6))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Calculate manual ROC-AUC score
roc_auc_manual = 0.0
for i in range(1, len(fpr_manual)):
    roc_auc_manual += (fpr_manual[i] - fpr_manual[i-1]) * (tpr_manual[i] + tpr_manual[i-1]) / 2

print(f'ROC-AUC (manual): {roc_auc_manual:.2f}')
```

## Requirements

- pandas
- scikit-learn
- matplotlib

You can install the required packages using the following command:

```bash
pip install pandas scikit-learn matplotlib
```

## Usage

To run the code, simply execute the script in your Python environment.

## Results

The ROC curve and Precision-Recall curve will be plotted, and the ROC-AUC score will be calculated both automatically and manually, providing insights into the model's performance.
