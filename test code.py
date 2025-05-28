## importing libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# loading the dataset
data = pd.read_csv("Location here")

## Start of Data handling 

## Data preview for understanding
##print("Dataset Preview")
##print(data.head())

## Data cleaning - Missing Values
## --- print("\nmissing values in every column")
## --- print(data.isnull().sum())

## Dropping columns with less than 20% vales that are missing to help with data cleaning
threshold = 0.2  # 20% missing data 
missing_percent = data.isnull().sum() / len(data)
data = data.drop(columns=missing_percent[missing_percent > threshold].index)


## Filling numerical missing values with the median of the entire ds
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
   data[col] = data[col].fillna(data[col].median())

## Filling missing values with the most mode of the entire ds
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])  # Fixed: mode() returns a Series

## Confirm missing values are being handled
print("\nMissing Values After Cleaning:", data.isnull().sum().sum())

## Data handling and feature extraction 
# For correlation analysis, use the same dataframe (data) to avoid confusion
# find columns with missing values in original dataset
df = pd.read_csv("C:\\Users\\encry\\Desktop\\cw1-ai\\TUDA-Reduced.csv") 
missing_cols = df.columns[df.isnull().sum() > 0]

# find correlation matrix
numerical_df = df.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numerical_df.corr()

# Finding matching features for each missing column
for col in missing_cols:
    if col in numerical_df.columns:  # Making sure columns are numerically displayed
        print(f"\nTop Correlations for {col}:")
        print(correlation_matrix[col].abs().sort_values(ascending=False).head(6))

## removing fab and mmse
df = pd.read_csv(r"C:\Users\encry\Desktop\cw1-ai\cleaned_dataset.csv")
df = df.drop(columns=['TotalFAB', 'MMSETotal'], errors='ignore')


selected_features = [
    "Bodymassindex", "WaistHip", "HbA1C", "Cholesterol", "LDL", 
     "Timedupngo", "Numberofcigarettesperday"
]

# Save cleaned dataset
data.to_csv("C:\\Users\\encry\\Desktop\\cw1-ai\\cleaned_dataset.csv", index=False)

## Visualising dataset 
# First, check which columns actually exist in our dataset
print("\nAvailable columns in cleaned dataset:")
print(data.columns.tolist())

# Identify which of our desired columns actually exist in the dataset
available_important_features = []
column_mapping = {
    "Waistmeasurementincm": ["Waistmeasurementincm", "Waist", "WaistCircumference"],
    "Weightinkg": ["Weightinkg", "Weight", "Weightkg"],
    "Bodymassindex": ["Bodymassindex", "BMI"],
    "Hipmeasurementincm": ["Hipmeasurementincm", "Hip", "HipCircumference"]
}

# Find the actual column names in the dataset
actual_important_features = []
for target, alternatives in column_mapping.items():
    found = False
    for alt in alternatives:
        if alt in data.columns:
            actual_important_features.append(alt)
            found = True
            break
    if not found:
        print(f"Warning: Could not find any column matching '{target}'")

print("\nColumns that will be used for visualization:", actual_important_features)

# Only proceed with visualization if we have columns to visualize
if len(actual_important_features) >= 2:
    # Create correlation matrix visualization
    plt.figure(figsize=(12, 8))
    corr_matrix = data[actual_important_features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Heatmap of Correlations Between Selected Features")
    plt.tight_layout()
    plt.savefig("C:\\Users\\encry\\Desktop\\cw1-ai\\correlation_heatmap.png")
    plt.close()  # Close to avoid displaying in non-interactive environments
    print("Correlation heatmap saved")
    
    # Pairplot visualization
    pairplot = sns.pairplot(data[actual_important_features], diag_kind="kde")
    pairplot.fig.suptitle("Pairplot of Selected Features", y=1.02)
    plt.tight_layout()
    pairplot.savefig("C:\\Users\\encry\\Desktop\\cw1-ai\\pairplot.png")
    plt.close()
    print("Pairplot saved")
    
    # Boxplot visualization
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data[actual_important_features])
    plt.title("Boxplot of Selected Features")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("C:\\Users\\encry\\Desktop\\cw1-ai\\boxplot.png")
    plt.close()
    print("Boxplot saved")
    
    # Scatterplot (if we have at least 3 features)
    if len(actual_important_features) >= 3:
        plt.figure(figsize=(8, 6))
        x_col = actual_important_features[0]
        y_col = actual_important_features[1]
        hue_col = actual_important_features[2]
        
        sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue_col, palette="coolwarm")
        plt.title(f"Scatterplot of {x_col} vs {y_col} (colored by {hue_col})")
        plt.tight_layout()
        plt.savefig("C:\\Users\\encry\\Desktop\\cw1-ai\\scatterplot.png")
        plt.close()
        print("Scatterplot saved")
else:
    print("\nimages saved in the directory.")

# Calculate basic statistics for numerical features
print("\nBasic Statistics for Numerical Features:")
print(data.describe())

# Import libraries for dimensionality reduction
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Select only numerical columns for PCA
numerical_data = data.select_dtypes(include=['int64', 'float64'])

# Handle any remaining missing values for PCA
numerical_data = numerical_data.fillna(numerical_data.mean())

# Standardize the data (important for PCA)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# Apply PCA
pca = PCA()
pca_data = pca.fit_transform(scaled_data)

# Calculate explained variance ratio
explained_variance = pca.explained_variance_ratio_

# Plot explained variance to determine optimal number of components
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center')
plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.title('Explained Variance by Components')
plt.savefig("C:\\Users\\encry\\Desktop\\cw1-ai\\pca_variance.png")
plt.close()
print("PCA variance plot saved")

# Select optimal number of components (e.g., explaining 80% of variance)
cumulative_variance = np.cumsum(explained_variance)
n_components = np.argmax(cumulative_variance >= 0.8) + 1
print(f"\nNumber of components explaining 80% of variance: {n_components}")

# Apply PCA with optimal number of components
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(scaled_data)

# Create DataFrame with PCA results
pca_df = pd.DataFrame(
    data=pca_result,
    columns=[f'PC{i+1}' for i in range(n_components)]
)

# Save PCA results
pca_df.to_csv("C:\\Users\\encry\\Desktop\\cw1-ai\\pca_results.csv", index=False)
print("PCA results saved to CSV")

# For this example, I'll use 'Cognitive_Status' if it exists, but you should replace this with your actual target
target_column = None
potential_target_columns = ['Cognitive_Status', 'DiagnosisStatus', 'CognitiveStatus', 'Diagnosis']

for col in potential_target_columns:
    if col in data.columns:
        target_column = col
        print(f"Using '{target_column}' as the target variable for classification.")
        break

if target_column is None:
    print("Could not identify a suitable target column. specify a target variable.")
    # As a fallback, i manually set a target column from the available columns
    print("Available columns:", data.columns.tolist())
    # I'll create a dummy target for demonstration purposes
    data['Dummy_Target'] = (data[numerical_cols[0]] > data[numerical_cols[0]].median()).astype(int)
    target_column = 'Dummy_Target'
    print(f"Created dummy target variable '{target_column}' for demonstration.")

# Get the target variable
y = data[target_column]

# For features, I'll use either PCA results or selected numerical features
# Option 1: Using PCA results as features
X_pca = pca_df.values

# Option 2: Using selected numerical features
selected_numerical_cols = [col for col in numerical_cols if col != target_column]
X_original = data[selected_numerical_cols].values

# Split the data into training and testing sets for both feature sets
X_pca_train, X_pca_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)
X_orig_train, X_orig_test, y_train_orig, y_test_orig = train_test_split(X_original, y, test_size=0.3, random_state=42)

# Standardize the original features (PCA data is already standardized)
scaler = StandardScaler()
X_orig_train_scaled = scaler.fit_transform(X_orig_train)
X_orig_test_scaled = scaler.transform(X_orig_test)

# Function to evaluate and compare models
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, feature_type):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Print results
    print(f"\n{model_name} with {feature_type} features:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name} with {feature_type} features')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"C:\\Users\\encry\\Desktop\\cw1-ai\\cm_{model_name}_{feature_type}.png")
    plt.close()
    
    # Detailed classification report
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)
    
    return accuracy, f1, model

# Define models to compare
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'Neural Network': MLPClassifier(random_state=42, max_iter=1000)
}

# Evaluate each model with the feature sets
results = []
for model_name, model in models.items():
    # Evaluate with PCA features
    acc_pca, f1_pca, _ = evaluate_model(model, X_pca_train, X_pca_test, y_train, y_test, model_name, 'PCA')
    results.append({'Model': model_name, 'Features': 'PCA', 'Accuracy': acc_pca, 'F1': f1_pca})
    
    # Evaluate with original features
    acc_orig, f1_orig, _ = evaluate_model(model, X_orig_train_scaled, X_orig_test_scaled, y_train_orig, y_test_orig, model_name, 'Original')
    results.append({'Model': model_name, 'Features': 'Original', 'Accuracy': acc_orig, 'F1': f1_orig})

# Create a results DataFrame
results_df = pd.DataFrame(results)
print("\nModel Comparison Summary:")
print(results_df)
results_df.to_csv("C:\\Users\\encry\\Desktop\\cw1-ai\\model_comparison_results.csv", index=False)



# Find the best model based on F1 score
best_model_idx = results_df['F1'].idxmax()
best_model_info = results_df.iloc[best_model_idx]
print(f"\nBest model: {best_model_info['Model']} with {best_model_info['Features']} features (F1: {best_model_info['F1']:.4f})")



# Now I'll perform hyperparameter tuning for the best model
print("\n---------- Hyperparameter Tuning ----------")

# Determine which model and features to tune
best_model_name = best_model_info['Model']
best_features_type = best_model_info['Features']



# Select the appropriate feature set
if best_features_type == 'PCA':
    X_train, X_test = X_pca_train, X_pca_test
else:
    X_train, X_test = X_orig_train_scaled, X_orig_test_scaled



# Defining parameter grids for each model !!!
param_grids = {
    'Decision Tree': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'linear']
    },
    'Neural Network': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01]
    }
}

# Get the parameter grid for the best model
if best_model_name in param_grids:
    param_grid = param_grids[best_model_name]
    
    # Get a fresh instance of the best model
    best_model = models[best_model_name]
    
    # Performing grid search with cross-validation
    grid_search = GridSearchCV(
        best_model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # search to best parameters and results
    print(f"\nBest parameters for {best_model_name}:")
    print(grid_search.best_params_)
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluating my tuned model on the test set ( could be very good or very bad :( )
    tuned_model = grid_search.best_estimator_
    tuned_accuracy, tuned_f1, _ = evaluate_model(
        tuned_model, X_train, X_test, y_train, y_test, 
        f"{best_model_name} (Tuned)", best_features_type
    )
    
    print(f"\nImprovement after tuning:")
    print(f"F1 Score: {best_model_info['F1']:.4f} -> {tuned_f1:.4f}")
    
    # K-fold cross-validation of the tuned model
    print("\n---------- K-Fold Cross-Validation ----------")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(tuned_model, X_train, y_train, cv=kf, scoring='f1_weighted')
    
    print(f"5-Fold Cross-Validation F1 Scores for {best_model_name} (Tuned):")
    for i, score in enumerate(cv_scores):
        print(f"Fold {i+1}: {score:.4f}")
    print(f"Mean F1: {cv_scores.mean():.4f}, Std: {cv_scores.std():.4f}")
    
else:
    print(f"Parameter grid not defined for {best_model_name}. Skipping hyperparameter tuning.")

print("\nClassification complete.")




############ ------------------ References ------------------------

## machine learning for dummies ed 1 
## stack overflow 
