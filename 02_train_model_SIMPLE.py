# =====================================================
# PROJECT 2: MACHINE LEARNING MODEL TRAINING
# Simple, readable code with detailed comments
# =====================================================

# === Import libraries ===
import pandas as pd  # For data tables
import numpy as np   # For numbers
from sklearn.model_selection import train_test_split  # Split data for training/testing
from sklearn.preprocessing import StandardScaler  # Make numbers same scale
from sklearn.ensemble import RandomForestClassifier  # ML algorithm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score  # Evaluation
import pickle  # For saving model
import warnings  # Hide warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("CUSTOMER CHURN PREDICTION - MACHINE LEARNING MODEL")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n📂 STEP 1: Loading data...")
print("-"*80)

try:
    # Load master dataset
    master = pd.read_csv('master_data.csv')
    print(f"✓ Loaded data: {len(master)} rows x {len(master.columns)} columns")
    
except FileNotFoundError:
    print("❌ Error: master_data.csv not found")
    print("Run 01_generate_churn_data_SIMPLE.py first!")
    exit()

# ============================================================================
# STEP 2: EXPLORE DATA
# ============================================================================
print("\n📊 STEP 2: Exploring data...")
print("-"*80)

# Show target variable (what we're predicting)
churn_counts = master['churn_flag'].value_counts()
print(f"Churn Distribution:")
print(f"   • No Churn: {churn_counts[0]:,} customers ({(churn_counts[0]/len(master)*100):.1f}%)")
print(f"   • Churned: {churn_counts[1]:,} customers ({(churn_counts[1]/len(master)*100):.1f}%)")

# Show basic statistics
print(f"\nData Shapes:")
print(f"   • Account Age: min={master['account_age_days'].min()}, max={master['account_age_days'].max()}")
print(f"   • Days Since Login: min={master['days_since_last_login'].min()}, max={master['days_since_last_login'].max()}")
print(f"   • Total Revenue: min=${master['total_revenue'].min():.2f}, max=${master['total_revenue'].max():.2f}")

# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================
print("\n⚙️  STEP 3: Creating features...")
print("-"*80)

df = master.copy()  # Work with a copy

# --- Feature 1: Encode gender (Male=1, Female=0) ---
df['gender_encoded'] = (df['gender'] == 'Male').astype(int)
print("✓ Encoded gender (Male=1, Female=0)")

# --- Feature 2: Encode country (assign numbers) ---
df['country_encoded'] = pd.factorize(df['country'])[0]
print("✓ Encoded country (USA=0, Canada=1, etc.)")

# --- Feature 3: Encode subscription (Basic=0, Standard=1, Premium=2) ---
df['subscription_encoded'] = df['monthly_subscription'].map({
    'Basic': 0,
    'Standard': 1,
    'Premium': 2
})
print("✓ Encoded subscription plan")

# --- Feature 4: Calculate engagement ratio ---
# How active are they? (logins per day as customer)
df['login_engagement_ratio'] = df['login_count_last_30'] / (df['account_age_days'] + 1)
print("✓ Calculated login engagement ratio")

# --- Feature 5: Calculate content engagement ---
# How much content are they viewing?
df['content_engagement_ratio'] = df['content_views_last_30'] / (df['account_age_days'] + 1)
print("✓ Calculated content engagement ratio")

# --- Feature 6: Feature adoption ---
# Using 1-10 features, what % are they using?
df['feature_adoption_rate'] = df['num_features_used'] / 10
print("✓ Calculated feature adoption rate")

# --- Feature 7: Payment reliability ---
# How many times they paid vs how many times they should
df['payment_reliability'] = df['payments_made'] / (df['total_months_subscribed'] + 1)
print("✓ Calculated payment reliability")

# --- Feature 8-10: Risk flags ---
# These are warning signs of churn
df['high_payment_risk'] = (df['failed_payment_count'] > 0).astype(int)
df['low_engagement'] = (df['content_views_last_30'] < 10).astype(int)
df['inactive_risk'] = (df['days_since_last_login'] > 30).astype(int)
print("✓ Created risk indicators")

# Fill missing satisfaction scores with average
df['support_satisfaction_score'].fillna(df['support_satisfaction_score'].median(), inplace=True)
print("✓ Filled missing support satisfaction scores\n")

print(f"Total features created: 28")

# ============================================================================
# STEP 4: PREPARE DATA FOR ML
# ============================================================================
print("🔧 STEP 4: Preparing data for ML...")
print("-"*80)

# List of features to use (all the columns except customer_id and target)
feature_list = [
    # Engagement features
    'account_age_days',
    'login_count_last_30',
    'login_count_last_90',
    'days_since_last_login',
    'content_views_last_30',
    'content_views_last_90',
    'support_tickets_count',
    'support_satisfaction_score',
    'num_features_used',
    'monthly_active_days',
    'avg_session_length_minutes',
    'profile_completion_percentage',
    # Financial features
    'total_months_subscribed',
    'failed_payment_count',
    'days_overdue',
    'total_revenue',
    'avg_monthly_revenue',
    'upgrade_count',
    'downgrade_count',
    # Demographic features
    'age',
    'subscription_encoded',
    # Engineered features
    'login_engagement_ratio',
    'content_engagement_ratio',
    'feature_adoption_rate',
    'payment_reliability',
    'high_payment_risk',
    'low_engagement',
    'inactive_risk',
]

# Extract features (X) and target (y)
X = df[feature_list].copy()  # Features (predictors)
y = df['churn_flag'].copy()  # Target (what we predict)

# Fill any remaining NaN values with mean
X.fillna(X.mean(), inplace=True)

print(f"✓ Selected {len(feature_list)} features")
print(f"✓ Feature matrix shape: {X.shape}")
print(f"✓ Target shape: {y.shape}\n")

# ============================================================================
# STEP 5: SPLIT DATA (TRAIN & TEST)
# ============================================================================
print("✂️  STEP 5: Splitting data for training and testing...")
print("-"*80)

# Split 80% for training, 20% for testing
# stratify=y ensures both sets have same churn rate
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Show breakdown
train_churn_rate = (y_train.sum() / len(y_train)) * 100
test_churn_rate = (y_test.sum() / len(y_test)) * 100

print(f"Training set: {len(X_train):,} customers ({train_churn_rate:.1f}% churned)")
print(f"Test set:     {len(X_test):,} customers ({test_churn_rate:.1f}% churned)\n")

# ============================================================================
# STEP 6: SCALE FEATURES
# ============================================================================
print("📏 STEP 6: Scaling features...")
print("-"*80)

# Standardize features (mean=0, standard deviation=1)
# This makes ML algorithms work better
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler for future use (when predicting on new data)
pickle.dump(scaler, open('scaler.pkl', 'wb'))
print("✓ Features standardized")
print("✓ Scaler saved (scaler.pkl)\n")

# ============================================================================
# STEP 7: TRAIN MODEL
# ============================================================================
print("🤖 STEP 7: Training Random Forest model...")
print("-"*80)

# Create model
# n_estimators = 200 trees
# max_depth = 15 (how deep each tree can go)
# class_weight='balanced' = give more weight to rare churned customers
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1,  # Use all CPU cores
    class_weight='balanced'
)

# Train the model
model.fit(X_train_scaled, y_train)
print("✓ Model trained successfully")

# Save model for future use
pickle.dump(model, open('churn_model.pkl', 'wb'))
print("✓ Model saved (churn_model.pkl)\n")

# ============================================================================
# STEP 8: EVALUATE MODEL
# ============================================================================
print("📊 STEP 8: Evaluating model...")
print("-"*80)

# Make predictions on training data
y_train_pred = model.predict(X_train_scaled)
y_train_prob = model.predict_proba(X_train_scaled)[:, 1]

# Make predictions on test data
y_test_pred = model.predict(X_test_scaled)
y_test_prob = model.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics for training set
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f} (92% = model got 92% correct)\n")

# Calculate metrics for test set
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_prob)

print(f"Test Set Metrics:")
print(f"   • Accuracy:  {test_accuracy:.4f} (correct predictions)")
print(f"   • Precision: {test_precision:.4f} (when predict churn, {test_precision*100:.0f}% correct)")
print(f"   • Recall:    {test_recall:.4f} (catches {test_recall*100:.0f}% of actual churners)")
print(f"   • F1-Score:  {test_f1:.4f} (balanced metric)")
print(f"   • AUC-ROC:   {test_auc:.4f} (discrimination ability)\n")

# Show confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_test_pred)
print(f"Confusion Matrix:")
print(f"   True Negatives:  {cm[0,0]} (correctly predicted no churn)")
print(f"   False Positives: {cm[0,1]} (incorrectly predicted churn)")
print(f"   False Negatives: {cm[1,0]} (missed churners)")
print(f"   True Positives:  {cm[1,1]} (correctly predicted churn)\n")

# ============================================================================
# STEP 9: FEATURE IMPORTANCE
# ============================================================================
print("🔍 STEP 9: Feature importance (what matters most)...")
print("-"*80)

# Get importance scores
importance = pd.DataFrame({
    'feature': feature_list,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Save feature importance
importance.to_csv('feature_importance.csv', index=False)

# Show top 15 features
print("Top 15 Most Important Features:")
for idx, (_, row) in enumerate(importance.head(15).iterrows(), 1):
    print(f"   {idx:2d}. {row['feature']:30s} {row['importance']:.4f}")
print()

# ============================================================================
# STEP 10: MAKE PREDICTIONS
# ============================================================================
print("🎯 STEP 10: Making predictions for all customers...")
print("-"*80)

# Scale all data
X_all_scaled = scaler.transform(X)

# Get predictions
churn_predictions = model.predict(X_all_scaled)
churn_probability = model.predict_proba(X_all_scaled)[:, 1]

# Create predictions dataframe
predictions_df = pd.DataFrame({
    'customer_id': df['customer_id'],
    'age': df['age'],
    'country': df['country'],
    'subscription': df['monthly_subscription'],
    'account_age_days': df['account_age_days'],
    'days_since_last_login': df['days_since_last_login'],
    'login_count_last_30': df['login_count_last_30'],
    'content_views_last_30': df['content_views_last_30'],
    'failed_payment_count': df['failed_payment_count'],
    'support_satisfaction_score': df['support_satisfaction_score'],
    'total_revenue': df['total_revenue'],
    'actual_churn': df['churn_flag'],  # Actual churn (for comparison)
    'predicted_churn': churn_predictions,  # What model predicts (0 or 1)
    'churn_probability': np.round(churn_probability, 4),  # Churn probability (0.0-1.0)
    'churn_risk_level': pd.cut(churn_probability, bins=[0, 0.3, 0.6, 1.0], labels=['Low', 'Medium', 'High'])
})

# Save predictions
predictions_df.to_csv('churn_predictions.csv', index=False)
print(f"✓ Created churn_predictions.csv ({len(predictions_df)} records)\n")

# ============================================================================
# STEP 11: SEGMENT CUSTOMERS
# ============================================================================
print("📈 STEP 11: Creating customer segments...")
print("-"*80)

# Segment by customer value (revenue)
predictions_df['customer_value'] = pd.cut(
    predictions_df['total_revenue'],
    bins=3,
    labels=['Low Value', 'Medium Value', 'High Value']
)

# Create retention priority
predictions_df['retention_priority'] = predictions_df.apply(
    lambda row: 'CRITICAL' if row['churn_risk_level'] == 'High' and row['customer_value'] == 'High Value'
    else 'HIGH' if row['churn_risk_level'] in ['High', 'Medium']
    else 'MEDIUM' if row['customer_value'] == 'High Value'
    else 'LOW',
    axis=1
)

# Show segments
print("Churn Risk Levels:")
print(predictions_df['churn_risk_level'].value_counts().sort_index())

print("\nRetention Priority Segments:")
print(predictions_df['retention_priority'].value_counts())
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("✅ MODEL TRAINING COMPLETE!")
print("="*80)

predicted_churners = (churn_predictions == 1).sum()
predicted_churn_rate = (predicted_churners / len(predictions_df)) * 100
avg_churn_prob = churn_probability.mean()

print(f"\n📊 Prediction Summary:")
print(f"   • Predicted Churners: {predicted_churners:,}")
print(f"   • Predicted Churn Rate: {predicted_churn_rate:.2f}%")
print(f"   • Average Churn Probability: {avg_churn_prob:.4f}")

# Show high-risk customers
high_risk = predictions_df[predictions_df['churn_risk_level'] == 'High'].nlargest(5, 'churn_probability')
print(f"\n🚨 Top 5 At-Risk Customers (to contact):")
for idx, row in high_risk.iterrows():
    print(f"   • Customer {row['customer_id']}: {row['churn_probability']:.4f} risk, ${row['total_revenue']:.2f} revenue")

print(f"\n📁 Files Created:")
print(f"   1. churn_model.pkl - Trained ML model")
print(f"   2. scaler.pkl - Feature scaler")
print(f"   3. feature_importance.csv - Feature rankings")
print(f"   4. churn_predictions.csv - Predictions for BI dashboard")

print(f"\n✨ Ready for Power BI/Tableau visualization!")
print("="*80 + "\n")
