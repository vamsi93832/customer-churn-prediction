# =====================================================
# PROJECT 2: CUSTOMER CHURN DATA GENERATION
# Simple, readable code with detailed comments
# =====================================================

# === Import libraries ===
import pandas as pd  # For data tables
import numpy as np   # For numbers
from datetime import datetime, timedelta  # For dates
import random  # For random values

# Make random numbers same each time (for testing)
np.random.seed(42)
random.seed(42)

# Print welcome message
print("\n" + "="*60)
print("CREATING CUSTOMER CHURN DATASET")
print("="*60 + "\n")

# ============================================================================
# SECTION 1: CREATE CUSTOMER PROFILES
# ============================================================================
print("1️⃣  Creating 10,000 customer profiles...")

num_customers = 10000  # How many customers

# Customer information
customer_data = {
    # ID: 1001, 1002, 1003...
    'customer_id': range(1001, 1001 + num_customers),
    
    # Random age (18-80)
    'age': np.random.randint(18, 80, num_customers),
    
    # Random gender
    'gender': np.random.choice(['Male', 'Female'], num_customers),
    
    # Random country
    'country': np.random.choice(['USA', 'Canada', 'UK', 'India', 'Australia'], num_customers),
    
    # When they signed up (random date in 2022)
    'registration_date': [datetime(2022, 1, 1) + timedelta(days=random.randint(0, 730)) for _ in range(num_customers)],
    
    # Subscription type
    'monthly_subscription': np.random.choice(['Basic', 'Standard', 'Premium'], num_customers, p=[0.4, 0.4, 0.2]),
}

# Convert to table
customers = pd.DataFrame(customer_data)

# Convert dates to proper format
customers['registration_date'] = pd.to_datetime(customers['registration_date'])

# Add subscription price
customers['subscription_price'] = customers['monthly_subscription'].map({
    'Basic': 9.99,
    'Standard': 19.99,
    'Premium': 49.99
})

# Save to CSV
customers.to_csv('customers.csv', index=False)
print(f"   ✓ Created {len(customers)} customer profiles\n")

# ============================================================================
# SECTION 2: CREATE BEHAVIORAL DATA
# ============================================================================
print("2️⃣  Creating customer behavior data...")

behavior_data = []

# For each customer, create behavior record
for idx, customer in customers.iterrows():
    # Days since customer signed up
    days_active = (datetime.now() - customer['registration_date']).days
    
    # How many times logged in last 30 days
    login_30 = random.randint(0, 30) if days_active > 0 else 0
    
    # Create behavior entry
    behavior_data.append({
        'customer_id': customer['customer_id'],
        # How long customer has been with us
        'account_age_days': days_active,
        # Logins in last 30 days
        'login_count_last_30': login_30,
        # Logins in last 90 days
        'login_count_last_90': random.randint(0, 90) if days_active > 0 else 0,
        # Days since last login (important for churn!)
        'days_since_last_login': random.randint(0, min(days_active, 365)),
        # Content views last 30 days
        'content_views_last_30': random.randint(0, 500),
        # Content views last 90 days
        'content_views_last_90': random.randint(0, 1000),
        # Support tickets (if they contacted support)
        'support_tickets_count': random.randint(0, 20),
        # Support satisfaction (1-5 or null)
        'support_satisfaction_score': random.uniform(1, 5) if random.random() > 0.3 else None,
        # How many features they use
        'num_features_used': random.randint(1, 10),
        # Active days in month (0-30)
        'monthly_active_days': random.randint(0, 30),
        # Average session time (minutes)
        'avg_session_length_minutes': random.uniform(5, 120),
        # Profile completeness (%)
        'profile_completion_percentage': random.randint(30, 100),
    })

# Convert to table
behavior = pd.DataFrame(behavior_data)

# Save to CSV
behavior.to_csv('behavior.csv', index=False)
print(f"   ✓ Created {len(behavior)} behavior records\n")

# ============================================================================
# SECTION 3: CREATE FINANCIAL DATA
# ============================================================================
print("3️⃣  Creating financial history...")

financial_data = []

# For each customer, create financial record
for idx, customer in customers.iterrows():
    # How many months they've been customer
    months_active = max(1, (datetime.now() - customer['registration_date']).days // 30)
    
    # How many payments made (less than months = missed some)
    payments_made = random.randint(max(0, months_active - 2), months_active)
    
    # Failed payments (bad credit card, etc.)
    failed_payments = random.randint(0, 3)
    
    # Days overdue (if failed payments)
    days_overdue = 0 if failed_payments == 0 else random.randint(1, 90)
    
    # Total money spent
    total_spent = customer['subscription_price'] * payments_made
    
    # Create financial entry
    financial_data.append({
        'customer_id': customer['customer_id'],
        # How long subscribed (months)
        'total_months_subscribed': months_active,
        # Successful payments
        'payments_made': payments_made,
        # Failed payment attempts
        'failed_payment_count': failed_payments,
        # Days account is overdue
        'days_overdue': days_overdue,
        # Total revenue from this customer
        'total_revenue': round(total_spent, 2),
        # Average monthly revenue
        'avg_monthly_revenue': round(total_spent / months_active, 2) if months_active > 0 else 0,
        # Upgrades (switched to higher plan)
        'upgrade_count': random.randint(0, 5),
        # Downgrades (switched to lower plan - churn signal!)
        'downgrade_count': random.randint(0, 3),
        # When billing happens (1-28)
        'billing_cycle_day': random.randint(1, 28),
    })

# Convert to table
financial = pd.DataFrame(financial_data)

# Save to CSV
financial.to_csv('financial.csv', index=False)
print(f"   ✓ Created {len(financial)} financial records\n")

# ============================================================================
# SECTION 4: CREATE CHURN LABELS (TARGET VARIABLE)
# ============================================================================
print("4️⃣  Creating churn labels (target)...")

churn_data = []

# For each customer, decide if they churned
for idx, customer in customers.iterrows():
    # Get their behavior and financial data
    behavior_row = behavior.iloc[idx]
    financial_row = financial.iloc[idx]
    
    # Start with 0 churn points
    churn_score = 0
    
    # Add points based on risk factors
    # Factor 1: Days since last login (most important!)
    if behavior_row['days_since_last_login'] > 90:
        churn_score += 3
    elif behavior_row['days_since_last_login'] > 30:
        churn_score += 2
    
    # Factor 2: Low engagement (few content views)
    if behavior_row['content_views_last_30'] < 10:
        churn_score += 2
    
    # Factor 3: Payment problems
    if financial_row['failed_payment_count'] > 1:
        churn_score += 2
    
    # Factor 4: Unhappy with support
    if behavior_row['support_satisfaction_score'] and behavior_row['support_satisfaction_score'] < 2:
        churn_score += 2
    
    # Factor 5: Downgraded (sign they're unhappy)
    if financial_row['downgrade_count'] > 0:
        churn_score += 1
    
    # Convert score to churn probability
    if churn_score >= 5:
        churn_probability = 0.8  # 80% chance of churn
    elif churn_score >= 3:
        churn_probability = 0.5  # 50% chance
    elif churn_score >= 2:
        churn_probability = 0.3  # 30% chance
    else:
        churn_probability = 0.1  # 10% chance
    
    # Decide if customer churned (random based on probability)
    is_churned = random.random() < churn_probability
    
    # Create churn entry
    churn_data.append({
        'customer_id': customer['customer_id'],
        # 1 = churned, 0 = active
        'churn_flag': 1 if is_churned else 0,
        # Probability of churn (0.0 to 1.0)
        'churn_probability_score': round(churn_probability, 3),
        # When they'll likely churn
        'likely_churn_date': (datetime.now() + timedelta(days=random.randint(1, 30))).date() if is_churned else None,
    })

# Convert to table
churn = pd.DataFrame(churn_data)

# Save to CSV
churn.to_csv('churn.csv', index=False)

# Count how many churned
churn_count = churn['churn_flag'].sum()
churn_rate = (churn_count / len(churn)) * 100
print(f"   ✓ Created {len(churn)} churn labels")
print(f"   ✓ Churn rate: {churn_count:,} customers ({churn_rate:.1f}%)\n")

# ============================================================================
# SECTION 5: CREATE MASTER DATASET (ALL DATA MERGED)
# ============================================================================
print("5️⃣  Merging all data...")

# Combine all data
master = customers.merge(behavior, on='customer_id', how='left')
master = master.merge(financial, on='customer_id', how='left')
master = master.merge(churn, on='customer_id', how='left')

# Fill missing satisfaction scores with median
master['support_satisfaction_score'].fillna(master['support_satisfaction_score'].median(), inplace=True)

# Save merged dataset
master.to_csv('master_data.csv', index=False)
print(f"   ✓ Created master_data.csv ({len(master)} records)\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("="*60)
print("✅ SUCCESS! CHURN DATASET CREATED")
print("="*60)

print(f"\n👥 Customer Statistics:")
print(f"   • Total Customers: {len(customers):,}")
print(f"   • Age Range: {customers['age'].min()}-{customers['age'].max()} years")
print(f"   • Countries: {customers['country'].nunique()}")

print(f"\n📊 Behavioral Statistics:")
print(f"   • Average Account Age: {behavior['account_age_days'].mean():.0f} days")
print(f"   • Average Logins (30 days): {behavior['login_count_last_30'].mean():.1f}")
print(f"   • Average Session Length: {behavior['avg_session_length_minutes'].mean():.1f} minutes")

print(f"\n💰 Financial Statistics:")
print(f"   • Average Subscription Period: {financial['total_months_subscribed'].mean():.1f} months")
print(f"   • Average Monthly Revenue: ${financial['avg_monthly_revenue'].mean():.2f}")
print(f"   • Avg Failed Payments: {financial['failed_payment_count'].mean():.2f}")

print(f"\n📉 Churn Statistics:")
print(f"   • Churned Customers: {churn_count:,}")
print(f"   • Churn Rate: {churn_rate:.2f}%")
print(f"   • Active Customers: {len(churn) - churn_count:,}")

print(f"\n📁 Files Created:")
print(f"   1. customers.csv - Customer profiles")
print(f"   2. behavior.csv - Engagement metrics")
print(f"   3. financial.csv - Payment history")
print(f"   4. churn.csv - Churn labels")
print(f"   5. master_data.csv - All data merged")

print(f"\n✨ Ready for ML model training!")
print("="*60 + "\n")
