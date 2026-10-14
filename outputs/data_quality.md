# Data quality notes

- Rows: 7,043, one per customer (customerID is unique: True)
- Missing values: none, apart from 11 blank TotalCharges, all on customers with 0 months of tenure. Set to 0, since they hadn't been billed yet.
- Churn rate: 26.5%
- Churn here means the customer left within the last month of the snapshot.
- Source: IBM sample data for a fictional telecom company, based on realistic usage patterns.
