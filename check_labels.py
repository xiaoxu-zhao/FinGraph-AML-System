import duckdb
try:
    # Query distinct values from the 'Is Laundering' column
    result = duckdb.sql("SELECT DISTINCT \"Is Laundering\" FROM 'data/HI-Small_Trans.csv'").fetchall()
    print("Unique values in 'Is Laundering':", result)
except Exception as e:
    print(f"Error: {e}")
