import duckdb
try:
    conn = duckdb.connect()
    conn.execute("CREATE TABLE transactions AS SELECT * FROM read_csv_auto('data/HI-Small_Trans.csv')")
    
    print("\n--- 🚨 CONFIRMED MONEY LAUNDERERS (True Positives) ---")
    launderers = conn.execute("SELECT DISTINCT From_Account FROM transactions WHERE Is_Laundering = 1 LIMIT 5").fetchall()
    for acc in launderers:
        print(acc[0])

    print("\n--- ✅ NORMAL ACCOUNTS (For Comparison) ---")
    normal = conn.execute("SELECT DISTINCT From_Account FROM transactions WHERE Is_Laundering = 0 LIMIT 5").fetchall()
    for acc in normal:
        print(acc[0])
        
except Exception as e:
    print(e)
