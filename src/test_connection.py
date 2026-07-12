import psycopg2

try:
    conn = psycopg2.connect(
        host="127.0.0.1",
        port=55432,
        database="fraud_warehouse",
        user="fraud_user",
        password="fraud_pass"
    )

    print("✅ Connected Successfully!")

    cur = conn.cursor()
    cur.execute("SELECT current_user, current_database();")
    print(cur.fetchone())

    cur.close()
    conn.close()

except Exception as e:
    print("❌ Connection Failed")
    print(e)