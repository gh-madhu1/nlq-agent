import sqlite3
import os

def init_sample_db(db_path='data/ecommerce.db'):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT,
        price REAL,
        stock_quantity INTEGER
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS orders (
        order_id INTEGER PRIMARY KEY,
        user_id INTEGER,
        order_date DATETIME DEFAULT CURRENT_TIMESTAMP,
        total_amount REAL,
        FOREIGN KEY (user_id) REFERENCES users (user_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS order_items (
        item_id INTEGER PRIMARY KEY,
        order_id INTEGER,
        product_id INTEGER,
        quantity INTEGER,
        price_at_purchase REAL,
        FOREIGN KEY (order_id) REFERENCES orders (order_id),
        FOREIGN KEY (product_id) REFERENCES products (product_id)
    )
    ''')

    # Insert sample data
    cursor.execute("INSERT OR IGNORE INTO users (name, email) VALUES ('Alice Smith', 'alice@example.com')")
    cursor.execute("INSERT OR IGNORE INTO users (name, email) VALUES ('Bob Jones', 'bob@example.com')")
    
    cursor.execute("INSERT OR IGNORE INTO products (name, category, price, stock_quantity) VALUES ('Laptop', 'Electronics', 1200.00, 10)")
    cursor.execute("INSERT OR IGNORE INTO products (name, category, price, stock_quantity) VALUES ('Mouse', 'Electronics', 25.00, 50)")
    cursor.execute("INSERT OR IGNORE INTO products (name, category, price, stock_quantity) VALUES ('Keyboard', 'Electronics', 75.00, 20)")
    
    cursor.execute("INSERT OR IGNORE INTO orders (user_id, total_amount) VALUES (1, 1225.00)")
    cursor.execute("INSERT OR IGNORE INTO order_items (order_id, product_id, quantity, price_at_purchase) VALUES (1, 1, 1, 1200.00)")
    cursor.execute("INSERT OR IGNORE INTO order_items (order_id, product_id, quantity, price_at_purchase) VALUES (1, 2, 1, 25.00)")

    conn.commit()
    conn.close()
    print(f"Sample database initialized at {db_path}")

if __name__ == "__main__":
    init_sample_db()
