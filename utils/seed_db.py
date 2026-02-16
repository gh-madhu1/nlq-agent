#!/usr/bin/env python3
"""Seed the database with comprehensive test data."""

import sqlite3
from datetime import datetime, timedelta
import random

DB_PATH = "data/ecommerce.db"

# Categories and products
CATEGORIES = {
    "Electronics": [
        ("Laptop Pro 15", 1299.99, 45),
        ("Wireless Mouse", 29.99, 150),
        ("USB-C Hub", 49.99, 80),
        ("Bluetooth Headphones", 89.99, 120),
        ("4K Monitor", 399.99, 30),
    ],
    "Home & Kitchen": [
        ("Coffee Maker", 79.99, 60),
        ("Blender", 59.99, 75),
        ("Air Fryer", 119.99, 40),
        ("Toaster Oven", 89.99, 50),
        ("Electric Kettle", 34.99, 100),
    ],
    "Books": [
        ("Python Programming Guide", 39.99, 200),
        ("Data Science Handbook", 49.99, 150),
        ("Machine Learning Basics", 44.99, 180),
        ("Web Development 101", 34.99, 220),
        ("Database Design Patterns", 54.99, 130),
    ],
    "Sports & Outdoors": [
        ("Yoga Mat", 24.99, 200),
        ("Resistance Bands Set", 19.99, 250),
        ("Water Bottle", 14.99, 300),
        ("Running Shoes", 89.99, 100),
        ("Camping Tent", 149.99, 40),
    ],
    "Clothing": [
        ("Cotton T-Shirt", 19.99, 500),
        ("Denim Jeans", 49.99, 300),
        ("Hoodie", 39.99, 250),
        ("Winter Jacket", 129.99, 80),
        ("Running Shorts", 24.99, 200),
    ],
}

# User names and emails
USERS = [
    ("Alice Johnson", "alice@example.com"),
    ("Bob Smith", "bob@example.com"),
    ("Carol Williams", "carol@example.com"),
    ("David Brown", "david@example.com"),
    ("Eve Davis", "eve@example.com"),
    ("Frank Miller", "frank@example.com"),
    ("Grace Wilson", "grace@example.com"),
    ("Henry Moore", "henry@example.com"),
]

def seed_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Clear existing data
    cursor.execute("DELETE FROM order_items")
    cursor.execute("DELETE FROM orders")
    cursor.execute("DELETE FROM products")
    cursor.execute("DELETE FROM users")
    
    print("Seeding database with test data...")
    
    # Insert users
    print(f"\nInserting {len(USERS)} users...")
    user_ids = []
    for name, email in USERS:
        created_at = datetime.now() - timedelta(days=random.randint(30, 365))
        cursor.execute(
            "INSERT INTO users (name, email, created_at) VALUES (?, ?, ?)",
            (name, email, created_at.strftime("%Y-%m-%d"))
        )
        user_ids.append(cursor.lastrowid)
    
    # Insert products
    print(f"Inserting {sum(len(products) for products in CATEGORIES.values())} products across {len(CATEGORIES)} categories...")
    product_ids = []
    for category, products in CATEGORIES.items():
        for name, price, stock in products:
            cursor.execute(
                "INSERT INTO products (name, category, price, stock_quantity) VALUES (?, ?, ?, ?)",
                (name, category, price, stock)
            )
            product_ids.append(cursor.lastrowid)
    
    # Insert orders and order items
    print("Inserting orders and order items...")
    order_count = 0
    item_count = 0
    
    for user_id in user_ids:
        # Each user makes 2-5 orders
        num_orders = random.randint(2, 5)
        
        for _ in range(num_orders):
            # Order date within last 90 days
            order_date = datetime.now() - timedelta(days=random.randint(0, 90))
            
            # Create order
            cursor.execute(
                "INSERT INTO orders (user_id, order_date, total_amount) VALUES (?, ?, ?)",
                (user_id, order_date.strftime("%Y-%m-%d"), 0)  # Will update total
            )
            order_id = cursor.lastrowid
            order_count += 1
            
            # Add 1-4 items to order
            num_items = random.randint(1, 4)
            selected_products = random.sample(product_ids, num_items)
            total_amount = 0
            
            for product_id in selected_products:
                # Get product price
                cursor.execute("SELECT price FROM products WHERE product_id = ?", (product_id,))
                price = cursor.fetchone()[0]
                
                # Random quantity 1-3
                quantity = random.randint(1, 3)
                
                # Price might vary slightly from current price (historical pricing)
                price_at_purchase = price * random.uniform(0.9, 1.1)
                
                cursor.execute(
                    "INSERT INTO order_items (order_id, product_id, quantity, price_at_purchase) VALUES (?, ?, ?, ?)",
                    (order_id, product_id, quantity, price_at_purchase)
                )
                item_count += 1
                total_amount += price_at_purchase * quantity
            
            # Update order total
            cursor.execute(
                "UPDATE orders SET total_amount = ? WHERE order_id = ?",
                (total_amount, order_id)
            )
    
    conn.commit()
    
    # Print summary
    print("\n" + "="*60)
    print("DATABASE SEEDED SUCCESSFULLY!")
    print("="*60)
    
    cursor.execute("SELECT COUNT(*) FROM users")
    print(f"Users: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM products")
    print(f"Products: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(DISTINCT category) FROM products")
    print(f"Categories: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM orders")
    print(f"Orders: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM order_items")
    print(f"Order Items: {cursor.fetchone()[0]}")
    
    print("\nSample queries to try:")
    print("- How many products are in each category?")
    print("- List all Electronics products under $100")
    print("- What did Alice purchase?")
    print("- Show orders from the last 30 days")
    print("- Which products are low in stock (less than 50)?")
    print("- What's the total revenue from all orders?")
    print("- Show the most expensive products in each category")
    
    conn.close()

if __name__ == "__main__":
    seed_database()
