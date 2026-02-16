# Test Database Overview

## Database Statistics

- **Users:** 8
- **Products:** 25 (5 per category)
- **Categories:** 5
- **Orders:** 27
- **Order Items:** 70

## Categories & Products

### Electronics (5 products)
- Laptop Pro 15 ($1,299.99) - 45 in stock
- Wireless Mouse ($29.99) - 150 in stock
- USB-C Hub ($49.99) - 80 in stock
- Bluetooth Headphones ($89.99) - 120 in stock
- 4K Monitor ($399.99) - 30 in stock

### Home & Kitchen (5 products)
- Coffee Maker ($79.99) - 60 in stock
- Blender ($59.99) - 75 in stock
- Air Fryer ($119.99) - 40 in stock
- Toaster Oven ($89.99) - 50 in stock
- Electric Kettle ($34.99) - 100 in stock

### Books (5 products)
- Python Programming Guide ($39.99) - 200 in stock
- Data Science Handbook ($49.99) - 150 in stock
- Machine Learning Basics ($44.99) - 180 in stock
- Web Development 101 ($34.99) - 220 in stock
- Database Design Patterns ($54.99) - 130 in stock

### Sports & Outdoors (5 products)
- Yoga Mat ($24.99) - 200 in stock
- Resistance Bands Set ($19.99) - 250 in stock
- Water Bottle ($14.99) - 300 in stock
- Running Shoes ($89.99) - 100 in stock
- Camping Tent ($149.99) - 40 in stock

### Clothing (5 products)
- Cotton T-Shirt ($19.99) - 500 in stock
- Denim Jeans ($49.99) - 300 in stock
- Hoodie ($39.99) - 250 in stock
- Winter Jacket ($129.99) - 80 in stock
- Running Shorts ($24.99) - 200 in stock

## Users

- Alice Johnson (alice@example.com)
- Bob Smith (bob@example.com)
- Carol Williams (carol@example.com)
- David Brown (david@example.com)
- Eve Davis (eve@example.com)
- Frank Miller (frank@example.com)
- Grace Wilson (grace@example.com)
- Henry Moore (henry@example.com)

## Sample Complex Queries to Test

### Basic Queries
1. "How many products are there?"
2. "List all users"
3. "Show all product categories"

### Filtering & Aggregation
4. "How many products are in each category?"
5. "List all Electronics products under $100"
6. "Which products are low in stock (less than 50)?"
7. "Show the most expensive product in each category"

### User-Specific Queries
8. "What did Alice purchase?"
9. "How many orders has Bob placed?"
10. "Show all orders by Carol"

### Date-Based Queries (SQLite syntax)
11. "Show orders from the last 30 days"
12. "What products were purchased in the last week?"
13. "List orders placed in the last 60 days"

### Multi-Table JOINs
14. "What's the total revenue from all orders?"
15. "Show product names and quantities for order 1"
16. "Which users bought Electronics products?"
17. "List all products purchased by users whose email contains 'example.com'"

### Advanced Aggregations
18. "What's the average order total?"
19. "Show total sales by category"
20. "Which category has the most products?"
21. "What's the total value of all inventory (price × stock)?"

### Complex Filters
22. "Show products with price between $20 and $100"
23. "List users who placed orders in the last 30 days"
24. "Find products in Books or Electronics categories under $50"

## Resetting the Database

To regenerate the test data:
```bash
cd /Users/madhukanukula/WORKSPACE/github-public/nlq-agent
.venv/bin/python utils/seed_db.py
```

This will:
- Clear all existing data
- Insert 8 users
- Insert 25 products across 5 categories
- Generate 27 orders with 70 order items
- Orders span the last 90 days with realistic pricing
