import pandas as pd
import random
from datetime import datetime, timedelta

# Categories
categories = {
    "Electronics": ["Smartphone", "Laptop", "Headphones", "Smart Watch", "Tablet", "Camera", "Speaker"],
    "Clothing": ["T-Shirt", "Jeans", "Jacket", "Shoes", "Dress", "Sweater", "Hat"],
    "Home & Kitchen": ["Coffee Maker", "Blender", "Microwave", "Vacuum", "Lamp", "Chair", "Table"],
    "Books": ["Fiction", "Non-Fiction", "Textbook", "Comic", "Biography", "Self-Help"],
    "Sports": ["Football", "Basketball", "Tennis Racket", "Yoga Mat", "Dumbbell", "Treadmill"]
}

# Generate product data
products = []
for i in range(1, 1001):  # 1000 products
    category = random.choice(list(categories.keys()))
    subcategory = random.choice(categories[category])
    
    # Generate realistic price
    if category == "Electronics":
        price = random.randint(50, 1500)
    elif category == "Clothing":
        price = random.randint(15, 200)
    elif category == "Home & Kitchen":
        price = random.randint(20, 500)
    elif category == "Books":
        price = random.randint(10, 50)
    else:  # Sports
        price = random.randint(25, 300)
    
    # Generate description
    descriptions = {
        "Electronics": f"High-quality {subcategory} with latest features. Perfect for everyday use. Includes 1-year warranty.",
        "Clothing": f"Comfortable {subcategory} made from premium materials. Available in multiple sizes and colors.",
        "Home & Kitchen": f"Professional-grade {subcategory} for your home. Easy to clean and maintain.",
        "Books": f"Bestselling {subcategory} that everyone is talking about. Get your copy today!",
        "Sports": f"Professional {subcategory} for athletes of all levels. Durable and reliable."
    }
    
    product = {
        "id": i,
        "title": f"{subcategory} {random.choice(['Pro', 'Max', 'Lite', 'Plus', 'Elite'])}",
        "category": category,
        "subcategory": subcategory,
        "price": price,
        "discount": random.choice([0, 5, 10, 15, 20]),
        "rating": round(random.uniform(3.5, 5.0), 1),
        "reviews": random.randint(0, 500),
        "in_stock": random.choice([True, False]),
        "brand": random.choice(["TechPro", "ComfortWear", "HomeStyle", "ReadWell", "SportFit"]),
        "description": descriptions[category],
        "features": f"Feature 1, Feature 2, Feature 3",
        "tags": f"{category.lower()}, {subcategory.lower()}, sale",
        "created_at": datetime.now() - timedelta(days=random.randint(1, 365))
    }
    products.append(product)

# Create DataFrame
df = pd.DataFrame(products)

# Save to CSV
df.to_csv("data/raw/ecommerce_products.csv", index=False)
print(f"✅ Generated {len(df)} products")
print(f"📁 Saved to: data/raw/ecommerce_products.csv")
print(f"\n📊 Preview:")
print(df.head())
print(f"\n📈 Statistics:")
print(df['category'].value_counts())