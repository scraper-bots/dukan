import csv
import json

def analyze_csv(filename='umico_products.csv'):
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        # Get column names
        columns = reader.fieldnames

        # Read first product
        first_product = next(reader)

        print("=" * 80)
        print("CSV DATA ANALYSIS")
        print("=" * 80)
        print(f"\nTotal Columns: {len(columns)}")
        print("\nColumn Names:")
        for i, col in enumerate(columns, 1):
            print(f"  {i:2d}. {col}")

        print("\n" + "=" * 80)
        print("SAMPLE PRODUCT (First Row)")
        print("=" * 80)

        for key, value in first_product.items():
            if len(str(value)) > 60:
                value = str(value)[:57] + "..."
            print(f"{key:25s}: {value}")

        print("\n" + "=" * 80)
        print("FILE STATISTICS")
        print("=" * 80)

    # Count total rows
    with open(filename, 'r', encoding='utf-8') as f:
        total_rows = sum(1 for line in f) - 1  # Subtract header

    import os
    file_size = os.path.getsize(filename)

    print(f"Total Products: {total_rows:,}")
    print(f"File Size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
    print(f"Average bytes per product: {file_size / total_rows:.0f}")

if __name__ == "__main__":
    analyze_csv()
