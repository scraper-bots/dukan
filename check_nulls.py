import csv
from collections import defaultdict

def check_null_fields(filename='umico_products.csv'):
    null_counts = defaultdict(int)
    total_rows = 0

    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames

        for row in reader:
            total_rows += 1
            for col in columns:
                value = row[col]
                if value == '' or value == 'None' or value == '[]':
                    null_counts[col] += 1

    print("=" * 80)
    print("NULL/EMPTY FIELD ANALYSIS")
    print("=" * 80)
    print(f"Total rows analyzed: {total_rows:,}\n")

    # Sort by null count descending
    sorted_nulls = sorted(null_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"{'Field Name':<35} {'Null Count':>12} {'Percentage':>12}")
    print("-" * 80)

    for field, count in sorted_nulls:
        percentage = (count / total_rows) * 100
        print(f"{field:<35} {count:>12,} {percentage:>11.1f}%")

    print("\n" + "=" * 80)
    print("FIELDS WITH NO NULL VALUES:")
    print("=" * 80)

    fields_with_no_nulls = [col for col in columns if col not in null_counts]
    for field in fields_with_no_nulls:
        print(f"  ✓ {field}")

if __name__ == "__main__":
    check_null_fields()
