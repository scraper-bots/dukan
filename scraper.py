import asyncio
import aiohttp
import csv
import json
from typing import List, Dict
from datetime import datetime

class UmicoScraper:
    def __init__(self):
        self.base_url = "https://mp-catalog.umico.az/api/v1/products"
        self.category_id = 4497
        self.per_page = 24
        self.sort = "global_popular_score"
        self.headers = {
            'accept': 'application/json, text/plain, */*',
            'accept-language': 'az',
            'content-language': 'az',
            'origin': 'https://birmarket.az',
            'referer': 'https://birmarket.az/',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36'
        }
        self.all_products = []

    async def fetch_page(self, session: aiohttp.ClientSession, page: int, retries: int = 3) -> Dict:
        """Fetch a single page of products with retry logic"""
        params = {
            'page': page,
            'category_id': self.category_id,
            'per_page': self.per_page,
            'sort': self.sort
        }

        for attempt in range(retries):
            try:
                async with session.get(self.base_url, params=params, headers=self.headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✓ Page {page} fetched successfully ({len(data.get('products', []))} products)")
                        return data
                    else:
                        print(f"✗ Error fetching page {page}: Status {response.status}")
                        if attempt < retries - 1:
                            await asyncio.sleep(2)
                            continue
                        return None
            except asyncio.TimeoutError:
                print(f"✗ Timeout fetching page {page} (attempt {attempt + 1}/{retries})")
                if attempt < retries - 1:
                    await asyncio.sleep(2)
                    continue
                return None
            except Exception as e:
                print(f"✗ Exception fetching page {page}: {type(e).__name__}: {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(2)
                    continue
                return None
        return None

    async def get_total_pages(self, session: aiohttp.ClientSession) -> int:
        """Get the total number of pages to scrape"""
        first_page = await self.fetch_page(session, 1)
        if first_page and 'meta' in first_page:
            total_products = first_page['meta'].get('total', 0)
            total_pages = (total_products + self.per_page - 1) // self.per_page
            print(f"\nTotal products: {total_products}")
            print(f"Total pages to scrape: {total_pages}\n")
            return total_pages, first_page
        return 0, None

    async def scrape_all_pages(self, max_concurrent: int = 10):
        """Scrape all pages with rate limiting"""
        async with aiohttp.ClientSession() as session:
            # Get total pages and first page data
            total_pages, first_page_data = await self.get_total_pages(session)

            if total_pages == 0:
                print("Failed to get total pages")
                return

            # Add first page products
            if first_page_data and 'products' in first_page_data:
                self.all_products.extend(first_page_data['products'])

            # Create tasks for remaining pages with semaphore for rate limiting
            semaphore = asyncio.Semaphore(max_concurrent)

            async def fetch_with_semaphore(page):
                async with semaphore:
                    data = await self.fetch_page(session, page)
                    # Small delay to be respectful to the server
                    await asyncio.sleep(0.1)
                    return data

            # Fetch pages 2 through total_pages
            tasks = [fetch_with_semaphore(page) for page in range(2, total_pages + 1)]
            results = await asyncio.gather(*tasks)

            # Collect all products
            for result in results:
                if result and 'products' in result:
                    self.all_products.extend(result['products'])

            print(f"\n✓ Total products scraped: {len(self.all_products)}")

    def safe_get(self, obj, *keys, default=''):
        """Safely get nested dictionary values with default"""
        for key in keys:
            if isinstance(obj, dict):
                obj = obj.get(key)
                if obj is None:
                    return default
            else:
                return default
        return obj if obj != '' else default

    def _format_labels(self, labels):
        """Format labels handling both strings and dicts"""
        if not labels:
            return 'N/A'
        formatted = []
        for label in labels:
            if isinstance(label, dict):
                # If label is a dict, try to get a name or id field
                formatted.append(str(label.get('name', label.get('id', label))))
            else:
                formatted.append(str(label))
        return ', '.join(formatted) if formatted else 'N/A'

    def flatten_product(self, product: Dict) -> Dict:
        """Flatten nested product structure for CSV with comprehensive field extraction"""

        # Get nested objects
        default_offer = product.get('default_offer', {})
        seller = default_offer.get('seller', {})
        marketing_name = seller.get('marketing_name', {})
        logo = seller.get('logo', {})
        category = product.get('category', {})
        main_img = product.get('main_img', {})
        ratings = product.get('ratings', {})

        flat = {
            # Basic product info
            'product_id': self.safe_get(product, 'id', default=None),
            'name': self.safe_get(product, 'name', default='N/A'),
            'slugged_name': self.safe_get(product, 'slugged_name', default='N/A'),
            'status': self.safe_get(product, 'status', default='unknown'),
            'avail_check': self.safe_get(product, 'avail_check', default=False),
            'min_qty': self.safe_get(product, 'min_qty', default=1),
            'preorder_available': self.safe_get(product, 'preorder_available', default=False),
            'brand': self.safe_get(product, 'brand', default='N/A'),

            # Category info - both from root and nested
            'category_id': self.safe_get(product, 'category_id', default=None),
            'category_name': self.safe_get(category, 'name', default='N/A'),
            'category_nested_id': self.safe_get(category, 'id', default=None),

            # Main images
            'img_big': self.safe_get(main_img, 'big', default='N/A'),
            'img_medium': self.safe_get(main_img, 'medium', default='N/A'),
            'img_small': self.safe_get(main_img, 'small', default='N/A'),

            # Ratings
            'rating_value': self.safe_get(ratings, 'rating_value', default=0),
            'rating_session_count': self.safe_get(ratings, 'session_count', default=0),
            'rating_assessment_id': self.safe_get(ratings, 'assessment_id', default=0),

            # Offer info
            'offer_uuid': self.safe_get(default_offer, 'uuid', default='N/A'),
            'installment_enabled': self.safe_get(default_offer, 'installment_enabled', default=False),
            'max_installment_months': self.safe_get(default_offer, 'max_installment_months', default=0),
            'old_price': self.safe_get(default_offer, 'old_price', default=0),
            'retail_price': self.safe_get(default_offer, 'retail_price', default=0),
            'offer_avail_check': self.safe_get(default_offer, 'avail_check', default=False),
            'stock_qty_threshold': self.safe_get(default_offer, 'show_stock_qty_threshold', default=0),
            'discount_start_date': self.safe_get(default_offer, 'discount_effective_start_date', default='N/A'),
            'discount_end_date': self.safe_get(default_offer, 'discount_effective_end_date', default='N/A'),
            'qty': self.safe_get(default_offer, 'qty', default=0),

            # Calculate discount percentage and savings
            'has_discount': self.safe_get(default_offer, 'old_price', default=0) > self.safe_get(default_offer, 'retail_price', default=0),
            'discount_amount': round(self.safe_get(default_offer, 'old_price', default=0) - self.safe_get(default_offer, 'retail_price', default=0), 2),
            'discount_percentage': round(((self.safe_get(default_offer, 'old_price', default=0) - self.safe_get(default_offer, 'retail_price', default=0)) / self.safe_get(default_offer, 'old_price', default=1)) * 100, 2) if self.safe_get(default_offer, 'old_price', default=0) > 0 else 0,

            # Seller info
            'seller_ext_id': self.safe_get(seller, 'ext_id', default='N/A'),
            'seller_name': self.safe_get(marketing_name, 'name', default='N/A'),
            'seller_marketing_name_id': self.safe_get(marketing_name, 'id', default=None),
            'seller_logo': self.safe_get(logo, 'thumbnail', default='N/A'),
            'seller_vat_payer': self.safe_get(seller, 'vat_payer', default=False),
            'seller_rating': self.safe_get(seller, 'rating', default=0),
            'seller_role': self.safe_get(seller, 'role_name', default='N/A'),

            # Labels (handle both strings and dicts)
            'product_labels_count': len(product.get('product_labels', [])),
            'product_labels': self._format_labels(product.get('product_labels', [])),
            'offer_labels_count': len(default_offer.get('product_offer_labels', [])),
            'offer_labels': self._format_labels(default_offer.get('product_offer_labels', [])),
        }

        return flat

    def save_to_csv(self, filename: str = 'umico_products.csv'):
        """Save all products to CSV file"""
        if not self.all_products:
            print("No products to save")
            return

        # Flatten all products
        flattened_products = [self.flatten_product(p) for p in self.all_products]

        # Get all unique keys from all products
        all_keys = set()
        for product in flattened_products:
            all_keys.update(product.keys())

        fieldnames = sorted(list(all_keys))

        # Write to CSV
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flattened_products)

        print(f"\n✓ Data saved to {filename}")
        print(f"  Total rows: {len(flattened_products)}")
        print(f"  Total columns: {len(fieldnames)}")

        # Verify data integrity
        self.verify_data_integrity(filename)

    def verify_data_integrity(self, filename: str):
        """Verify that all data was saved correctly"""
        with open(filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            row_count = sum(1 for row in reader)

        print(f"\n✓ Data Integrity Check:")
        print(f"  Products scraped: {len(self.all_products)}")
        print(f"  Rows in CSV: {row_count}")

        if row_count == len(self.all_products):
            print(f"  ✓ All data saved successfully!")
        else:
            print(f"  ✗ Warning: Row count mismatch!")

async def main():
    scraper = UmicoScraper()

    print("=" * 60)
    print("UMICO Product Scraper")
    print("=" * 60)
    print(f"Category ID: {scraper.category_id}")
    print(f"Starting scrape at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")

    # Scrape all pages
    await scraper.scrape_all_pages(max_concurrent=10)

    # Save to CSV
    scraper.save_to_csv('umico_products.csv')

    print("\n" + "=" * 60)
    print(f"Scraping completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
