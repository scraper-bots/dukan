import asyncio
import aiohttp
import json

async def test_scrape():
    """Test scraping a few pages to verify the scraper works"""
    base_url = "https://mp-catalog.umico.az/api/v1/products"
    headers = {
        'accept': 'application/json, text/plain, */*',
        'accept-language': 'az',
        'content-language': 'az',
        'origin': 'https://birmarket.az',
        'referer': 'https://birmarket.az/',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36'
    }

    async with aiohttp.ClientSession() as session:
        # Test page 1
        params = {
            'page': 1,
            'category_id': 4497,
            'per_page': 24,
            'sort': 'global_popular_score'
        }

        async with session.get(base_url, params=params, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                print(f"✓ Test successful!")
                print(f"  Status: {response.status}")
                print(f"  Products in page 1: {len(data.get('products', []))}")
                print(f"  Total products: {data.get('meta', {}).get('total', 0)}")
                print(f"  Calculated pages: {(data.get('meta', {}).get('total', 0) + 23) // 24}")

                # Show first product structure
                if data.get('products'):
                    print(f"\n✓ First product sample:")
                    first_product = data['products'][0]
                    print(f"  ID: {first_product.get('id')}")
                    print(f"  Name: {first_product.get('name')}")
                    print(f"  Price: {first_product.get('default_offer', {}).get('retail_price')}")
                    print(f"  Brand: {first_product.get('brand')}")

                return True
            else:
                print(f"✗ Error: Status {response.status}")
                return False

if __name__ == "__main__":
    print("Testing UMICO API connection...\n")
    asyncio.run(test_scrape())
