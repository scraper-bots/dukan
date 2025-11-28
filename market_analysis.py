import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for professional-looking charts
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class FoodMarketAnalysis:
    def __init__(self, csv_file='umico_products.csv'):
        """Initialize the analysis with data loading and preprocessing"""
        self.df = pd.read_csv(csv_file)
        self.charts_dir = Path('charts')
        self.charts_dir.mkdir(exist_ok=True)

        # Convert numeric columns
        numeric_cols = ['retail_price', 'old_price', 'discount_amount', 'discount_percentage',
                       'rating_value', 'qty', 'seller_rating']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Clean data
        self.df['has_discount'] = self.df['has_discount'].astype(bool)

        print(f"✓ Loaded {len(self.df)} products")
        print(f"✓ Data date range: {self.df['category_name'].nunique()} unique categories")

    def generate_all_charts(self):
        """Generate all market analysis charts"""
        print("\n" + "="*60)
        print("GENERATING FOOD MARKET ANALYSIS CHARTS")
        print("="*60 + "\n")

        self.chart_1_price_distribution()
        self.chart_2_discount_analysis()
        self.chart_3_category_analysis()
        self.chart_4_brand_performance()
        self.chart_5_seller_analysis()
        self.chart_6_market_opportunities()
        self.chart_7_pricing_strategy()
        self.chart_8_product_availability()

        print("\n" + "="*60)
        print("✓ ALL CHARTS GENERATED SUCCESSFULLY")
        print("="*60)

    def chart_1_price_distribution(self):
        """Price Distribution Analysis"""
        print("→ Generating Chart 1: Price Distribution Analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Price Distribution & Market Positioning Analysis',
                     fontsize=18, fontweight='bold', y=0.995)

        # 1. Overall price distribution
        ax1 = axes[0, 0]
        prices = self.df['retail_price'].dropna()
        prices_filtered = prices[prices < prices.quantile(0.95)]  # Remove outliers for better viz
        ax1.hist(prices_filtered, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
        ax1.axvline(prices.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {prices.median():.2f} AZN')
        ax1.axvline(prices.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {prices.mean():.2f} AZN')
        ax1.set_xlabel('Price (AZN)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Number of Products', fontsize=11, fontweight='bold')
        ax1.set_title('Overall Price Distribution', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)

        # 2. Price range segmentation
        ax2 = axes[0, 1]
        price_ranges = pd.cut(prices, bins=[0, 5, 10, 20, 50, 100, prices.max()],
                              labels=['0-5', '5-10', '10-20', '20-50', '50-100', '100+'])
        range_counts = price_ranges.value_counts().sort_index()
        bars = ax2.bar(range_counts.index, range_counts.values, color='#2ecc71', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Price Range (AZN)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Number of Products', fontsize=11, fontweight='bold')
        ax2.set_title('Product Count by Price Range', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')

        # 3. Top 10 Categories by Average Price
        ax3 = axes[1, 0]
        cat_prices = self.df.groupby('category_name')['retail_price'].agg(['mean', 'count'])
        cat_prices = cat_prices[cat_prices['count'] >= 10].sort_values('mean', ascending=False).head(10)
        bars = ax3.barh(range(len(cat_prices)), cat_prices['mean'], color='#e74c3c', alpha=0.7, edgecolor='black')
        ax3.set_yticks(range(len(cat_prices)))
        ax3.set_yticklabels(cat_prices.index, fontsize=9)
        ax3.set_xlabel('Average Price (AZN)', fontsize=11, fontweight='bold')
        ax3.set_title('Top 10 Most Expensive Categories', fontsize=13, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.2f}', ha='left', va='center', fontweight='bold', fontsize=9)

        # 4. Price vs Rating scatter
        ax4 = axes[1, 1]
        df_rated = self.df[(self.df['rating_value'] > 0) & (self.df['retail_price'] > 0)]
        scatter = ax4.scatter(df_rated['retail_price'], df_rated['rating_value'],
                            alpha=0.5, c=df_rated['discount_percentage'],
                            cmap='RdYlGn', s=50, edgecolor='black', linewidth=0.5)
        ax4.set_xlabel('Retail Price (AZN)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Rating (0-5)', fontsize=11, fontweight='bold')
        ax4.set_title('Price vs Customer Rating', fontsize=13, fontweight='bold')
        ax4.set_xlim(0, df_rated['retail_price'].quantile(0.95))
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Discount %', fontsize=10, fontweight='bold')
        ax4.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.charts_dir / '01_price_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Chart 1 saved: 01_price_distribution.png")

    def chart_2_discount_analysis(self):
        """Discount Strategy Analysis"""
        print("→ Generating Chart 2: Discount Analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Discount Strategy & Pricing Opportunities',
                     fontsize=18, fontweight='bold', y=0.995)

        # 1. Discount distribution
        ax1 = axes[0, 0]
        discounted = self.df[self.df['has_discount'] == True]
        labels = [f'Discounted\n({len(discounted)} products)\n{len(discounted)/len(self.df)*100:.1f}%',
                  f'Regular Price\n({len(self.df)-len(discounted)} products)\n{(len(self.df)-len(discounted))/len(self.df)*100:.1f}%']
        colors = ['#e74c3c', '#95a5a6']
        explode = (0.1, 0)
        ax1.pie([len(discounted), len(self.df)-len(discounted)], labels=labels,
                autopct='', colors=colors, explode=explode, startangle=90,
                textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax1.set_title('Products with vs without Discounts', fontsize=13, fontweight='bold')

        # 2. Discount percentage distribution
        ax2 = axes[0, 1]
        disc_pct = discounted['discount_percentage'].dropna()
        disc_pct = disc_pct[disc_pct > 0]
        ax2.hist(disc_pct, bins=30, color='#e67e22', alpha=0.7, edgecolor='black')
        ax2.axvline(disc_pct.median(), color='red', linestyle='--', linewidth=2,
                   label=f'Median: {disc_pct.median():.1f}%')
        ax2.axvline(disc_pct.mean(), color='green', linestyle='--', linewidth=2,
                   label=f'Mean: {disc_pct.mean():.1f}%')
        ax2.set_xlabel('Discount Percentage (%)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Number of Products', fontsize=11, fontweight='bold')
        ax2.set_title('Discount Percentage Distribution', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)

        # 3. Top 10 categories by discount percentage
        ax3 = axes[1, 0]
        cat_disc = discounted.groupby('category_name').agg({
            'discount_percentage': 'mean',
            'product_id': 'count'
        }).rename(columns={'product_id': 'count'})
        cat_disc = cat_disc[cat_disc['count'] >= 5].sort_values('discount_percentage', ascending=False).head(10)
        bars = ax3.barh(range(len(cat_disc)), cat_disc['discount_percentage'],
                       color='#9b59b6', alpha=0.7, edgecolor='black')
        ax3.set_yticks(range(len(cat_disc)))
        ax3.set_yticklabels(cat_disc.index, fontsize=9)
        ax3.set_xlabel('Average Discount %', fontsize=11, fontweight='bold')
        ax3.set_title('Top 10 Categories by Average Discount', fontsize=13, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.1f}%', ha='left', va='center', fontweight='bold', fontsize=9)

        # 4. Savings potential
        ax4 = axes[1, 1]
        total_savings = discounted['discount_amount'].sum()
        avg_savings = discounted['discount_amount'].mean()
        max_savings = discounted['discount_amount'].max()

        metrics = ['Total Market\nSavings', 'Avg Savings\nPer Product', 'Max Single\nSaving']
        values = [total_savings, avg_savings, max_savings]
        colors_bars = ['#27ae60', '#f39c12', '#c0392b']

        bars = ax4.bar(metrics, values, color=colors_bars, alpha=0.7, edgecolor='black', width=0.6)
        ax4.set_ylabel('Amount (AZN)', fontsize=11, fontweight='bold')
        ax4.set_title('Savings Metrics Analysis', fontsize=13, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}\nAZN', ha='center', va='bottom',
                    fontweight='bold', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.charts_dir / '02_discount_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Chart 2 saved: 02_discount_analysis.png")

    def chart_3_category_analysis(self):
        """Category Market Share & Performance"""
        print("→ Generating Chart 3: Category Analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Category Analysis & Market Segmentation',
                     fontsize=18, fontweight='bold', y=0.995)

        # 1. Top 15 categories by product count
        ax1 = axes[0, 0]
        cat_counts = self.df['category_name'].value_counts().head(15)
        bars = ax1.barh(range(len(cat_counts)), cat_counts.values,
                       color='#3498db', alpha=0.7, edgecolor='black')
        ax1.set_yticks(range(len(cat_counts)))
        ax1.set_yticklabels(cat_counts.index, fontsize=9)
        ax1.set_xlabel('Number of Products', fontsize=11, fontweight='bold')
        ax1.set_title('Top 15 Categories by Product Count', fontsize=13, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{int(width)}', ha='left', va='center', fontweight='bold', fontsize=9)

        # 2. Category market share pie (top 10 + others)
        ax2 = axes[0, 1]
        top_10_cats = cat_counts.head(10)
        others_count = cat_counts.iloc[10:].sum()

        pie_data = list(top_10_cats.values) + [others_count]
        pie_labels = list(top_10_cats.index) + ['Others']

        colors = plt.cm.Set3(range(len(pie_data)))
        ax2.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', colors=colors,
               textprops={'fontsize': 8, 'fontweight': 'bold'}, startangle=90)
        ax2.set_title('Category Market Share Distribution', fontsize=13, fontweight='bold')

        # 3. Category revenue potential (price * count)
        ax3 = axes[1, 0]
        cat_revenue = self.df.groupby('category_name').agg({
            'retail_price': 'sum',
            'product_id': 'count'
        }).rename(columns={'retail_price': 'total_value', 'product_id': 'count'})
        cat_revenue = cat_revenue[cat_revenue['count'] >= 10].sort_values('total_value', ascending=False).head(10)

        bars = ax3.barh(range(len(cat_revenue)), cat_revenue['total_value'],
                       color='#2ecc71', alpha=0.7, edgecolor='black')
        ax3.set_yticks(range(len(cat_revenue)))
        ax3.set_yticklabels(cat_revenue.index, fontsize=9)
        ax3.set_xlabel('Total Catalog Value (AZN)', fontsize=11, fontweight='bold')
        ax3.set_title('Top 10 Categories by Total Catalog Value', fontsize=13, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:,.0f}', ha='left', va='center', fontweight='bold', fontsize=8)

        # 4. Average rating by category
        ax4 = axes[1, 1]
        cat_ratings = self.df[self.df['rating_value'] > 0].groupby('category_name').agg({
            'rating_value': 'mean',
            'product_id': 'count'
        }).rename(columns={'product_id': 'count'})
        cat_ratings = cat_ratings[cat_ratings['count'] >= 20].sort_values('rating_value', ascending=False).head(10)

        bars = ax4.barh(range(len(cat_ratings)), cat_ratings['rating_value'],
                       color='#f39c12', alpha=0.7, edgecolor='black')
        ax4.set_yticks(range(len(cat_ratings)))
        ax4.set_yticklabels(cat_ratings.index, fontsize=9)
        ax4.set_xlabel('Average Rating', fontsize=11, fontweight='bold')
        ax4.set_title('Top 10 Highest-Rated Categories', fontsize=13, fontweight='bold')
        ax4.set_xlim(0, 5)
        ax4.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax4.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.2f}', ha='left', va='center', fontweight='bold', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.charts_dir / '03_category_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Chart 3 saved: 03_category_analysis.png")

    def chart_4_brand_performance(self):
        """Brand Market Performance"""
        print("→ Generating Chart 4: Brand Performance Analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Brand Performance & Market Positioning',
                     fontsize=18, fontweight='bold', y=0.995)

        # 1. Top 20 brands by product count
        ax1 = axes[0, 0]
        brand_counts = self.df[self.df['brand'] != 'N/A']['brand'].value_counts().head(20)
        bars = ax1.barh(range(len(brand_counts)), brand_counts.values,
                       color='#e74c3c', alpha=0.7, edgecolor='black')
        ax1.set_yticks(range(len(brand_counts)))
        ax1.set_yticklabels(brand_counts.index, fontsize=8)
        ax1.set_xlabel('Number of Products', fontsize=11, fontweight='bold')
        ax1.set_title('Top 20 Brands by Product Count', fontsize=13, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{int(width)}', ha='left', va='center', fontweight='bold', fontsize=7)

        # 2. Brand average price comparison
        ax2 = axes[0, 1]
        brand_prices = self.df[self.df['brand'] != 'N/A'].groupby('brand').agg({
            'retail_price': 'mean',
            'product_id': 'count'
        }).rename(columns={'product_id': 'count'})
        brand_prices = brand_prices[brand_prices['count'] >= 10].sort_values('retail_price', ascending=False).head(15)

        bars = ax2.barh(range(len(brand_prices)), brand_prices['retail_price'],
                       color='#9b59b6', alpha=0.7, edgecolor='black')
        ax2.set_yticks(range(len(brand_prices)))
        ax2.set_yticklabels(brand_prices.index, fontsize=8)
        ax2.set_xlabel('Average Price (AZN)', fontsize=11, fontweight='bold')
        ax2.set_title('Top 15 Premium Brands by Avg Price', fontsize=13, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.2f}', ha='left', va='center', fontweight='bold', fontsize=7)

        # 3. Brand discount strategy
        ax3 = axes[1, 0]
        brand_disc = self.df[(self.df['brand'] != 'N/A') & (self.df['has_discount'] == True)].groupby('brand').agg({
            'discount_percentage': 'mean',
            'product_id': 'count'
        }).rename(columns={'product_id': 'count'})
        brand_disc = brand_disc[brand_disc['count'] >= 5].sort_values('discount_percentage', ascending=False).head(15)

        bars = ax3.barh(range(len(brand_disc)), brand_disc['discount_percentage'],
                       color='#27ae60', alpha=0.7, edgecolor='black')
        ax3.set_yticks(range(len(brand_disc)))
        ax3.set_yticklabels(brand_disc.index, fontsize=8)
        ax3.set_xlabel('Average Discount %', fontsize=11, fontweight='bold')
        ax3.set_title('Top 15 Brands by Discount Percentage', fontsize=13, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.1f}%', ha='left', va='center', fontweight='bold', fontsize=7)

        # 4. Brand market concentration
        ax4 = axes[1, 1]
        total_products = len(self.df)
        top_brands = brand_counts.head(10)

        cumulative_share = []
        cumulative_pct = 0
        for count in top_brands.values:
            cumulative_pct += (count / total_products * 100)
            cumulative_share.append(cumulative_pct)

        x = range(1, len(top_brands) + 1)
        ax4.plot(x, cumulative_share, marker='o', linewidth=3, markersize=8,
                color='#e74c3c', label='Cumulative Market Share')
        ax4.fill_between(x, cumulative_share, alpha=0.3, color='#e74c3c')
        ax4.set_xlabel('Top N Brands', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Cumulative Market Share (%)', fontsize=11, fontweight='bold')
        ax4.set_title('Brand Market Concentration Curve', fontsize=13, fontweight='bold')
        ax4.grid(alpha=0.3)
        ax4.set_xticks(x)
        ax4.legend(fontsize=10)

        # Add percentage labels
        for i, (xi, yi) in enumerate(zip(x, cumulative_share)):
            ax4.text(xi, yi + 1, f'{yi:.1f}%', ha='center', fontsize=8, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.charts_dir / '04_brand_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Chart 4 saved: 04_brand_performance.png")

    def chart_5_seller_analysis(self):
        """Seller Performance Analysis"""
        print("→ Generating Chart 5: Seller Analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Seller Performance & Competition Analysis',
                     fontsize=18, fontweight='bold', y=0.995)

        # 1. Top sellers by product count
        ax1 = axes[0, 0]
        seller_counts = self.df[self.df['seller_name'] != 'N/A']['seller_name'].value_counts().head(15)
        bars = ax1.barh(range(len(seller_counts)), seller_counts.values,
                       color='#3498db', alpha=0.7, edgecolor='black')
        ax1.set_yticks(range(len(seller_counts)))
        ax1.set_yticklabels(seller_counts.index, fontsize=9)
        ax1.set_xlabel('Number of Products', fontsize=11, fontweight='bold')
        ax1.set_title('Top 15 Sellers by Product Count', fontsize=13, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{int(width)}', ha='left', va='center', fontweight='bold', fontsize=8)

        # 2. Seller ratings
        ax2 = axes[0, 1]
        seller_ratings = self.df[(self.df['seller_name'] != 'N/A') & (self.df['seller_rating'] > 0)].groupby('seller_name').agg({
            'seller_rating': 'mean',
            'product_id': 'count'
        }).rename(columns={'product_id': 'count'})
        seller_ratings = seller_ratings[seller_ratings['count'] >= 20].sort_values('seller_rating', ascending=False).head(15)

        bars = ax2.barh(range(len(seller_ratings)), seller_ratings['seller_rating'],
                       color='#f39c12', alpha=0.7, edgecolor='black')
        ax2.set_yticks(range(len(seller_ratings)))
        ax2.set_yticklabels(seller_ratings.index, fontsize=9)
        ax2.set_xlabel('Average Seller Rating', fontsize=11, fontweight='bold')
        ax2.set_title('Top 15 Sellers by Rating', fontsize=13, fontweight='bold')
        ax2.set_xlim(0, 5)
        ax2.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.2f}', ha='left', va='center', fontweight='bold', fontsize=8)

        # 3. Seller pricing strategy
        ax3 = axes[1, 0]
        seller_prices = self.df[self.df['seller_name'] != 'N/A'].groupby('seller_name').agg({
            'retail_price': 'mean',
            'product_id': 'count'
        }).rename(columns={'product_id': 'count'})
        seller_prices = seller_prices[seller_prices['count'] >= 20].sort_values('retail_price', ascending=False).head(15)

        bars = ax3.barh(range(len(seller_prices)), seller_prices['retail_price'],
                       color='#2ecc71', alpha=0.7, edgecolor='black')
        ax3.set_yticks(range(len(seller_prices)))
        ax3.set_yticklabels(seller_prices.index, fontsize=9)
        ax3.set_xlabel('Average Product Price (AZN)', fontsize=11, fontweight='bold')
        ax3.set_title('Top 15 Premium Sellers by Avg Price', fontsize=13, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.2f}', ha='left', va='center', fontweight='bold', fontsize=8)

        # 4. Market share pie chart
        ax4 = axes[1, 1]
        top_10_sellers = seller_counts.head(10)
        others_count = seller_counts.iloc[10:].sum()

        pie_data = list(top_10_sellers.values) + [others_count]
        pie_labels = list(top_10_sellers.index) + ['Others']

        colors = plt.cm.Paired(range(len(pie_data)))
        ax4.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', colors=colors,
               textprops={'fontsize': 7, 'fontweight': 'bold'}, startangle=90)
        ax4.set_title('Seller Market Share Distribution', fontsize=13, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.charts_dir / '05_seller_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Chart 5 saved: 05_seller_analysis.png")

    def chart_6_market_opportunities(self):
        """Market Gap & Opportunity Analysis"""
        print("→ Generating Chart 6: Market Opportunities...")

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        fig.suptitle('Market Opportunities & Growth Potential',
                     fontsize=18, fontweight='bold', y=0.995)

        # 1. High-discount opportunities
        ax1 = fig.add_subplot(gs[0, 0])
        high_disc = self.df[self.df['discount_percentage'] >= 30].groupby('category_name').size().sort_values(ascending=False).head(10)
        bars = ax1.barh(range(len(high_disc)), high_disc.values,
                       color='#e74c3c', alpha=0.7, edgecolor='black')
        ax1.set_yticks(range(len(high_disc)))
        ax1.set_yticklabels(high_disc.index, fontsize=9)
        ax1.set_xlabel('Number of Products', fontsize=11, fontweight='bold')
        ax1.set_title('Categories with High Discounts (≥30%)', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{int(width)}', ha='left', va='center', fontweight='bold', fontsize=8)

        # 2. Underserved categories (low product count but high demand indicators)
        ax2 = fig.add_subplot(gs[0, 1])
        cat_analysis = self.df.groupby('category_name').agg({
            'product_id': 'count',
            'rating_value': 'mean'
        }).rename(columns={'product_id': 'count', 'rating_value': 'avg_rating'})

        # Categories with few products but decent ratings (potential gaps)
        underserved = cat_analysis[(cat_analysis['count'] < 50) & (cat_analysis['avg_rating'] > 4.0)]
        underserved = underserved.sort_values('avg_rating', ascending=False).head(10)

        if len(underserved) > 0:
            bars = ax2.barh(range(len(underserved)), underserved['count'],
                           color='#3498db', alpha=0.7, edgecolor='black')
            ax2.set_yticks(range(len(underserved)))
            ax2.set_yticklabels(underserved.index, fontsize=9)
            ax2.set_xlabel('Product Count (Low Competition)', fontsize=11, fontweight='bold')
            ax2.set_title('Underserved High-Quality Categories', fontsize=12, fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)

            for i, bar in enumerate(bars):
                width = bar.get_width()
                rating = underserved.iloc[i]['avg_rating']
                ax2.text(width, bar.get_y() + bar.get_height()/2.,
                        f'{int(width)} (★{rating:.1f})', ha='left', va='center',
                        fontweight='bold', fontsize=8)

        # 3. Price gap opportunities
        ax3 = fig.add_subplot(gs[1, 0])
        # Categories with high price variance (opportunity for mid-range products)
        cat_price_var = self.df.groupby('category_name').agg({
            'retail_price': ['std', 'mean', 'count']
        })
        cat_price_var.columns = ['price_std', 'price_mean', 'count']
        cat_price_var = cat_price_var[cat_price_var['count'] >= 20]
        cat_price_var['coefficient_variation'] = (cat_price_var['price_std'] / cat_price_var['price_mean']) * 100
        cat_price_var = cat_price_var.sort_values('coefficient_variation', ascending=False).head(10)

        bars = ax3.barh(range(len(cat_price_var)), cat_price_var['coefficient_variation'],
                       color='#9b59b6', alpha=0.7, edgecolor='black')
        ax3.set_yticks(range(len(cat_price_var)))
        ax3.set_yticklabels(cat_price_var.index, fontsize=9)
        ax3.set_xlabel('Price Variation Coefficient (%)', fontsize=11, fontweight='bold')
        ax3.set_title('Categories with High Price Variation (Gap Opportunity)', fontsize=12, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.1f}%', ha='left', va='center', fontweight='bold', fontsize=8)

        # 4. Low-rated categories (improvement opportunity)
        ax4 = fig.add_subplot(gs[1, 1])
        low_rated = self.df[self.df['rating_value'] > 0].groupby('category_name').agg({
            'rating_value': 'mean',
            'product_id': 'count'
        }).rename(columns={'product_id': 'count'})
        low_rated = low_rated[low_rated['count'] >= 20].sort_values('rating_value').head(10)

        bars = ax4.barh(range(len(low_rated)), low_rated['rating_value'],
                       color='#e67e22', alpha=0.7, edgecolor='black')
        ax4.set_yticks(range(len(low_rated)))
        ax4.set_yticklabels(low_rated.index, fontsize=9)
        ax4.set_xlabel('Average Rating', fontsize=11, fontweight='bold')
        ax4.set_title('Lowest-Rated Categories (Quality Opportunity)', fontsize=12, fontweight='bold')
        ax4.set_xlim(0, 5)
        ax4.grid(axis='x', alpha=0.3)

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax4.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.2f}', ha='left', va='center', fontweight='bold', fontsize=8)

        # 5. Premium opportunity matrix (large text summary)
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')

        # Calculate key opportunity metrics
        total_discounted = len(self.df[self.df['has_discount'] == True])
        avg_discount = self.df[self.df['has_discount'] == True]['discount_percentage'].mean()
        total_categories = self.df['category_name'].nunique()
        total_brands = len(self.df[self.df['brand'] != 'N/A']['brand'].unique())

        opportunity_text = f"""
        KEY MARKET OPPORTUNITIES IDENTIFIED:

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        📊 MARKET OVERVIEW:
        • Total Products Analyzed: {len(self.df):,}
        • Active Categories: {total_categories}
        • Brands Present: {total_brands}
        • Products on Discount: {total_discounted:,} ({total_discounted/len(self.df)*100:.1f}%)

        🎯 TOP 3 OPPORTUNITIES:

        1. DISCOUNT OPTIMIZATION: {avg_discount:.1f}% average discount indicates aggressive pricing
           → Opportunity: Focus on categories with high discounts for competitive entry

        2. UNDERSERVED SEGMENTS: {len(underserved)} high-rated categories with <50 products
           → Opportunity: Low competition in quality-focused niches

        3. PRICE GAP MARKETS: High price variation in {len(cat_price_var)} categories
           → Opportunity: Introduce mid-range products to capture price-sensitive customers

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """

        ax5.text(0.5, 0.5, opportunity_text,
                ha='center', va='center', fontsize=10,
                family='monospace', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.savefig(self.charts_dir / '06_market_opportunities.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Chart 6 saved: 06_market_opportunities.png")

    def chart_7_pricing_strategy(self):
        """Competitive Pricing Analysis"""
        print("→ Generating Chart 7: Pricing Strategy Analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Competitive Pricing Strategy Analysis',
                     fontsize=18, fontweight='bold', y=0.995)

        # 1. Price elasticity by category (price vs count)
        ax1 = axes[0, 0]
        cat_price_count = self.df.groupby('category_name').agg({
            'retail_price': 'mean',
            'product_id': 'count'
        }).rename(columns={'product_id': 'count'})
        cat_price_count = cat_price_count[cat_price_count['count'] >= 10]

        scatter = ax1.scatter(cat_price_count['retail_price'], cat_price_count['count'],
                            s=100, alpha=0.6, c=cat_price_count['count'],
                            cmap='viridis', edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Average Price (AZN)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Number of Products', fontsize=11, fontweight='bold')
        ax1.set_title('Price vs Product Count by Category', fontsize=13, fontweight='bold')
        ax1.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Product Count')

        # 2. Discount depth by price range
        ax2 = axes[0, 1]
        discounted = self.df[self.df['has_discount'] == True].copy()
        discounted['price_range'] = pd.cut(discounted['retail_price'],
                                          bins=[0, 5, 10, 20, 50, 100, discounted['retail_price'].max()],
                                          labels=['0-5', '5-10', '10-20', '20-50', '50-100', '100+'])

        range_disc = discounted.groupby('price_range')['discount_percentage'].mean().sort_index()
        bars = ax2.bar(range(len(range_disc)), range_disc.values,
                      color='#e74c3c', alpha=0.7, edgecolor='black')
        ax2.set_xticks(range(len(range_disc)))
        ax2.set_xticklabels(range_disc.index, fontsize=10)
        ax2.set_xlabel('Price Range (AZN)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Average Discount %', fontsize=11, fontweight='bold')
        ax2.set_title('Discount Strategy by Price Range', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 3. Competitive price positioning
        ax3 = axes[1, 0]
        # Compare old_price vs retail_price distribution
        has_old_price = self.df[self.df['old_price'] > 0]

        ax3.scatter(has_old_price['old_price'], has_old_price['retail_price'],
                   alpha=0.3, s=30, c='#3498db', edgecolor='none')

        # Add diagonal line (no discount)
        max_price = max(has_old_price['old_price'].max(), has_old_price['retail_price'].max())
        ax3.plot([0, max_price], [0, max_price], 'r--', linewidth=2, label='No Discount Line')

        ax3.set_xlabel('Original Price (AZN)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Retail Price (AZN)', fontsize=11, fontweight='bold')
        ax3.set_title('Price Positioning: Original vs Retail', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(alpha=0.3)
        ax3.set_xlim(0, has_old_price['old_price'].quantile(0.95))
        ax3.set_ylim(0, has_old_price['retail_price'].quantile(0.95))

        # 4. Installment availability by price
        ax4 = axes[1, 1]
        installment_data = self.df.groupby('installment_enabled').agg({
            'retail_price': 'mean',
            'product_id': 'count'
        }).rename(columns={'product_id': 'count'})

        # Create labels dynamically based on actual data
        label_map = {False: 'No Installment', True: 'Installment Available'}
        labels = [label_map.get(idx, str(idx)) for idx in installment_data.index]
        sizes = installment_data['count'].values
        avg_prices = installment_data['retail_price'].values
        colors = ['#95a5a6', '#27ae60'] if len(sizes) == 2 else ['#95a5a6']
        explode = tuple([0.05] * len(sizes))

        wedges, texts, autotexts = ax4.pie(sizes, labels=labels, autopct='%1.1f%%',
                                           colors=colors[:len(sizes)], explode=explode, startangle=90,
                                           textprops={'fontsize': 11, 'fontweight': 'bold'})

        # Add average price info
        for i, (label, avg_price) in enumerate(zip(labels, avg_prices)):
            texts[i].set_text(f'{label}\n(Avg: {avg_price:.2f} AZN)')

        ax4.set_title('Installment Payment Availability', fontsize=13, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.charts_dir / '07_pricing_strategy.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Chart 7 saved: 07_pricing_strategy.png")

    def chart_8_product_availability(self):
        """Product Availability & Stock Analysis"""
        print("→ Generating Chart 8: Product Availability Analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Product Availability & Stock Management Analysis',
                     fontsize=18, fontweight='bold', y=0.995)

        # 1. Stock quantity distribution
        ax1 = axes[0, 0]
        stock_data = self.df[self.df['qty'] > 0]['qty']
        stock_filtered = stock_data[stock_data <= stock_data.quantile(0.95)]

        ax1.hist(stock_filtered, bins=40, color='#2ecc71', alpha=0.7, edgecolor='black')
        ax1.axvline(stock_data.median(), color='red', linestyle='--', linewidth=2,
                   label=f'Median: {stock_data.median():.0f} units')
        ax1.axvline(stock_data.mean(), color='blue', linestyle='--', linewidth=2,
                   label=f'Mean: {stock_data.mean():.0f} units')
        ax1.set_xlabel('Stock Quantity', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Number of Products', fontsize=11, fontweight='bold')
        ax1.set_title('Stock Quantity Distribution', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)

        # 2. Availability by category
        ax2 = axes[0, 1]
        cat_availability = self.df.groupby('category_name').agg({
            'qty': 'sum',
            'product_id': 'count'
        }).rename(columns={'product_id': 'count'})
        cat_availability = cat_availability[cat_availability['count'] >= 10].sort_values('qty', ascending=False).head(15)

        bars = ax2.barh(range(len(cat_availability)), cat_availability['qty'],
                       color='#3498db', alpha=0.7, edgecolor='black')
        ax2.set_yticks(range(len(cat_availability)))
        ax2.set_yticklabels(cat_availability.index, fontsize=9)
        ax2.set_xlabel('Total Stock Units', fontsize=11, fontweight='bold')
        ax2.set_title('Top 15 Categories by Total Stock', fontsize=13, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{int(width):,}', ha='left', va='center', fontweight='bold', fontsize=8)

        # 3. Product status distribution
        ax3 = axes[1, 0]
        status_counts = self.df['status'].value_counts()
        colors_status = plt.cm.Set3(range(len(status_counts)))

        wedges, texts, autotexts = ax3.pie(status_counts.values, labels=status_counts.index,
                                           autopct='%1.1f%%', colors=colors_status, startangle=90,
                                           textprops={'fontsize': 10, 'fontweight': 'bold'})

        # Add counts to labels
        for i, (label, count) in enumerate(zip(status_counts.index, status_counts.values)):
            texts[i].set_text(f'{label}\n({count:,} products)')

        ax3.set_title('Product Status Distribution', fontsize=13, fontweight='bold')

        # 4. Stock level categories
        ax4 = axes[1, 1]
        # Categorize stock levels
        def stock_category(qty):
            if qty == 0:
                return 'Out of Stock'
            elif qty <= 10:
                return 'Low Stock (1-10)'
            elif qty <= 50:
                return 'Medium Stock (11-50)'
            elif qty <= 100:
                return 'Good Stock (51-100)'
            else:
                return 'High Stock (100+)'

        self.df['stock_category'] = self.df['qty'].apply(stock_category)
        stock_cat_counts = self.df['stock_category'].value_counts()

        # Order categories
        order = ['Out of Stock', 'Low Stock (1-10)', 'Medium Stock (11-50)',
                'Good Stock (51-100)', 'High Stock (100+)']
        stock_cat_counts = stock_cat_counts.reindex([o for o in order if o in stock_cat_counts.index])

        colors_stock = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#27ae60']
        bars = ax4.bar(range(len(stock_cat_counts)), stock_cat_counts.values,
                      color=colors_stock[:len(stock_cat_counts)], alpha=0.7, edgecolor='black')

        ax4.set_xticks(range(len(stock_cat_counts)))
        ax4.set_xticklabels(stock_cat_counts.index, rotation=45, ha='right', fontsize=9)
        ax4.set_ylabel('Number of Products', fontsize=11, fontweight='bold')
        ax4.set_title('Stock Level Distribution', fontsize=13, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.charts_dir / '08_product_availability.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Chart 8 saved: 08_product_availability.png")

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print(" "*15 + "FOOD MARKET ANALYSIS SYSTEM")
    print("="*70)

    # Initialize analyzer
    analyzer = FoodMarketAnalysis('umico_products.csv')

    # Generate all charts
    analyzer.generate_all_charts()

    print("\n" + "="*70)
    print("✓ Analysis complete! All charts saved in /charts directory")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
