import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for professional charts
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

class FoodMarketAnalysis:
    def __init__(self, csv_file='umico_products.csv'):
        """Initialize with data loading and preprocessing"""
        self.df = pd.read_csv(csv_file)
        self.charts_dir = Path('charts')
        self.charts_dir.mkdir(exist_ok=True)

        # Convert numeric columns
        numeric_cols = ['retail_price', 'old_price', 'discount_amount', 'discount_percentage',
                       'rating_value', 'rating_session_count']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Convert boolean
        if 'has_discount' in self.df.columns:
            self.df['has_discount'] = self.df['has_discount'].astype(bool)

        print(f"✓ Loaded {len(self.df):,} products")
        print(f"✓ Categories: {self.df['category_name'].nunique()}")
        print(f"✓ Brands: {len(self.df[self.df['brand'] != 'N/A']['brand'].unique())}")

    def generate_all_charts(self):
        """Generate all analysis charts"""
        print("\n" + "="*70)
        print("GENERATING FOOD MARKET ANALYSIS CHARTS")
        print("="*70 + "\n")

        self.chart_1_price_distribution()
        self.chart_2_discount_analysis()
        self.chart_3_category_analysis()
        self.chart_4_brand_performance()
        self.chart_5_product_labels()
        self.chart_6_market_opportunities()
        self.chart_7_pricing_strategy()
        self.chart_8_product_assortment()

        print("\n" + "="*70)
        print("✓ ALL CHARTS GENERATED SUCCESSFULLY")
        print("="*70)

    def chart_1_price_distribution(self):
        """Price Distribution Analysis"""
        print("→ Chart 1: Price Distribution Analysis...")

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
        fig.suptitle('Price Distribution & Market Positioning Analysis',
                     fontsize=20, fontweight='bold', y=0.98)

        # 1. Overall price distribution histogram
        ax1 = fig.add_subplot(gs[0, :2])
        prices = self.df['retail_price'].dropna()
        prices_filtered = prices[prices < prices.quantile(0.95)]

        ax1.hist(prices_filtered, bins=60, color='#3498db', alpha=0.75, edgecolor='black', linewidth=0.8)
        ax1.axvline(prices.median(), color='#e74c3c', linestyle='--', linewidth=2.5,
                   label=f'Median: {prices.median():.2f} AZN')
        ax1.axvline(prices.mean(), color='#27ae60', linestyle='--', linewidth=2.5,
                   label=f'Mean: {prices.mean():.2f} AZN')
        ax1.set_xlabel('Price (AZN)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Products', fontsize=12, fontweight='bold')
        ax1.set_title('Overall Price Distribution', fontsize=14, fontweight='bold', pad=15)
        ax1.legend(fontsize=11, framealpha=0.9)
        ax1.grid(alpha=0.4, linestyle=':')

        # 2. Price range segmentation
        ax2 = fig.add_subplot(gs[0, 2])
        price_ranges = pd.cut(prices, bins=[0, 5, 10, 20, 50, 100, prices.max()],
                              labels=['0-5\nAZN', '5-10\nAZN', '10-20\nAZN', '20-50\nAZN', '50-100\nAZN', '100+\nAZN'])
        range_counts = price_ranges.value_counts().sort_index()

        colors_grad = plt.cm.viridis(np.linspace(0.3, 0.9, len(range_counts)))
        bars = ax2.bar(range(len(range_counts)), range_counts.values,
                      color=colors_grad, edgecolor='black', linewidth=1.2, width=0.7)
        ax2.set_xticks(range(len(range_counts)))
        ax2.set_xticklabels(range_counts.index, fontsize=9, fontweight='bold')
        ax2.set_ylabel('Product Count', fontsize=11, fontweight='bold')
        ax2.set_title('Products by Price Range', fontsize=12, fontweight='bold', pad=12)
        ax2.grid(axis='y', alpha=0.4, linestyle=':')

        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'{int(height):,}', ha='center', va='bottom',
                    fontweight='bold', fontsize=9)

        # 3. Top 12 expensive categories
        ax3 = fig.add_subplot(gs[1, :2])
        cat_prices = self.df.groupby('category_name')['retail_price'].agg(['mean', 'count'])
        cat_prices = cat_prices[cat_prices['count'] >= 10].sort_values('mean', ascending=True).tail(12)

        bars = ax3.barh(range(len(cat_prices)), cat_prices['mean'],
                       color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.8)
        ax3.set_yticks(range(len(cat_prices)))
        ax3.set_yticklabels(cat_prices.index, fontsize=10)
        ax3.set_xlabel('Average Price (AZN)', fontsize=12, fontweight='bold')
        ax3.set_title('Top 12 Most Expensive Categories (min 10 products)',
                     fontsize=13, fontweight='bold', pad=12)
        ax3.grid(axis='x', alpha=0.4, linestyle=':')

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width + 0.3, bar.get_y() + bar.get_height()/2.,
                    f'{width:.2f} AZN', ha='left', va='center',
                    fontweight='bold', fontsize=9)

        # 4. Price vs Rating scatter
        ax4 = fig.add_subplot(gs[1, 2])
        df_rated = self.df[(self.df['rating_value'] > 0) & (self.df['retail_price'] > 0)]

        scatter = ax4.scatter(df_rated['retail_price'], df_rated['rating_value'],
                            alpha=0.6, c=df_rated['discount_percentage'],
                            cmap='RdYlGn_r', s=60, edgecolor='black', linewidth=0.5)
        ax4.set_xlabel('Price (AZN)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Rating (0-5)', fontsize=11, fontweight='bold')
        ax4.set_title('Price vs Rating', fontsize=12, fontweight='bold', pad=12)
        ax4.set_xlim(0, df_rated['retail_price'].quantile(0.95))
        ax4.set_ylim(0, 5.2)
        ax4.grid(alpha=0.3, linestyle=':')

        cbar = plt.colorbar(scatter, ax=ax4, pad=0.02)
        cbar.set_label('Discount %', fontsize=10, fontweight='bold')

        plt.savefig(self.charts_dir / '01_price_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 01_price_distribution.png")

    def chart_2_discount_analysis(self):
        """Discount Strategy Analysis"""
        print("→ Chart 2: Discount Analysis...")

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
        fig.suptitle('Discount Strategy & Promotional Pricing',
                     fontsize=20, fontweight='bold', y=0.98)

        discounted = self.df[self.df['has_discount'] == True]

        # 1. Pie chart
        ax1 = fig.add_subplot(gs[0, 0])
        sizes = [len(discounted), len(self.df)-len(discounted)]
        labels = [f'Discounted\n{len(discounted):,} products\n({len(discounted)/len(self.df)*100:.1f}%)',
                  f'Regular Price\n{len(self.df)-len(discounted):,} products\n({(len(self.df)-len(discounted))/len(self.df)*100:.1f}%)']
        colors = ['#e74c3c', '#95a5a6']
        explode = (0.08, 0)

        wedges, texts = ax1.pie(sizes, labels=labels, colors=colors, explode=explode,
                               startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
        ax1.set_title('Discount vs Regular Price', fontsize=13, fontweight='bold', pad=15)

        # 2. Discount percentage distribution
        ax2 = fig.add_subplot(gs[0, 1:])
        disc_pct = discounted['discount_percentage'].dropna()
        disc_pct = disc_pct[disc_pct > 0]

        ax2.hist(disc_pct, bins=40, color='#e67e22', alpha=0.75, edgecolor='black', linewidth=0.8)
        ax2.axvline(disc_pct.median(), color='#e74c3c', linestyle='--', linewidth=2.5,
                   label=f'Median: {disc_pct.median():.1f}%')
        ax2.axvline(disc_pct.mean(), color='#27ae60', linestyle='--', linewidth=2.5,
                   label=f'Mean: {disc_pct.mean():.1f}%')
        ax2.set_xlabel('Discount Percentage (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Products', fontsize=12, fontweight='bold')
        ax2.set_title('Discount Percentage Distribution', fontsize=14, fontweight='bold', pad=15)
        ax2.legend(fontsize=11, framealpha=0.9)
        ax2.grid(alpha=0.4, linestyle=':')

        # 3. Top categories by discount
        ax3 = fig.add_subplot(gs[1, :2])
        cat_disc = discounted.groupby('category_name').agg({
            'discount_percentage': 'mean',
            'product_id': 'count'
        }).rename(columns={'product_id': 'count'})
        cat_disc = cat_disc[cat_disc['count'] >= 5].sort_values('discount_percentage', ascending=True).tail(12)

        bars = ax3.barh(range(len(cat_disc)), cat_disc['discount_percentage'],
                       color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=0.8)
        ax3.set_yticks(range(len(cat_disc)))
        ax3.set_yticklabels(cat_disc.index, fontsize=10)
        ax3.set_xlabel('Average Discount %', fontsize=12, fontweight='bold')
        ax3.set_title('Top 12 Categories by Average Discount (min 5 products)',
                     fontsize=13, fontweight='bold', pad=12)
        ax3.grid(axis='x', alpha=0.4, linestyle=':')

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                    f'{width:.1f}%', ha='left', va='center',
                    fontweight='bold', fontsize=9)

        # 4. Savings metrics
        ax4 = fig.add_subplot(gs[1, 2])
        total_savings = discounted['discount_amount'].sum()
        avg_savings = discounted['discount_amount'].mean()
        max_savings = discounted['discount_amount'].max()

        metrics = ['Total\nSavings', 'Avg per\nProduct', 'Max\nSaving']
        values = [total_savings, avg_savings, max_savings]
        colors_bars = ['#27ae60', '#f39c12', '#c0392b']

        bars = ax4.bar(metrics, values, color=colors_bars, alpha=0.8,
                      edgecolor='black', linewidth=1.2, width=0.6)
        ax4.set_ylabel('Amount (AZN)', fontsize=11, fontweight='bold')
        ax4.set_title('Savings Metrics', fontsize=12, fontweight='bold', pad=12)
        ax4.grid(axis='y', alpha=0.4, linestyle=':')

        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                    f'{height:.2f}\nAZN', ha='center', va='bottom',
                    fontweight='bold', fontsize=10)

        plt.savefig(self.charts_dir / '02_discount_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 02_discount_analysis.png")

    def chart_3_category_analysis(self):
        """Category Market Share"""
        print("→ Chart 3: Category Analysis...")

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        fig.suptitle('Category Market Analysis & Segmentation',
                     fontsize=20, fontweight='bold', y=0.98)

        # 1. Top 18 categories
        ax1 = fig.add_subplot(gs[:, 0])
        cat_counts = self.df['category_name'].value_counts().head(18)

        colors_grad = plt.cm.viridis(np.linspace(0.3, 0.9, len(cat_counts)))
        bars = ax1.barh(range(len(cat_counts)), cat_counts.values,
                       color=colors_grad, edgecolor='black', linewidth=0.8)
        ax1.set_yticks(range(len(cat_counts)))
        ax1.set_yticklabels(cat_counts.index, fontsize=10)
        ax1.set_xlabel('Number of Products', fontsize=12, fontweight='bold')
        ax1.set_title('Top 18 Categories by Product Count', fontsize=14, fontweight='bold', pad=15)
        ax1.grid(axis='x', alpha=0.4, linestyle=':')

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 5, bar.get_y() + bar.get_height()/2.,
                    f'{int(width):,}', ha='left', va='center',
                    fontweight='bold', fontsize=9)

        # 2. Market share pie
        ax2 = fig.add_subplot(gs[0, 1])
        top_8 = cat_counts.head(8)
        others = cat_counts.iloc[8:].sum()

        pie_data = list(top_8.values) + [others]
        pie_labels = list(top_8.index) + ['Others']
        colors = plt.cm.Set3(range(len(pie_data)))

        wedges, texts, autotexts = ax2.pie(pie_data, labels=None, autopct='%1.1f%%',
                                           colors=colors, startangle=90,
                                           textprops={'fontsize': 9, 'fontweight': 'bold'})
        ax2.legend(pie_labels, loc='center left', bbox_to_anchor=(1, 0.5),
                  fontsize=9, framealpha=0.9)
        ax2.set_title('Market Share Distribution', fontsize=13, fontweight='bold', pad=15)

        # 3. Category ratings
        ax3 = fig.add_subplot(gs[1, 1])
        cat_ratings = self.df[self.df['rating_value'] > 0].groupby('category_name').agg({
            'rating_value': 'mean',
            'product_id': 'count'
        }).rename(columns={'product_id': 'count'})
        cat_ratings = cat_ratings[cat_ratings['count'] >= 15].sort_values('rating_value', ascending=True).tail(10)

        bars = ax3.barh(range(len(cat_ratings)), cat_ratings['rating_value'],
                       color='#f39c12', alpha=0.8, edgecolor='black', linewidth=0.8)
        ax3.set_yticks(range(len(cat_ratings)))
        ax3.set_yticklabels(cat_ratings.index, fontsize=9)
        ax3.set_xlabel('Average Rating', fontsize=11, fontweight='bold')
        ax3.set_title('Top 10 Highest-Rated Categories', fontsize=12, fontweight='bold', pad=12)
        ax3.set_xlim(0, 5)
        ax3.grid(axis='x', alpha=0.4, linestyle=':')

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width + 0.05, bar.get_y() + bar.get_height()/2.,
                    f'{width:.2f}★', ha='left', va='center',
                    fontweight='bold', fontsize=9)

        plt.savefig(self.charts_dir / '03_category_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 03_category_analysis.png")

    def chart_4_brand_performance(self):
        """Brand Market Performance"""
        print("→ Chart 4: Brand Performance...")

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        fig.suptitle('Brand Performance & Competitive Landscape',
                     fontsize=20, fontweight='bold', y=0.98)

        # 1. Top 25 brands by product count
        ax1 = fig.add_subplot(gs[:, 0])
        brand_counts = self.df[self.df['brand'] != 'N/A']['brand'].value_counts().head(25)

        colors_grad = plt.cm.plasma(np.linspace(0.2, 0.9, len(brand_counts)))
        bars = ax1.barh(range(len(brand_counts)), brand_counts.values,
                       color=colors_grad, edgecolor='black', linewidth=0.8)
        ax1.set_yticks(range(len(brand_counts)))
        ax1.set_yticklabels(brand_counts.index, fontsize=9)
        ax1.set_xlabel('Number of Products', fontsize=12, fontweight='bold')
        ax1.set_title('Top 25 Brands by Product Count', fontsize=14, fontweight='bold', pad=15)
        ax1.grid(axis='x', alpha=0.4, linestyle=':')

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 1, bar.get_y() + bar.get_height()/2.,
                    f'{int(width)}', ha='left', va='center',
                    fontweight='bold', fontsize=8)

        # 2. Brand average prices
        ax2 = fig.add_subplot(gs[0, 1])
        brand_prices = self.df[self.df['brand'] != 'N/A'].groupby('brand').agg({
            'retail_price': 'mean',
            'product_id': 'count'
        }).rename(columns={'product_id': 'count'})
        brand_prices = brand_prices[brand_prices['count'] >= 10].sort_values('retail_price', ascending=True).tail(12)

        bars = ax2.barh(range(len(brand_prices)), brand_prices['retail_price'],
                       color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=0.8)
        ax2.set_yticks(range(len(brand_prices)))
        ax2.set_yticklabels(brand_prices.index, fontsize=9)
        ax2.set_xlabel('Average Price (AZN)', fontsize=11, fontweight='bold')
        ax2.set_title('Top 12 Premium Brands', fontsize=12, fontweight='bold', pad=12)
        ax2.grid(axis='x', alpha=0.4, linestyle=':')

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + 0.3, bar.get_y() + bar.get_height()/2.,
                    f'{width:.2f}', ha='left', va='center',
                    fontweight='bold', fontsize=8)

        # 3. Brand discount strategy
        ax3 = fig.add_subplot(gs[1, 1])
        brand_disc = self.df[(self.df['brand'] != 'N/A') & (self.df['has_discount'] == True)].groupby('brand').agg({
            'discount_percentage': 'mean',
            'product_id': 'count'
        }).rename(columns={'product_id': 'count'})
        brand_disc = brand_disc[brand_disc['count'] >= 5].sort_values('discount_percentage', ascending=True).tail(12)

        bars = ax3.barh(range(len(brand_disc)), brand_disc['discount_percentage'],
                       color='#27ae60', alpha=0.8, edgecolor='black', linewidth=0.8)
        ax3.set_yticks(range(len(brand_disc)))
        ax3.set_yticklabels(brand_disc.index, fontsize=9)
        ax3.set_xlabel('Average Discount %', fontsize=11, fontweight='bold')
        ax3.set_title('Top 12 Brands by Discount %', fontsize=12, fontweight='bold', pad=12)
        ax3.grid(axis='x', alpha=0.4, linestyle=':')

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                    f'{width:.1f}%', ha='left', va='center',
                    fontweight='bold', fontsize=8)

        plt.savefig(self.charts_dir / '04_brand_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 04_brand_performance.png")

    def chart_5_product_labels(self):
        """Product Labels & Tags Analysis"""
        print("→ Chart 5: Product Labels Analysis...")

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        fig.suptitle('Product Labels & Marketing Tags Analysis',
                     fontsize=20, fontweight='bold', y=0.98)

        # 1. Label distribution
        ax1 = fig.add_subplot(gs[0, 0])
        label_counts = self.df['product_labels_count'].value_counts().sort_index()

        bars = ax1.bar(label_counts.index, label_counts.values,
                      color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_xlabel('Number of Labels per Product', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Products', fontsize=12, fontweight='bold')
        ax1.set_title('Distribution of Labels per Product', fontsize=14, fontweight='bold', pad=15)
        ax1.grid(axis='y', alpha=0.4, linestyle=':')

        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'{int(height):,}', ha='center', va='bottom',
                    fontweight='bold', fontsize=9)

        # 2. Offer labels distribution
        ax2 = fig.add_subplot(gs[0, 1])
        offer_label_counts = self.df['offer_labels_count'].value_counts().sort_index()

        bars = ax2.bar(offer_label_counts.index, offer_label_counts.values,
                      color='#e67e22', alpha=0.8, edgecolor='black', linewidth=1)
        ax2.set_xlabel('Number of Offer Labels per Product', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Products', fontsize=12, fontweight='bold')
        ax2.set_title('Distribution of Offer Labels', fontsize=14, fontweight='bold', pad=15)
        ax2.grid(axis='y', alpha=0.4, linestyle=':')

        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'{int(height):,}', ha='center', va='bottom',
                    fontweight='bold', fontsize=9)

        # 3. Average price by label count
        ax3 = fig.add_subplot(gs[1, 0])
        price_by_labels = self.df.groupby('product_labels_count')['retail_price'].mean()

        bars = ax3.bar(price_by_labels.index, price_by_labels.values,
                      color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=1)
        ax3.set_xlabel('Number of Labels', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Average Price (AZN)', fontsize=12, fontweight='bold')
        ax3.set_title('Average Price by Label Count', fontsize=14, fontweight='bold', pad=15)
        ax3.grid(axis='y', alpha=0.4, linestyle=':')

        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=9)

        # 4. Discount rate by label presence
        ax4 = fig.add_subplot(gs[1, 1])

        has_labels = self.df[self.df['product_labels_count'] > 0]
        no_labels = self.df[self.df['product_labels_count'] == 0]

        categories = ['With\nLabels', 'Without\nLabels']
        avg_prices = [has_labels['retail_price'].mean(), no_labels['retail_price'].mean()]
        discount_rates = [
            (has_labels['has_discount'].sum() / len(has_labels) * 100) if len(has_labels) > 0 else 0,
            (no_labels['has_discount'].sum() / len(no_labels) * 100) if len(no_labels) > 0 else 0
        ]

        x = np.arange(len(categories))
        width = 0.35

        ax4_twin = ax4.twinx()

        bars1 = ax4.bar(x - width/2, avg_prices, width, label='Avg Price (AZN)',
                       color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax4_twin.bar(x + width/2, discount_rates, width, label='Discount Rate (%)',
                            color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1)

        ax4.set_xlabel('Product Type', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Average Price (AZN)', fontsize=11, fontweight='bold', color='#3498db')
        ax4_twin.set_ylabel('Discount Rate (%)', fontsize=11, fontweight='bold', color='#e74c3c')
        ax4.set_title('Price & Discount by Label Presence', fontsize=14, fontweight='bold', pad=15)
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories, fontsize=11, fontweight='bold')
        ax4.tick_params(axis='y', labelcolor='#3498db')
        ax4_twin.tick_params(axis='y', labelcolor='#e74c3c')
        ax4.grid(alpha=0.3, linestyle=':')

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=9, color='#3498db')

        for bar in bars2:
            height = bar.get_height()
            ax4_twin.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                         f'{height:.1f}%', ha='center', va='bottom',
                         fontweight='bold', fontsize=9, color='#e74c3c')

        plt.savefig(self.charts_dir / '05_product_labels.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 05_product_labels.png")

    def chart_6_market_opportunities(self):
        """Market Opportunities"""
        print("→ Chart 6: Market Opportunities...")

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
        fig.suptitle('Market Opportunities & Strategic Gaps',
                     fontsize=20, fontweight='bold', y=0.98)

        # 1. High discount categories
        ax1 = fig.add_subplot(gs[0, 0])
        high_disc = self.df[self.df['discount_percentage'] >= 30].groupby('category_name').size().sort_values(ascending=True).tail(10)

        bars = ax1.barh(range(len(high_disc)), high_disc.values,
                       color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.8)
        ax1.set_yticks(range(len(high_disc)))
        ax1.set_yticklabels(high_disc.index, fontsize=9)
        ax1.set_xlabel('Product Count', fontsize=11, fontweight='bold')
        ax1.set_title('Categories with High Discounts (≥30%)', fontsize=12, fontweight='bold', pad=12)
        ax1.grid(axis='x', alpha=0.4, linestyle=':')

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                    f'{int(width)}', ha='left', va='center',
                    fontweight='bold', fontsize=9)

        # 2. Underserved categories
        ax2 = fig.add_subplot(gs[0, 1])
        cat_analysis = self.df.groupby('category_name').agg({
            'product_id': 'count',
            'rating_value': 'mean'
        }).rename(columns={'product_id': 'count', 'rating_value': 'avg_rating'})

        underserved = cat_analysis[(cat_analysis['count'] < 50) & (cat_analysis['count'] > 5) & (cat_analysis['avg_rating'] > 3.5)]
        underserved = underserved.sort_values('avg_rating', ascending=True).tail(10)

        if len(underserved) > 0:
            bars = ax2.barh(range(len(underserved)), underserved['count'],
                           color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.8)
            ax2.set_yticks(range(len(underserved)))
            ax2.set_yticklabels(underserved.index, fontsize=9)
            ax2.set_xlabel('Product Count', fontsize=11, fontweight='bold')
            ax2.set_title('Underserved High-Rated Categories', fontsize=12, fontweight='bold', pad=12)
            ax2.grid(axis='x', alpha=0.4, linestyle=':')

            for i, bar in enumerate(bars):
                width = bar.get_width()
                rating = underserved.iloc[i]['avg_rating']
                ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                        f'{int(width)} (★{rating:.1f})', ha='left', va='center',
                        fontweight='bold', fontsize=9)

        # 3. Price variation categories
        ax3 = fig.add_subplot(gs[1, 0])
        cat_price_var = self.df.groupby('category_name').agg({
            'retail_price': ['std', 'mean', 'count']
        })
        cat_price_var.columns = ['price_std', 'price_mean', 'count']
        cat_price_var = cat_price_var[cat_price_var['count'] >= 15]
        cat_price_var['coefficient_variation'] = (cat_price_var['price_std'] / cat_price_var['price_mean']) * 100
        cat_price_var = cat_price_var.sort_values('coefficient_variation', ascending=True).tail(10)

        bars = ax3.barh(range(len(cat_price_var)), cat_price_var['coefficient_variation'],
                       color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=0.8)
        ax3.set_yticks(range(len(cat_price_var)))
        ax3.set_yticklabels(cat_price_var.index, fontsize=9)
        ax3.set_xlabel('Price Variation (%)', fontsize=11, fontweight='bold')
        ax3.set_title('High Price Variation Categories', fontsize=12, fontweight='bold', pad=12)
        ax3.grid(axis='x', alpha=0.4, linestyle=':')

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width + 1, bar.get_y() + bar.get_height()/2.,
                    f'{width:.1f}%', ha='left', va='center',
                    fontweight='bold', fontsize=9)

        # 4. Low-rated categories
        ax4 = fig.add_subplot(gs[1, 1])
        low_rated = self.df[self.df['rating_value'] > 0].groupby('category_name').agg({
            'rating_value': 'mean',
            'product_id': 'count'
        }).rename(columns={'product_id': 'count'})
        low_rated = low_rated[low_rated['count'] >= 15].sort_values('rating_value').head(10)

        bars = ax4.barh(range(len(low_rated)), low_rated['rating_value'],
                       color='#e67e22', alpha=0.8, edgecolor='black', linewidth=0.8)
        ax4.set_yticks(range(len(low_rated)))
        ax4.set_yticklabels(low_rated.index, fontsize=9)
        ax4.set_xlabel('Average Rating', fontsize=11, fontweight='bold')
        ax4.set_title('Lowest-Rated Categories', fontsize=12, fontweight='bold', pad=12)
        ax4.set_xlim(0, 5)
        ax4.grid(axis='x', alpha=0.4, linestyle=':')

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax4.text(width + 0.05, bar.get_y() + bar.get_height()/2.,
                    f'{width:.2f}★', ha='left', va='center',
                    fontweight='bold', fontsize=9)

        # 5. Summary text box
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')

        total_discounted = len(self.df[self.df['has_discount'] == True])
        avg_discount = self.df[self.df['has_discount'] == True]['discount_percentage'].mean()
        total_categories = self.df['category_name'].nunique()
        total_brands = len(self.df[self.df['brand'] != 'N/A']['brand'].unique())

        summary_text = f"""
KEY MARKET OPPORTUNITIES

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 MARKET OVERVIEW:
• Total Products: {len(self.df):,}  |  Categories: {total_categories}  |  Brands: {total_brands}
• Products with Discounts: {total_discounted:,} ({total_discounted/len(self.df)*100:.1f}%)
• Average Discount: {avg_discount:.1f}%

🎯 TOP STRATEGIC OPPORTUNITIES:

1. DISCOUNT COMPETITION: {avg_discount:.1f}% avg discount shows aggressive pricing environment
   → Action: Enter categories with 30%+ discounts for competitive advantage

2. UNDERSERVED NICHES: {len(underserved)} high-quality categories with limited products (<50)
   → Action: Target low-competition, high-satisfaction segments

3. PRICE OPTIMIZATION: High price variance signals opportunity for mid-market positioning
   → Action: Fill pricing gaps between budget and premium offerings

4. QUALITY IMPROVEMENT: Several high-volume categories have ratings below 3.5
   → Action: Compete on quality in categories with dissatisfied customers

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """

        ax5.text(0.5, 0.5, summary_text, ha='center', va='center',
                fontsize=10, family='monospace', fontweight='bold',
                bbox=dict(boxstyle='round,pad=1', facecolor='#f0f0f0',
                         edgecolor='black', linewidth=2, alpha=0.9))

        plt.savefig(self.charts_dir / '06_market_opportunities.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 06_market_opportunities.png")

    def chart_7_pricing_strategy(self):
        """Pricing Strategy Analysis"""
        print("→ Chart 7: Pricing Strategy...")

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        fig.suptitle('Competitive Pricing Strategy Analysis',
                     fontsize=20, fontweight='bold', y=0.98)

        # 1. Price vs count scatter
        ax1 = fig.add_subplot(gs[0, 0])
        cat_price_count = self.df.groupby('category_name').agg({
            'retail_price': 'mean',
            'product_id': 'count'
        }).rename(columns={'product_id': 'count'})
        cat_price_count = cat_price_count[cat_price_count['count'] >= 10]

        scatter = ax1.scatter(cat_price_count['retail_price'], cat_price_count['count'],
                            s=120, alpha=0.6, c=cat_price_count['count'],
                            cmap='viridis', edgecolor='black', linewidth=0.8)
        ax1.set_xlabel('Average Price (AZN)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Product Count', fontsize=12, fontweight='bold')
        ax1.set_title('Price vs Product Volume', fontsize=14, fontweight='bold', pad=15)
        ax1.grid(alpha=0.4, linestyle=':')

        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Count', fontsize=10, fontweight='bold')

        # 2. Discount by price range
        ax2 = fig.add_subplot(gs[0, 1])
        discounted = self.df[self.df['has_discount'] == True].copy()
        discounted['price_range'] = pd.cut(discounted['retail_price'],
                                          bins=[0, 5, 10, 20, 50, discounted['retail_price'].max()],
                                          labels=['0-5', '5-10', '10-20', '20-50', '50+'])

        range_disc = discounted.groupby('price_range')['discount_percentage'].mean().dropna()

        colors_grad = plt.cm.Reds(np.linspace(0.4, 0.9, len(range_disc)))
        bars = ax2.bar(range(len(range_disc)), range_disc.values,
                      color=colors_grad, edgecolor='black', linewidth=1.2, width=0.6)
        ax2.set_xticks(range(len(range_disc)))
        ax2.set_xticklabels(range_disc.index, fontsize=11, fontweight='bold')
        ax2.set_xlabel('Price Range (AZN)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Avg Discount %', fontsize=12, fontweight='bold')
        ax2.set_title('Discount Strategy by Price Tier', fontsize=14, fontweight='bold', pad=15)
        ax2.grid(axis='y', alpha=0.4, linestyle=':')

        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom',
                    fontweight='bold', fontsize=10)

        # 3. Price positioning scatter
        ax3 = fig.add_subplot(gs[1, 0])
        has_old_price = self.df[self.df['old_price'] > 0]

        ax3.scatter(has_old_price['old_price'], has_old_price['retail_price'],
                   alpha=0.4, s=40, c='#3498db', edgecolor='none')

        max_price = max(has_old_price['old_price'].max(), has_old_price['retail_price'].max())
        ax3.plot([0, max_price], [0, max_price], 'r--', linewidth=2.5,
                label='No Discount Line', alpha=0.8)

        ax3.set_xlabel('Original Price (AZN)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Retail Price (AZN)', fontsize=12, fontweight='bold')
        ax3.set_title('Original vs Retail Pricing', fontsize=14, fontweight='bold', pad=15)
        ax3.legend(fontsize=11, framealpha=0.9)
        ax3.grid(alpha=0.4, linestyle=':')
        ax3.set_xlim(0, has_old_price['old_price'].quantile(0.95))
        ax3.set_ylim(0, has_old_price['retail_price'].quantile(0.95))

        # 4. Brand price comparison
        ax4 = fig.add_subplot(gs[1, 1])
        top_brands = self.df[self.df['brand'] != 'N/A']['brand'].value_counts().head(10).index
        brand_price_data = []

        for brand in top_brands:
            brand_data = self.df[self.df['brand'] == brand]['retail_price'].dropna()
            if len(brand_data) > 5:
                brand_price_data.append(brand_data)

        if brand_price_data:
            bp = ax4.boxplot(brand_price_data, labels=[b[:15] for b in top_brands[:len(brand_price_data)]],
                            patch_artist=True, showfliers=False)

            for patch, color in zip(bp['boxes'], plt.cm.Set3(range(len(brand_price_data)))):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor('black')
                patch.set_linewidth(1)

            for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bp[element], color='black', linewidth=1.5)

            ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=9)
            ax4.set_ylabel('Price (AZN)', fontsize=12, fontweight='bold')
            ax4.set_title('Price Distribution by Top Brands', fontsize=14, fontweight='bold', pad=15)
            ax4.grid(axis='y', alpha=0.4, linestyle=':')

        plt.savefig(self.charts_dir / '07_pricing_strategy.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 07_pricing_strategy.png")

    def chart_8_product_assortment(self):
        """Product Assortment & Variety"""
        print("→ Chart 8: Product Assortment...")

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
        fig.suptitle('Product Assortment & Category Diversity',
                     fontsize=20, fontweight='bold', y=0.98)

        # 1. Category size distribution
        ax1 = fig.add_subplot(gs[0, :2])
        cat_sizes = self.df['category_name'].value_counts()
        size_bins = pd.cut(cat_sizes, bins=[0, 10, 25, 50, 100, 200, cat_sizes.max()],
                          labels=['1-10', '11-25', '26-50', '51-100', '101-200', '200+'])
        size_counts = size_bins.value_counts().sort_index()

        colors_grad = plt.cm.Blues(np.linspace(0.4, 0.9, len(size_counts)))
        bars = ax1.bar(range(len(size_counts)), size_counts.values,
                      color=colors_grad, edgecolor='black', linewidth=1.2, width=0.7)
        ax1.set_xticks(range(len(size_counts)))
        ax1.set_xticklabels(size_counts.index, fontsize=11, fontweight='bold')
        ax1.set_xlabel('Products per Category', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Categories', fontsize=12, fontweight='bold')
        ax1.set_title('Category Size Distribution', fontsize=14, fontweight='bold', pad=15)
        ax1.grid(axis='y', alpha=0.4, linestyle=':')

        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom',
                    fontweight='bold', fontsize=11)

        # 2. Rating distribution
        ax2 = fig.add_subplot(gs[0, 2])
        ratings = self.df[self.df['rating_value'] > 0]['rating_value']

        ax2.hist(ratings, bins=20, color='#f39c12', alpha=0.8, edgecolor='black', linewidth=0.8)
        ax2.axvline(ratings.mean(), color='#e74c3c', linestyle='--', linewidth=2.5,
                   label=f'Mean: {ratings.mean():.2f}')
        ax2.set_xlabel('Rating', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Product Count', fontsize=11, fontweight='bold')
        ax2.set_title('Rating Distribution', fontsize=12, fontweight='bold', pad=12)
        ax2.legend(fontsize=10, framealpha=0.9)
        ax2.grid(alpha=0.4, linestyle=':')

        # 3. Brand diversity by category
        ax3 = fig.add_subplot(gs[1, :2])
        cat_brands = self.df[self.df['brand'] != 'N/A'].groupby('category_name')['brand'].nunique().sort_values(ascending=True).tail(12)

        bars = ax3.barh(range(len(cat_brands)), cat_brands.values,
                       color='#27ae60', alpha=0.8, edgecolor='black', linewidth=0.8)
        ax3.set_yticks(range(len(cat_brands)))
        ax3.set_yticklabels(cat_brands.index, fontsize=10)
        ax3.set_xlabel('Number of Unique Brands', fontsize=12, fontweight='bold')
        ax3.set_title('Top 12 Categories by Brand Diversity', fontsize=14, fontweight='bold', pad=15)
        ax3.grid(axis='x', alpha=0.4, linestyle=':')

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width + 0.3, bar.get_y() + bar.get_height()/2.,
                    f'{int(width)}', ha='left', va='center',
                    fontweight='bold', fontsize=9)

        # 4. Market concentration metrics
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.axis('off')

        total_products = len(self.df)
        total_categories = self.df['category_name'].nunique()
        total_brands = len(self.df[self.df['brand'] != 'N/A']['brand'].unique())
        avg_products_per_cat = total_products / total_categories
        avg_price = self.df['retail_price'].mean()
        median_price = self.df['retail_price'].median()

        top_5_cat_share = (self.df['category_name'].value_counts().head(5).sum() / total_products * 100)
        top_10_brand_share = (self.df[self.df['brand'] != 'N/A']['brand'].value_counts().head(10).sum() / len(self.df[self.df['brand'] != 'N/A']) * 100)

        metrics_text = f"""
MARKET METRICS

━━━━━━━━━━━━━━━━━━━━

📊 ASSORTMENT:
Products: {total_products:,}
Categories: {total_categories}
Brands: {total_brands}

📈 AVERAGES:
Prod/Category: {avg_products_per_cat:.1f}
Avg Price: {avg_price:.2f} AZN
Median Price: {median_price:.2f} AZN

🎯 CONCENTRATION:
Top 5 Categories: {top_5_cat_share:.1f}%
Top 10 Brands: {top_10_brand_share:.1f}%

━━━━━━━━━━━━━━━━━━━━
        """

        ax4.text(0.5, 0.5, metrics_text, ha='center', va='center',
                fontsize=11, family='monospace', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#e8f4f8',
                         edgecolor='#2980b9', linewidth=2, alpha=0.9))

        plt.savefig(self.charts_dir / '08_product_assortment.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 08_product_assortment.png")


def main():
    """Main execution"""
    print("\n" + "="*70)
    print(" "*18 + "FOOD MARKET ANALYSIS")
    print("="*70)

    analyzer = FoodMarketAnalysis('umico_products.csv')
    analyzer.generate_all_charts()

    print("\n" + "="*70)
    print("✓ Analysis complete! Charts saved in /charts directory")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
