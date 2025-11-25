
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

def create_real_bigmac_dataset():
    data = {
        'Country': [
            'United States', 'Argentina', 'Australia', 'Brazil', 'Britain',
            'Canada', 'Chile', 'China', 'Czech Republic', 'Denmark',
            'Egypt', 'Euro area', 'Hong Kong', 'Hungary', 'India',
            'Indonesia', 'Israel', 'Japan', 'Malaysia', 'Mexico',
            'New Zealand', 'Peru', 'Philippines', 'Poland', 'Russia',
            'Singapore', 'South Africa', 'South Korea', 'Sweden', 
            'Switzerland', 'Taiwan', 'Thailand', 'Turkey', 'Ukraine'
        ],
        'Dollar_Price': [
            5.04, 3.35, 4.30, 4.78, 3.94, 4.60, 3.53, 2.79, 3.06, 4.44,
            2.59, 4.21, 2.48, 3.15, 2.41, 2.36, 4.38, 3.47, 4.03, 2.37,
            4.22, 3.02, 2.82, 2.42, 2.05, 4.01, 2.10, 3.86, 5.23,
            6.59, 2.15, 3.40, 3.53, 1.57
        ],
        'Valuation_Percent': [
            0, -34, -15, -5, -22, -9, -30, -45, -39, -12,
            -49, -17, -51, -38, -52, -53, -13, -31, -20, -53,
            -16, -40, -44, -52, -59, -20, -58, -24, 4,
            31, -57, -32, -30, -69
        ]
    }
    return pd.DataFrame(data)

def find_optimal_k(scaled_features, max_k=8):
    k_range = range(2, min(max_k + 1, len(scaled_features)))
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            labels = kmeans.fit_predict(scaled_features)
            inertias.append(kmeans.inertia_)
            
            if len(set(labels)) > 1:
                sil_score = silhouette_score(scaled_features, labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)
                
        except Exception as e:
            print(f"Error untuk k={k}: {str(e)}")
            continue
    
    return k_range, inertias, silhouette_scores

def enhanced_kmeans_analysis():
    print("REAL BIG MAC INDEX K-MEANS CLUSTERING")
    print("=" * 50)
    
    try:
        df = create_real_bigmac_dataset()
        print(f"Real dataset loaded: {len(df)} countries")
        
        if df.isnull().any().any():
            print("Warning: Missing values detected. Cleaning data...")
            df = df.dropna()
        
        print("\nData Preview:")
        print(df.head())
        print(f"\nPrice range: ${df['Dollar_Price'].min():.2f} - ${df['Dollar_Price'].max():.2f}")
        print(f"Valuation range: {df['Valuation_Percent'].min()}% to {df['Valuation_Percent'].max()}%")
        
        features = df[['Dollar_Price', 'Valuation_Percent']].copy()
        
        if len(features) < 3:
            raise ValueError("Not enough data points for clustering")
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        print("\nFinding optimal number of clusters...")
        k_range, inertias, silhouette_scores = find_optimal_k(scaled_features)
        
        if not inertias:
            raise ValueError("Could not compute clustering metrics")

        try:
            plt.figure(figsize=(15, 5))
            
            # Elbow plot
            plt.subplot(1, 3, 1)
            plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
            plt.title('Elbow Method', fontsize=12, fontweight='bold')
            plt.xlabel('Number of clusters (k)')
            plt.ylabel('Inertia')
            plt.grid(True, alpha=0.3)
            
            # Silhouette plot
            if silhouette_scores:
                plt.subplot(1, 3, 2)
                plt.plot(list(k_range), silhouette_scores, 'ro-', linewidth=2, markersize=8)
                plt.title('Silhouette Analysis', fontsize=12, fontweight='bold')
                plt.xlabel('Number of clusters (k)')
                plt.ylabel('Silhouette Score')
                plt.grid(True, alpha=0.3)
        
        except Exception as e:
            print(f"Warning: Could not create elbow/silhouette plots: {str(e)}")
        
        optimal_k = 3
        
        if silhouette_scores and len(k_range) >= 2:
            print(f"Silhouette scores for different k values:")
            for i, k in enumerate(k_range):
                if i < len(silhouette_scores):
                    print(f"   k={k}: {silhouette_scores[i]:.3f}")
        
        print(f"Using k={optimal_k} for economic classification analysis")
        
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
        clusters = kmeans.fit_predict(scaled_features)
        df['Cluster'] = clusters

        try:
            plt.subplot(1, 3, 3)
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            
            for i in range(optimal_k):
                cluster_data = df[df['Cluster'] == i]
                if len(cluster_data) > 0:
                    plt.scatter(cluster_data['Dollar_Price'], 
                               cluster_data['Valuation_Percent'],
                               c=colors[i % len(colors)], label=f'Cluster {i}', 
                               alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
            
            # Add centroids
            centroids = scaler.inverse_transform(kmeans.cluster_centers_)
            plt.scatter(centroids[:, 0], centroids[:, 1], 
                       c='red', marker='x', s=200, linewidths=3, label='Centroids')
            
            plt.title('K-Means Clustering Results', fontsize=12, fontweight='bold')
            plt.xlabel('Big Mac Price (USD)')
            plt.ylabel('Currency Valuation (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Warning: Could not create clustering plot: {str(e)}")
        
        print(f"\nDETAILED CLUSTER ANALYSIS (k={optimal_k}):")
        print("=" * 60)
        
        for i in range(optimal_k):
            cluster_data = df[df['Cluster'] == i]
            if len(cluster_data) == 0:
                continue
                
            avg_price = cluster_data['Dollar_Price'].mean()
            avg_valuation = cluster_data['Valuation_Percent'].mean()
            price_std = cluster_data['Dollar_Price'].std() if len(cluster_data) > 1 else 0
            val_std = cluster_data['Valuation_Percent'].std() if len(cluster_data) > 1 else 0
            
            print(f"\nCLUSTER {i}: {len(cluster_data)} countries")
            print(f"   Average price: ${avg_price:.2f} (±${price_std:.2f})")
            print(f"   Average valuation: {avg_valuation:.1f}% (±{val_std:.1f}%)")
            
            if avg_price >= 4.5 and avg_valuation >= -25:
                category = "DEVELOPED MARKETS (Advanced economies)"
                strategy = "Premium pricing, quality focus, established infrastructure"
                eco_level = "Maju"
            elif avg_price >= 3.0 and avg_valuation >= -45:
                category = "DEVELOPING MARKETS (Growing economies)"
                strategy = "Balanced pricing, market expansion, infrastructure building"
                eco_level = "Berkembang"
            else:
                category = "EMERGING MARKETS (Cost-competitive economies)"
                strategy = "Value pricing, volume strategy, market penetration"
                eco_level = "Emerging"
            
            print(f"   Economic Category: {category}")
            print(f"   Economic Level: {eco_level}")
            print(f"   Business Strategy: {strategy}")
            
            # Currency analysis
            if avg_valuation <= -40:
                currency_status = "SIGNIFICANTLY UNDERVALUED"
            elif avg_valuation <= -20:
                currency_status = "MODERATELY UNDERVALUED"
            elif avg_valuation <= 10:
                currency_status = "FAIRLY VALUED"
            else:
                currency_status = "OVERVALUED"
            
            print(f"   Currency Status: {currency_status}")
            
            # Top countries in cluster
            top_countries = cluster_data.nlargest(min(3, len(cluster_data)), 'Dollar_Price')['Country'].tolist()
            print(f"   Top Countries: {', '.join(top_countries)}")
        
        try:
            silhouette_avg = silhouette_score(scaled_features, clusters)
            print(f"\nCLUSTERING PERFORMANCE:")
            print(f"   Silhouette Score: {silhouette_avg:.3f} {'(Good)' if silhouette_avg > 0.5 else '(Fair)' if silhouette_avg > 0.3 else '(Poor)'}")
            print(f"   Inertia: {kmeans.inertia_:.2f}")
            print(f"   Optimal K: {optimal_k}")
        except:
            print("\nCLUSTERING PERFORMANCE: Could not calculate some metrics")

        print(f"\nECONOMIC CLASSIFICATION INSIGHTS:")
        print("=" * 50)        
        cluster_classification = {}
        for i in range(optimal_k):
            cluster_data = df[df['Cluster'] == i]
            if len(cluster_data) == 0:
                continue
            avg_price = cluster_data['Dollar_Price'].mean()
            avg_valuation = cluster_data['Valuation_Percent'].mean()
            
            if avg_price >= 4.5 and avg_valuation >= -25:
                cluster_classification[i] = "Maju"
            elif avg_price >= 3.0 and avg_valuation >= -45:
                cluster_classification[i] = "Berkembang"
            else:
                cluster_classification[i] = "Emerging"
        
        # Show countries by economic level
        for level in ["Maju", "Berkembang", "Emerging"]:
            countries_in_level = []
            for cluster_id, classification in cluster_classification.items():
                if classification == level:
                    cluster_countries = df[df['Cluster'] == cluster_id]['Country'].tolist()
                    countries_in_level.extend(cluster_countries)
            
            if countries_in_level:
                print(f"\nNEGARA {level.upper()}:")
                for country in sorted(countries_in_level):
                    price = df[df['Country'] == country]['Dollar_Price'].iloc[0]
                    valuation = df[df['Country'] == country]['Valuation_Percent'].iloc[0]
                    print(f"   • {country}: ${price} ({valuation:+.0f}%)")
        
        # Most expensive vs cheapest
        most_expensive = df.nlargest(3, 'Dollar_Price')[['Country', 'Dollar_Price', 'Cluster']]
        cheapest = df.nsmallest(3, 'Dollar_Price')[['Country', 'Dollar_Price', 'Cluster']]
        
        print("\nMost Expensive Markets:")
        for _, row in most_expensive.iterrows():
            print(f"   {row['Country']}: ${row['Dollar_Price']} (Cluster {row['Cluster']})")
        
        print("\nMost Cost-Effective Markets:")
        for _, row in cheapest.iterrows():
            print(f"   {row['Country']}: ${row['Dollar_Price']} (Cluster {row['Cluster']})")
        
        # Investment opportunities analysis
        print(f"\nINVESTMENT & BUSINESS OPPORTUNITIES:")
        print("-" * 45)
        
        # High-value markets (potential for premium products)
        high_value = df[df['Dollar_Price'] >= 4.5][['Country', 'Dollar_Price', 'Valuation_Percent']]
        if len(high_value) > 0:
            print("\nPremium Market Opportunities (High purchasing power):")
            for _, row in high_value.iterrows():
                print(f"   • {row['Country']}: ${row['Dollar_Price']} - Strong consumer market")
        
        # Expansion opportunities (growing markets)
        growing_markets = df[(df['Dollar_Price'] >= 3.0) & (df['Dollar_Price'] < 4.5)][['Country', 'Dollar_Price', 'Valuation_Percent']]
        if len(growing_markets) > 0:
            print("\nMarket Expansion Opportunities (Growing economies):")
            for _, row in growing_markets.iterrows():
                print(f"   • {row['Country']}: ${row['Dollar_Price']} - Balanced growth potential")
        
        # Cost-effective markets (volume opportunities)
        cost_effective = df[df['Dollar_Price'] < 3.0][['Country', 'Dollar_Price', 'Valuation_Percent']]
        if len(cost_effective) > 0:
            print("\nVolume Market Opportunities (Cost-competitive):")
            for _, row in cost_effective.iterrows():
                print(f"   • {row['Country']}: ${row['Dollar_Price']} - High volume potential")
        
        # Currency arbitrage opportunities
        print(f"\n💱 CURRENCY ARBITRAGE ANALYSIS:")
        print("-" * 35)
        undervalued = df[df['Valuation_Percent'] <= -50][['Country', 'Valuation_Percent', 'Dollar_Price']]
        if len(undervalued) > 0:
            print("Significantly Undervalued Currencies (>50% undervalued):")
            for _, row in undervalued.iterrows():
                print(f"   • {row['Country']}: {row['Valuation_Percent']}% undervalued, ${row['Dollar_Price']}")
        else:
            print("No significantly undervalued currencies found (threshold: -50%)")
        
        # Summary statistics
        print(f"\nCLUSTER DISTRIBUTION SUMMARY:")
        print("-" * 35)
        for level in ["Maju", "Berkembang", "Emerging"]:
            count = sum(1 for classification in cluster_classification.values() if classification == level)
            if count > 0:
                total_countries = sum(len(df[df['Cluster'] == cluster_id]) 
                                    for cluster_id, classification in cluster_classification.items() 
                                    if classification == level)
                print(f"   {level}: {total_countries} negara dalam {count} cluster")
            else:
                print(f"   {level}: 0 negara")
        
        return df, kmeans
        
    except Exception as e:
        print(f"Error dalam analisis: {str(e)}")
        print("Troubleshooting tips:")
        print("1. Pastikan semua library terinstall: pip install pandas numpy matplotlib scikit-learn")
        print("2. Check data untuk missing values atau outliers")
        print("3. Reduce jumlah cluster jika dataset kecil")
        return None, None

def export_results(df, kmeans):
    """Export results to CSV with error handling"""
    if df is None or kmeans is None:
        print("Cannot export results - clustering failed")
        return
        
    try:
        # Add cluster centers to dataframe for reference
        scaler = StandardScaler()
        features = df[['Dollar_Price', 'Valuation_Percent']].copy()
        scaler.fit(features)
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        
        # Create summary
        summary_data = []
        for i in range(len(centroids)):
            cluster_data = df[df['Cluster'] == i]
            if len(cluster_data) > 0:
                summary_data.append({
                    'Cluster': i,
                    'Count': len(cluster_data),
                    'Avg_Price': cluster_data['Dollar_Price'].mean(),
                    'Avg_Valuation': cluster_data['Valuation_Percent'].mean(),
                    'Centroid_Price': centroids[i][0],
                    'Centroid_Valuation': centroids[i][1]
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Export
        df.to_csv('real_bigmac_clustering_results.csv', index=False)
        summary_df.to_csv('clustering_summary.csv', index=False)
        
        print(f"\nResults exported:")
        print(f"   - real_bigmac_clustering_results.csv ({len(df)} rows)")
        print(f"   - clustering_summary.csv ({len(summary_df)} clusters)")
        
    except Exception as e:
        print(f"Error exporting results: {str(e)}")

if __name__ == "__main__":
    print("Starting Big Mac Index K-Means Analysis...")
    
    # enhanced analysis
    results_df, model = enhanced_kmeans_analysis()
    
    if results_df is not None and model is not None:
        export_results(results_df, model)
        print("\nEnhanced Big Mac Index analysis complete!")
        print("Check the generated plots and CSV files for detailed results.")
    else:

        print("\nAnalysis failed. Please check the error messages above.")
