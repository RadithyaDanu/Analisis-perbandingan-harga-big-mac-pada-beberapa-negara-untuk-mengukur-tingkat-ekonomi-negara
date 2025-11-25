import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def create_economic_labels(df):
    df_copy = df.copy()

    economic_labels = []
    
    for _, row in df_copy.iterrows():
        price = row['Dollar_Price']
        valuation = row['Valuation_Percent']

        if price >= 4.5 and valuation >= -25:
            economic_labels.append('Maju')
        elif price >= 3.0 and valuation >= -45:
            economic_labels.append('Berkembang')
        else:
            economic_labels.append('Emerging')
    
    df_copy['Economic_Level'] = economic_labels
    return df_copy

def decision_tree_analysis():
    print("BIG MAC INDEX DECISION TREE CLASSIFICATION")
    print("=" * 55)
    
    try:
        df = pd.read_csv('real_bigmac_clustering_results.csv')
        print(f"Data loaded: {len(df)} countries")

        df = create_economic_labels(df)

        print(f"\nEconomic Level Distribution:")
        level_counts = df['Economic_Level'].value_counts()
        for level, count in level_counts.items():
            percentage = (count/len(df))*100
            print(f"   {level}: {count} negara ({percentage:.1f}%)")
        
        # Prepare features and target
        X = df[['Dollar_Price', 'Valuation_Percent']]
        y = df['Economic_Level']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train Decision Tree
        dt = DecisionTreeClassifier(
            max_depth=4, 
            min_samples_split=3,
            random_state=42
        )
        dt.fit(X_train, y_train)
        
        # Model accuracy
        y_pred = dt.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Accuracy: {accuracy:.3f}")
        
        # Visualize Decision Tree
        plt.figure(figsize=(12, 8))
        plot_tree(dt, filled=True, 
                 feature_names=['Dollar_Price', 'Valuation_Percent'],
                 class_names=dt.classes_, 
                 rounded=True, fontsize=10)
        plt.title('Big Mac Index Decision Tree Classification', fontsize=14, fontweight='bold')
        plt.show()
        
        # Predict all countries
        all_predictions = dt.predict(X)
        df['DT_Prediction'] = all_predictions
        
        # Show country classifications
        print(f"\n Country Classifications:")
        for level in ['Maju', 'Berkembang', 'Emerging']:
            countries = df[df['DT_Prediction'] == level]
            print(f"\n{level} ({len(countries)} negara):")
            for _, row in countries.iterrows():
                print(f"   • {row['Country']}: ${row['Dollar_Price']} ({row['Valuation_Percent']:+.0f}%)")
        
        # Export results
        df.to_csv('bigmac_decision_tree_results.csv', index=False)
        print(f"\n Results exported to bigmac_decision_tree_results.csv")
        
        return dt, df
        
    except FileNotFoundError:
        print("Error: File 'real_bigmac_clustering_results.csv' tidak ditemukan")
        return None, None
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None, None

if __name__ == "__main__":
    print("Starting Decision Tree Analysis...")
    model, data = decision_tree_analysis()
    
    if model is not None:
        print("\nDecision Tree analysis complete!")
    else:

        print("\nAnalysis failed.")

