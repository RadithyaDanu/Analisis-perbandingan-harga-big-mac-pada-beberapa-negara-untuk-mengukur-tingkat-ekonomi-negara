import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import norm

class SimpleNaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.feature_means = {}
        self.feature_vars = {}
        self.classes = []
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples = len(y)
        
        for class_val in self.classes:
            class_mask = y == class_val
            self.class_priors[class_val] = np.sum(class_mask) / n_samples

            self.feature_means[class_val] = X[class_mask].mean()
            self.feature_vars[class_val] = X[class_mask].var() + 1e-6
    
    def _gaussian_prob(self, x, mean, var):
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))
    
    def predict(self, X):
        predictions = []
        
        for _, sample in X.iterrows():
            class_scores = {}
            
            for class_val in self.classes:
                score = self.class_priors[class_val]

                for feature in X.columns:
                    mean = self.feature_means[class_val][feature]
                    var = self.feature_vars[class_val][feature]
                    score *= self._gaussian_prob(sample[feature], mean, var)
                
                class_scores[class_val] = score

            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
        
        return np.array(predictions)

def run_simple_analysis():
    """Jalankan analisis Naive Bayes sederhana"""
    print("NAIVE BAYES SEDERHANA - BIG MAC INDEX")
    print("=" * 50)
    
    # Load data
    try:
        df = pd.read_csv('real_bigmac_clustering_results.csv')
        print(f"Data berhasil dimuat: {df.shape[0]} negara")
    except FileNotFoundError:
        print("File CSV tidak ditemukan!")
        return

    X = df[['Dollar_Price', 'Valuation_Percent']]
    y = df['Cluster']

    cluster_names = {0: 'Emerging', 1: 'Berkembang', 2: 'Maju'}
    
    print(f"\nDistribusi Kelas:")
    for cluster, count in y.value_counts().sort_index().items():
        print(f"   {cluster_names[cluster]}: {count} negara")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nMelatih model..")
    nb = SimpleNaiveBayes()
    nb.fit(X_train, y_train)

    print(f"\nPARAMETER MODEL:")
    print("Prior Probabilities:")
    for class_val in nb.classes:
        print(f"   P({cluster_names[class_val]}) = {nb.class_priors[class_val]:.3f}")
    
    print(f"\nMean & Variance per Kelas:")
    for class_val in nb.classes:
        print(f"   {cluster_names[class_val]}:")
        for feature in X.columns:
            mean = nb.feature_means[class_val][feature]
            var = nb.feature_vars[class_val][feature]
            print(f"     {feature}: μ={mean:.2f}, σ²={var:.2f}")
    
    # Evaluasi
    y_pred = nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nHASIL EVALUASI:")
    print(f"Akurasi: {accuracy:.3f}")
    
    target_names = [cluster_names[i] for i in sorted(y.unique())]
    print(f"\nLaporan Klasifikasi:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Visualisasi 
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Distribusi fitur per kelas
    plt.subplot(1, 3, 1)
    colors = ['red', 'green', 'blue']
    for i, class_val in enumerate(nb.classes):
        class_data = X_train[y_train == class_val]
        plt.scatter(class_data['Dollar_Price'], class_data['Valuation_Percent'], 
                   c=colors[i], label=cluster_names[class_val], alpha=0.6)
    plt.xlabel('Dollar Price ($)')
    plt.ylabel('Valuation Percent (%)')
    plt.title('Data Training per Kelas')
    plt.legend()
    
    # Plot 2: Prior probabilities
    plt.subplot(1, 3, 2)
    classes = [cluster_names[c] for c in nb.classes]
    priors = [nb.class_priors[c] for c in nb.classes]
    plt.bar(classes, priors, color=['lightcoral', 'lightgreen', 'lightblue'])
    plt.title('Prior Probabilities')
    plt.ylabel('P(Class)')
    
    # Plot 3: Hasil prediksi
    plt.subplot(1, 3, 3)
    correct = y_test == y_pred
    incorrect = ~correct
    
    plt.scatter(X_test.loc[correct, 'Dollar_Price'], X_test.loc[correct, 'Valuation_Percent'], 
               c='green', marker='o', label='Benar', alpha=0.7)
    plt.scatter(X_test.loc[incorrect, 'Dollar_Price'], X_test.loc[incorrect, 'Valuation_Percent'], 
               c='red', marker='x', label='Salah', s=100)
    plt.xlabel('Dollar Price ($)')
    plt.ylabel('Valuation Percent (%)')
    plt.title(f'Hasil Prediksi (Akurasi: {accuracy:.2f})')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nCONTOH PREDIKSI:")

    def predict_manual(price, valuation, country_name):
        sample = pd.DataFrame({'Dollar_Price': [price], 'Valuation_Percent': [valuation]})
        prediction = nb.predict(sample)[0]
        
        print(f"\n{country_name}:")
        print(f"   Harga: ${price}, Valuasi: {valuation}%")
        print(f"   Prediksi: {cluster_names[prediction]}")
        
        # Probabilitas untuk setiap kelas
        print("   Probabilitas:")
        for class_val in nb.classes:
            score = nb.class_priors[class_val]
            for feature in sample.columns:
                mean = nb.feature_means[class_val][feature]
                var = nb.feature_vars[class_val][feature]
                score *= nb._gaussian_prob(sample[feature].iloc[0], mean, var)
            print(f"     {cluster_names[class_val]}: {score:.6f}")
    
    predict_manual(6.0, 20, "Negara Mahal")
    predict_manual(2.0, -60, "Negara Murah") 
    predict_manual(4.0, -10, "Negara Sedang")
    
    print(f"\nAnalisis selesai!")
    return nb

if __name__ == "__main__":

    model = run_simple_analysis()
