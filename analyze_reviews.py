import pandas as pd
from transformers import pipeline

def run_analysis():
    print("--- 🤖 Loading BERT Sentiment Model ---")
    # This downloads the model on the first run (about 670MB)
    classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    try:
        # 1. Load your data
        df = pd.read_csv("data.csv")
        
        # 2. Run analysis (assuming your reviews are in a column named 'review')
        print("--- 📊 Analyzing Reviews... ---")
        results = classifier(df['review'].tolist(), truncation=True)
        
        # 3. Add results back to the dataframe
        df['sentiment'] = [res['label'] for res in results]
        df['score'] = [res['score'] for res in results]
        
        # 4. Save the results
        df.to_csv("analyzed_reviews.csv", index=False)
        print("--- ✅ Success! Results saved to analyzed_reviews.csv ---")

    except FileNotFoundError:
        print("--- ❌ Error: 'data.csv' not found in this folder! ---")

if __name__ == "__main__":
    run_analysis()