from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import FINBERT_MODEL


class FinBERTSentimentAnalyzer:
    
    def __init__(self, model_name=FINBERT_MODEL):
        print(f"Loading FinBERT: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        self.labels = ['positive', 'negative', 'neutral']
        
        print(f"Loaded on {self.device}")
    
    def analyze_sentiment(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        scores = predictions[0].cpu().numpy()
        sentiment_idx = np.argmax(scores)
        sentiment_label = self.labels[sentiment_idx]
        
        return {
            'label': sentiment_label,
            'positive': float(scores[0]),
            'negative': float(scores[1]),
            'neutral': float(scores[2]),
            'confidence': float(scores[sentiment_idx])
        }
    
    def analyze_batch(self, texts, batch_size=8):
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            for j, scores in enumerate(predictions.cpu().numpy()):
                sentiment_idx = np.argmax(scores)
                sentiment_label = self.labels[sentiment_idx]
                
                results.append({
                    'text': batch[j][:100] + '...' if len(batch[j]) > 100 else batch[j],
                    'label': sentiment_label,
                    'positive': float(scores[0]),
                    'negative': float(scores[1]),
                    'neutral': float(scores[2]),
                    'confidence': float(scores[sentiment_idx])
                })
        
        print(f"Analyzed {len(results)} texts")
        return results
    
    def get_sentiment_score(self, text):
        # simple score: positive - negative
        result = self.analyze_sentiment(text)
        score = result['positive'] - result['negative']
        return score


# quick test
if __name__ == "__main__":
    analyzer = FinBERTSentimentAnalyzer()
    
    samples = [
        "Apple reports record-breaking quarterly earnings, stock surges",
        "Tech stocks plummet amid recession fears",
        "Microsoft announces new AI partnership, investors optimistic"
    ]
    
    print("\nTesting...")
    for sample in samples:
        result = analyzer.analyze_sentiment(sample)
        print(f"\nText: {sample}")
        print(f"Sentiment: {result['label']} ({result['confidence']:.2f})")