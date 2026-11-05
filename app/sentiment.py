from typing import List

def analyze_sentiment(messages: List[str]) -> str:
    """
    Analyzes the sentiment of a list of messages.
    For now, it uses a simple rule-based approach.
    """
    positive_words = {"good", "great", "excellent", "happy", "thanks", "thank", "helpful", "love", "awesome", "nice"}
    negative_words = {"bad", "terrible", "poor", "unhappy", "angry", "hate", "useless", "broken", "wrong", "error", "slow", "difficult"}
    
    combined_text = " ".join(messages).lower()
    words = combined_text.split()
    
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)
    
    if pos_count > neg_count:
        return "Positive"
    elif neg_count > pos_count:
        return "Negative"
    else:
        return "Neutral"
