from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from langdetect import detect, LangDetectException
import functools
import logging
from retrying import retry

# Initialize logger
logging.basicConfig(level=logging.INFO)


# Load pre-trained model
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


# Retry decorator for transient errors
def retry_if_exception(exception):
    return isinstance(exception, RuntimeError)


# Caching decorator
@retry(retry_on_exception=retry_if_exception, stop_max_attemp_number=3)
@functools.lru_cache(maxsize=1000)
def analyze_sentiment(news):
    try:
        result = nlp(news)
        return result[0]["score"], result[0]["label"]
    except RuntimeError as e:
        logging.error("Runtime error in sentiment analysis: {e}")
        raise e
    except ValueError as e:
        logging.error("Value error in data processing: {e}")
        return 0, "error"
    except Exception as e:
        logging.error("Unexpected error in sentiment analysis: {e}")
        return 0, "error"


def estimate_sentiment(news_list):
    sentiments = []
    for news in news_list:
        if not news:
            sentiments.append((0, "neutral"))
            continue

        try:
            # Language detection
            lang = detect(news)
        except LangDetectException as e:
            logging.error("Error detecting language: {e}")
            sentiments.append((0, "error"))
            continue

        if lang != 'en':
            logging.info("Non-English news detected: {news}")
            sentiments.append((0, "non-English"))
            continue

        try:
            sentiment = analyze_sentiment(news)
            sentiments.append(sentiment)
        except RuntimeError as e:
            logging.error("Error processing news item: {e}")
            sentiments.append((0, "error"))

    return sentiments
