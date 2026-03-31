from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from textblob import TextBlob

# i put 6 sentences for each emotion so its balanced
train_data = [
    "I am very happy today", "I feel great",
    "I am feeling joyful", "I am excited",
    "I love this", "This is amazing",
    "I feel very sad", "I am feeling lonely",
    "I am depressed", "I feel down",
    "I am unhappy", "I feel like crying",
    "I am very angry", "I am feeling angry right now",
    "I am furious", "This makes me angry",
    "I am irritated", "I am annoyed",
    "I feel stressed", "I am under pressure",
    "I feel overwhelmed", "I am anxious",
    "I am worried", "I feel tense"
]

# 6 happy, 6 sad, 6 anger, 6 stress
trin_lbl = ["happy","happy","happy","happy","happy","happy", "sad","sad","sad","sad","sad","sad", "anger","anger","anger","anger","anger","anger", "stress","stress","stress","stress","stress","stress"]

# using bigrams this time for better accuracy
vec = TfidfVectorizer(ngram_range=(1,2))
x_train = vec.fit_transform(train_data)

mdl = LogisticRegression(max_iter=200)
mdl.fit(x_train, trin_lbl)

# keep asking user until they type exit
while True:
    text = input("\nEnter your text (type 'exit' to stop): ")
    if text.lower() == "exit":
        break

    x_test = vec.transform([text])
    result = mdl.predict(x_test)
    emotion = result[0]

    # textblob for sentiment
    b = TextBlob(text)
    p = b.sentiment.polarity
    sntimnt = ""
    if p > 0:
        sntimnt = "Positive"
    elif p < 0:
        sntimnt = "Negative"
    else:
        sntimnt = "Neutral"

    # giving suggestion
    sug = ""
    if emotion == "happy":
        sug = "Keep smiling and enjoy your day "
    elif emotion == "sad":
        sug = "Talk to someone you trust "
    elif emotion == "anger":
        sug = "Take deep breaths and calm down "
    elif emotion == "stress":
        sug = "Take a short break and relax "
    else:
        sug = "Stay positive!"

    print("\n----- RESULT -----")
    print("Your Text:", text)
    print("Detected Emotion:", emotion)
    print("Sentiment:", sntimnt)
    print("Suggestion:", sug)
