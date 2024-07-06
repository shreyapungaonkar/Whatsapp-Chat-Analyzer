from urlextract import URLExtract
from wordcloud import WordCloud
extract = URLExtract()
import pandas as pd
from collections import Counter
import emoji
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
#def fetch_stats(selected_user,df):

 #   if selected_user != 'Overall':
  #      df = df[df['user'] == selected_user]

def fetch_stats(selected_user, df, period=None):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    if period == 'month':
        df = df[df['date'].dt.year == pd.Timestamp.now().year]
        df = df[df['date'].dt.month == pd.Timestamp.now().month]
    elif period == 'year':
        df = df[df['date'].dt.year == pd.Timestamp.now().year]
    elif period == 'last_10_days':
        last_10_days = pd.Timestamp.now() - pd.DateOffset(days=10)
        df = df[df['date'] >= last_10_days]
    elif period == 'last_15_days':
        last_15_days = pd.Timestamp.now() - pd.DateOffset(days=15)
        df = df[df['date'] >= last_15_days]
    #1.fetch no of msg
    num_messages = df.shape[0]
    
    
    #2. no. of words
    words = []
    for message in df['message']:
        words.extend(message.split())
    
    
    #3. No. f media shared
    num_media_msg = df[df['message']=='<Media omitted>\n'].shape[0]
    
    
    #4. ftech no. of links shared
    links=[]
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_msg,len(links)
    

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts()/df.shape[0])*100,2).reset_index().rename(
        columns={'user':'name','count':'percentage'})
    return x,df


def create_wordcloud(selected_user,df):
    f = open('marathi stopwords.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)
    

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):
    f  =open('marathi stopwords.txt','r')
    stop_words = f.read()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != "gropu_notification"]
    temp = temp[temp['message']!= '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df =  pd.DataFrame(Counter(words).most_common(20))
    return most_common_df


def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))), columns=['Emoji', 'Count'])
    
    return emoji_df



def chat_sentiment(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Initialize the Sentiment Intensity Analyzer
    sentiments = SentimentIntensityAnalyzer()

    # Calculate sentiment scores for each message
    df["positive"] = [sentiments.polarity_scores(i)["pos"] for i in df["message"]]
    df["negative"] = [sentiments.polarity_scores(i)["neg"] for i in df["message"]]
    df["neutral"] = [sentiments.polarity_scores(i)["neu"] for i in df["message"]]

    # Determine overall sentiment for each message
    def overall_sentiment(row):
        if row["positive"] > row["negative"]:
            return 1.0
        elif row["negative"] > row["positive"]:
            return -1.0
        else:
            return 0.0

    df["sentiment"] = df.apply(overall_sentiment, axis=1)

    sentiment_df = df[['positive', 'negative', 'neutral', 'sentiment']]
    return sentiment_df

# Define function to determine overall sentiment
def get_overall_sentiment(sentiment_df):
    sentiment_counts = Counter(sentiment_df['sentiment'])
    overall_sentiment = {
        'Positive': sentiment_counts.get(1.0, 0),
        'Negative': sentiment_counts.get(-1.0, 0),
        'Neutral': sentiment_counts.get(0.0, 0)
    }
    return overall_sentiment