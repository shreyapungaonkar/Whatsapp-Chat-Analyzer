import streamlit as st
import preprocessor,helper
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
st.sidebar.title("Whatsapp Chat analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    st.dataframe(df)

    user_list = df['user'].unique().tolist()
    user_list.remove("group_notification")
    user_list.sort()
    user_list.insert(0,"Overall")

    selected_user = st.sidebar.selectbox("Show analysis with",user_list)


    period = st.sidebar.radio("Select time frame", ['Overall', 'Month', 'Year', 'Last 10 Days', 'Last 15 Days'])

    if st.sidebar.button("Show Analysis"):

        if period == 'Month':
            num_messages, words, num_media_msg, links = helper.fetch_stats(selected_user, df, 'month')
        elif period == 'Year':
            num_messages, words, num_media_msg, links = helper.fetch_stats(selected_user, df, 'year')
        elif period == 'Last 10 Days':
            num_messages, words, num_media_msg, links = helper.fetch_stats(selected_user, df, 'last_10_days')
        elif period == 'Last 15 Days':
            num_messages, words, num_media_msg, links = helper.fetch_stats(selected_user, df, 'last_15_days')
        else:
            num_messages, words, num_media_msg, links = helper.fetch_stats(selected_user, df)

        col1, col2, col3, col4 = st.columns(4)



    #if st.sidebar.button("Show Analysis"):

        #num_messages, words, num_media_msg, links = helper.fetch_stats(selected_user,df)
        #col1,col2,col3,col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)

        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_msg)
        with col4:
            st.header("Links Shared")
            st.title(links)


        #finding busiest members of group
        if selected_user =='Overall':
            st.title("Most Busy Users")
            x,new_df= helper.most_busy_users(df)
            fig ,ax = plt.subplots()
            col1,col2 = st.columns(2)

            with col1:
                ax.bar(x.index,x.values,color = 'green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        #word colud

        df_wc  = helper.create_wordcloud(selected_user,df)
        fig,ax =plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)


        #most common words
        most_common_df = helper.most_common_words(selected_user,df)

        fig,ax = plt.subplots()
        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')
        st.title("Most Common Words")
        st.pyplot(fig)
        

        # Emoji analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")

        # Check if the DataFrame is empty
        if emoji_df.empty:
            st.write("No emojis found in the selected chat.")
        else:
            st.dataframe(emoji_df)


        # Sentiment analysis
        
        sentiment_df = helper.chat_sentiment(selected_user, df)
        st.title("Sentiment Analysis")
        st.dataframe(sentiment_df)


        # Overall sentiment
        overall_sentiment = helper.get_overall_sentiment(sentiment_df)
        st.title("Overall Sentiment")
         # Convert overall sentiment to DataFrame
        overall_sentiment_df = pd.DataFrame(list(overall_sentiment.items()), columns=['Sentiment', 'Count'])
        st.table(overall_sentiment_df)

        # Sentiment distribution
        sentiment_counts = Counter(sentiment_df['sentiment'])
        sentiment_labels = {1.0: 'Positive', -1.0: 'Negative', 0.0: 'Neutral'}
        sentiment_counts = {sentiment_labels[k]: v for k, v in sentiment_counts.items()}

        st.title("Sentiment Distribution")
        fig, ax = plt.subplots()
        ax.bar(sentiment_counts.keys(), sentiment_counts.values())
        st.pyplot(fig)


        
        