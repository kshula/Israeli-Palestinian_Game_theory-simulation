import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Load CSV data from the specified file path with encoding handling
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path, encoding='ISO-8859-1')  # Changed encoding to ISO-8859-1
    return data

# Define function to analyze sentiment in Notes
def analyze_sentiment(notes):
    scores = notes.apply(lambda note: sid.polarity_scores(note)['compound'])
    return scores

# Main function for Streamlit app
def main():
    st.title('Israeli-Palestinian Conflict Analysis')
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Home", "History", "Simulation", "Conflict"])

    # Specify the file path
    file_path = "conflict.csv"
    try:
        data = load_data(file_path)
        
        if page == "Home":
            st.header("Home")
            st.write("Welcome to the Israeli-Palestinian Conflict Analysis App.")
            st.write("Use the sidebar to navigate between pages.")
        
        elif page == "History":
            st.header("History")
            st.write("Historical Data Overview")
            st.write(data)

            # Sentiment analysis on Notes
            st.subheader("Sentiment Analysis on Notes")
            data['Sentiment_Score'] = analyze_sentiment(data['Notes'])
            st.write(data[['Date', 'Event_Type', 'Notes', 'Sentiment_Score']])
            
            # Plotting sentiment scores over time using Plotly
            st.subheader("Sentiment Scores Over Time")
            fig = px.line(data, x=pd.to_datetime(data['Date']), y='Sentiment_Score', title='Sentiment Score Over Time')
            st.plotly_chart(fig)
        
        elif page == "Simulation":
            st.header("Simulation")
            st.write("Simulation Data and Analysis")
            # Placeholder for future game theory simulation integration
            st.write("This section will include game theory simulations based on the data.")
        
        elif page == "Conflict":
            st.header("Conflict Analysis")
            st.write("Analyzing Conflict Related Data")

            # Categorical analysis of Impact column
            st.subheader("Impact Analysis")
            impact_counts = data['Impact'].value_counts()
            fig = px.bar(impact_counts, x=impact_counts.index, y=impact_counts.values, labels={'x':'Impact', 'y':'Count'}, title='Impact Analysis')
            st.plotly_chart(fig)
            
            # Analyzing global reactions
            st.subheader("Global Reactions")
            global_reactions = ['Global Disappointment', 'Global Support', 'Global Criticism']
            for reaction in global_reactions:
                st.subheader(reaction)
                reaction_counts = data[data['Notes'].str.contains(reaction, na=False)]
                st.write(reaction_counts[['Date', 'Event_Type', 'Notes']])

                # Pie chart for visualizing reaction counts
                reaction_counts_pie = reaction_counts['Event_Type'].value_counts()
                fig = px.pie(reaction_counts_pie, values=reaction_counts_pie.values, names=reaction_counts_pie.index, title=f'{reaction} Distribution')
                st.plotly_chart(fig)
    except FileNotFoundError:
        st.error("The file 'conflict.csv' was not found. Please ensure the file is in the correct path and try again.")
    except UnicodeDecodeError:
        st.error("There was an error decoding the file. Please check the file encoding and try again.")

# Run the Streamlit app
if __name__ == '__main__':
    main()
