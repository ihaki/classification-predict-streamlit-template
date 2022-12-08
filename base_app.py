"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
import pycountry, requests

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information", "News"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown(
		"""# Tweet classifier for climate change
## Resources;

 Twitter [data](https://www.kaggle.com/competitions/edsa-sentiment-classification/data) relating to climate change collected between 2015 and 2017

### objectives

     * To use Natural language processing and machine learning models
	  to correctly classify a tweet

     * To help marketing teams to plan marketing strategies and run
	  successfull add campaigns

    *  To help marketing teams to correctly make their stance on climate
	 change get well known to clients

    *To help companies build good customer relations with clients and
	 have a returning clients""")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Run the tweet through a machine learning model. ")
		# Creating a text box for user input
		tweet_text = st.text_area("Please enter the tweet as text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))
	if selection == "News":

		st.title("Latest Climate News")
		btn = st.button("Click to get latest climate change news")

		if btn:
			url ="https://newsapi.org/v2/everything?"
			request_params = {
				"q": 'climate change OR global warming OR climate disaster',
				"sort by": "latest",
				"language": 'en',
				"apikey": "950fae5906d4465cb25932f4c5e1202c"
			}
			r = requests.get(url, request_params)
			r = r.json()
			articles = r['articles']

			for article in articles:
				st.header(article['title'])
				if article['author']:
					st.write(f"Author: {article['author']}")
				st.write(f"Source: {article['source']['name']}")
				st.write(article['description'])
				st.write(f"link to article: {article['url']}")
				st.image(article['urlToImage'])






# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
