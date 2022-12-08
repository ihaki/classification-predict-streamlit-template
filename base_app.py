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
import matplotlib.pyplot as plt

# Data dependencies
import pandas as pd
from wordcloud import WordCloud
import advertools as adv
import string


# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")


# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	st.set_page_config(layout='wide')
	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information", 'Data overview',"News"]
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

	if selection == 'Data overview':
		
		st.title('A visual description of the dataset')

		#create a list of all visuals
		visuals = ['general worcloud', 'wordclouds by hashtags', 'wordclouds by sentiments']

		# create subsets of the data

		news_data = raw[raw['sentiment'] == 2]
		pro_data = pd.DataFrame(raw[raw['sentiment'] == 1])
		neutral_data = raw[raw['sentiment'] == 0]
		anti_data = raw[raw['sentiment'] == 1]

		# define a function to extract hashtags

		def hashtag_extractor(data):

			"""returns all the hashtags from a dataset
			arguments:
				a column of the DataFrame
			returns:
				a sequence of strings(hashtags) seperated by whitespace
			"""

			hashtag_summary = adv.extract_hashtags(data)
			hashtags = hashtag_summary['hashtags_flat']
			tags = (" ").join(hashtags)

			return tags 

		#define a function to extract mentions

		def mentions_extractor(data):

			mentions_summary = adv.extract_mentions(data)
			mentions = mentions_summary['mentions_flat']
			usernames = (" ").join(mentions)

			return usernames

		# define a functions that returns  a wordcloud

		def wordcloud_visualizer(extracted_entity, color):

			"""
			created  a wordcloud of the provided entity
			arguments:
				extracted_entity: a list of strings 
				colour: the colour argument of the wordcloud module
			returns:
				a wordcloud visual of the extracted entity
			"""
			wordcloud = WordCloud(collocations = False, colormap = color, background_color = 'white').generate(extracted_entity)
			
			return wordcloud

		# extract the hashtags of the different sentiments

		all_tags = hashtag_extractor(raw['message'])
		news_tags = hashtag_extractor(news_data['message'])
		pro_tags = hashtag_extractor(pro_data['message'])
		neutral_tags = hashtag_extractor(neutral_data['message'])
		anti_tags = hashtag_extractor(anti_data['message'])

		#extract mentions by sentiment

		all_mentions = mentions_extractor(raw['message'])
		news_mentions = mentions_extractor(news_data['message'])
		pro_mentions = mentions_extractor(pro_data['message'])
		neutral_mentions = mentions_extractor(neutral_data['message'])
		anti_mentions = mentions_extractor(anti_data['message'])


		# start working on the visuals

		st.sidebar.markdown("### Select a visual")
		visual = st.sidebar.selectbox('visuals', visuals)

		if visual == 'general wordcloud':
			gen_wordcloud_fig, axarr = plt.subplots(0,2, figsize = (12,8))
			axarr[0,0].imshow(wordcloud_visualizer(all_tags, 'brg'))
			axarr[0,1].imshow(wordcloud_visualizer(all_mentions, 'brg'))

			for ax in gen_wordcloud_fig:
				plt.sca(ax)
				plt.axis('off')

			plt.axarr[0,0].set_title('All hashtags\n', fontsize = 50)
			plt.axarr[0,1].set_title('All mentions\n', fontsize = 50)
			plt.suptitle("General tags and mentions")
			plt.tight_layout()
			st.pyplot(gen_wordcloud_fig)
		
		elif visual == 'wordclouds by hashtags':
			hash_wordcloud_fig, axarr = plt.subplots(2,2, figsize=(35,25))
			axarr[0,0].imshow(wordcloud_visualizer(news_tags, 'summer'), interpolation="bilinear")
			axarr[0,1].imshow(wordcloud_visualizer(pro_tags, 'Blues'), interpolation="bilinear")
			axarr[1,0].imshow(wordcloud_visualizer(neutral_tags, 'Wistia'), interpolation="bilinear")
			axarr[1,1].imshow(wordcloud_visualizer(anti_tags, 'gist_gray'), interpolation="bilinear")

			# Remove the ticks on the x and y axarres
			for ax in hash_wordcloud_fig.axes:
				plt.sca(ax)
				plt.axis('off')

			axarr[0,0].set_title('News label hashtags\n', fontsize=50)
			axarr[0,1].set_title('Pro climate change hashtags\n', fontsize=50)
			axarr[1,0].set_title('Neutral label hashtags\n', fontsize=50)
			axarr[1,1].set_title('Anti climate change hashtags\n', fontsize=50)
			plt.suptitle("Climate Change Hashtags by Label", fontsize = 100)
			plt.tight_layout()
			st.pyplot(hash_wordcloud_fig)

		else:
			men_wordcloud_fig, axarr = plt.subplots(2,2, figsize=(35,25))
			axarr[0,0].imshow(wordcloud_visualizer(news_mentions, 'summer'), interpolation="bilinear")
			axarr[0,1].imshow(wordcloud_visualizer(pro_mentions, 'Blues'), interpolation="bilinear")
			axarr[1,0].imshow(wordcloud_visualizer(neutral_mentions, 'Wistia'), interpolation="bilinear")
			axarr[1,1].imshow(wordcloud_visualizer(anti_mentions, 'gist_gray'), interpolation="bilinear")

			# Remove the ticks on the x and y axarres
			for ax in men_wordcloud_fig.axes:
				plt.sca(ax)
				plt.axis('off')

			axarr[0,0].set_title('News label mentions\n', fontsize=50)
			axarr[0,1].set_title('Pro climate change mentions\n', fontsize=50)
			axarr[1,0].set_title('Neutral label mentions\n', fontsize=50)
			axarr[1,1].set_title('Anti climate change mentions\n', fontsize=50)
			plt.suptitle("Climate Change mentions by Label", fontsize = 100)
			plt.tight_layout()
			st.pyplot(men_wordcloud_fig)
			

		
		


		
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
