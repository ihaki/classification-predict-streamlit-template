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
import streamlit_lottie
from streamlit_lottie import st_lottie
from PIL import Image


#load required images
image = Image.open('resources\imgs\Climate-change.jpg')

#define a function to access lottiee files

def load_lottieurl(url):
	r = requests.get(url)
	if r.status_code != 200:
		return None
	return r.json()

#load lottie urls
data_lottie = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_8gmx5ktv.json")

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


	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Our Team", "Prediction", "Project Information", 'Data overview',"News"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Project Information":
		st.title("Tweet Classifer")
		st.subheader("Climate change tweet classification")
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
		st.title("Tweet Classifer")
		st.subheader("Climate change tweet classification")
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
			prediction_map = {
				2:'News',
				1:'Pro',
				0: 'Neutral',
				-1 : 'Anti'


			}
			st.success("Text Categorized as: {}".format(prediction_map[int(prediction)]))
	if selection == "News":

		st.title("Get The Latest Climate News")
		st.write("""
		click the button below to to get a round up of the latest news in chimate change and global warming from the web.
		 You can proceed to the news source by clicking the provided link to the article
		""")
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
		st.write('---')
		st.write('###')
		with st.container():
			left_column, right_column = st.columns((2,1))
			with left_column:
				st.write("""
				Get a visual representation of the dataset in form of wordclouds and bar chart. 
				Select an option to be displayed from the side bar. the options are;
				- general wordcloud: displays two wordclouds, a hashtags wordcloud and a mentions wordcloud
				- wordclouds by sentiment: displays four sets of wordclouds of mentions for each sentiment
				- wordclouds by hashtags: displays four setsof wordclouds of hashtags for each sentiment  
				"""
				)
			with right_column:
				st_lottie(data_lottie)
			st.write('##')

		#create a list of all visuals
		visuals = ['general wordcloud', 'wordclouds by hashtags', 'wordclouds by mentions']

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
			gen_wordcloud_fig, axarr = plt.subplots(2,3, figsize = (35,25))
			axarr[0,0].imshow(wordcloud_visualizer(all_tags, 'brg'))
			axarr[0,1].imshow(wordcloud_visualizer(all_mentions, 'brg'))

			for ax in gen_wordcloud_fig.axes:
				plt.sca(ax)
				plt.axis('off')

			axarr[0,0].set_title('All hashtags\n', fontsize = 45)
			axarr[0,1].set_title('All mentions\n', fontsize = 45)
			plt.suptitle("General tags and mentions", fontsize=100)
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

		elif visual == "wordclouds by mentions":
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
	if selection == 'Our Team':
		
		
		#define a function to access lottiee files

		def load_lottieurl(url):
			r = requests.get(url)
			if r.status_code != 200:
				return None
			return r.json()
		
		lottie_coding = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_dsxct2el.json")
		phonecall_lottie = load_lottieurl("https://assets2.lottiefiles.com/private_files/lf30_rvyzng8q.json")
		# header section
		with st.container():
			st.subheader("Hi, we are Team CW-4 :wave: ")
			st.write('---')
			st.title('A Market Research team based in the cloud ')
			st.write(""" \n we are passionate about the use of data to help
			companies to make informed decisions about  marketing strategies""")
		
		#what do we do?
		with st.container():
			st.write('---')
			left_column, right_column = st.columns(2)

			with left_column:
				st.header("What do we do?")
				st.write('##')
				st.write(
					"""
					We create viable market solutions to clients to increase their reach while reducing 
					marketing costs by:
					 - leveraging available data to analyse the market trends
					 - creating machine learning models to analyse the data
					 - using classification to accurately predict a user's opinion on a product
					 - building ready to use web applications that clients can use to get a user's sentiment
					 - deploying our web applications to make them available to a wide array of users
					If this sounds interesting, visit our predictions page to try out one of our models
					    """
				)
			with right_column:
				st_lottie(lottie_coding, height = 300, key = "coding" )
			
		# This project
		with st.container():
			st.write("---")
			st.header("Projects")
			st.write("##")

			image_column, text_column = st.columns((1,2))

			with image_column:
			# import the image
				st.image(image)
			with text_column:
				st.subheader("Get climate sentiments from tweets on a  streamlit web application")
				st.write(
					"""
					Input a tweet as text and get an instant reply of the writers sentiment on climate change.
					The sentiments are in four classes which are;
					- pro: fully supports climate change and would want to see actions to reduce carbon emissions
					- anti: does not believe in climate change
					- neutral: not anti climate chnage and does not support climate change either
					- news: the tweet is a news source 
				""")
			
			with st.container():
				st.write('---')
				st.header("Get In Touch With Us")
				st.write("##")
				contact_form = """
				<form action="https://formsubmit.co/andrewpharisihaki@gmail.com" method="POST">
     <input type="text" name="name" placeholder = "enter your name" required>
     <input type="email" name="email" placeholder = "enter your email" required>
     <button type="submit">Send</button>
</form>
					"""
				info_column, phonecall_column = st.columns((2,1))

				with info_column:
					st.markdown(contact_form, unsafe_allow_html=True)
				
				with phonecall_column:
					st_lottie(phonecall_lottie)

				#styling the contact form
			def locall_css(filename):
				with open(filename) as f:
					st.markdown(f"<style>{f.read()}</style", unsafe_allow_html=True)
			locall_css("style/style.css")


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
