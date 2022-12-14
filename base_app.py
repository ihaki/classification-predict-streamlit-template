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
import requests
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
news_image = Image.open(r'resources\imgs\news_img.jpg')
function_image = Image.open(r'resources\imgs\function.png')
#define a function to access lottiee files

def load_lottieurl(url):
	r = requests.get(url)
	if r.status_code != 200:
		return None
	return r.json()

#load lottie urls
data_lottie = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_8gmx5ktv.json")
info_lottie = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_HhOOsG.json")
# Vectorizer
news_vectorizer = open("resources/vect_pkl.pkl","rb")
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
	options = ["Our Team", "Project Information", 'Data overview',"models","Prediction", "News"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Project Information":
		
		
		st.title("Tweet classifier for climate change")
		st.write('##')
	

		st.image(image, caption='www.freepik.com/free-photo/digital-screen-with-environment-day',
		)
		with st.container():
			left_column, right_column = st.columns((4,1))
			with left_column:
				
				st.write('---')
				st.info("General Information")
			
				st.write('##')
				st.write("""
				Use a machine model to accurately classify tweets and text as either pro, anti, neutral or news
				towards climate change. The project was done using data of tweets collected on 
				the clmate change debate. We aim to help companies to correctly predict 
				user's sentiments before running add campaigns as part of their market research process
				""")
			with right_column:
				st_lottie(info_lottie)
		st.write('#')
		st.write('---')
		# You can read a markdown file from supporting resources folder
		st.markdown(
		"""
## Resources;

 Twitter [data](https://www.kaggle.com/competitions/edsa-sentiment-classification/data) relating to climate change 
 collected between 2015 and 2017. The data was analyzed using Natural Language Peocessing.
 It was then used to train a model using Python programmimg 

 ##
 [Streamlit](https://streamlit.io/) was used to create a web application that hosts the model and project information.
 The app was then deployed to allow acces by various marketing teams.


##
---
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
		st.write('---')
		st.info("Run the tweet through a machine learning model. ")

		st.write("""
		Run the tweet through our machine learning models and get an instant prediction
		of the author's sentiment. The sentiment will be classified into either;
		- pro: believes in climate change and its effects
		- anti: does not beleive in climate change
		- neutral: does not support nor do they believe in climate change
		- news: the tweet is a news article
		"""
		)
		models = st.radio(
    "Please select a model",
    ('SVC linear', 'Support Vector Classifier', 'Random Forest Classifier'))
		# Creating a text box for user input
		st.write('##')
		st.write('---')
		tweet_text = st.text_area("Please enter the tweet as text","Type Here")


		if models == 'SVC linear':

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/lsvc_pkl.pkl"),"rb"))
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

		elif models == 'Support Vector Classifier':
			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/svc_pkl.pkl"),"rb"))
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
		elif models == 'Random Forest Classifier':
			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/rf_pkl.pkl"),"rb"))
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
		st.write('---')
		
		st.write("""Stay in the know on matters climate change with our collection of 
		news articles focusing on climate change from around the world. Give your
		marketing team and edge with the latest factual news. 
		Know about climate change events happening around the globe with a single click

		"""
		)
		st.write('##')
		st.image(news_image, width=600, caption="Image by https://www.freepik.com/free-photo/newspaper-background-concept_29016059.htm#query=news&position=10&from_view=search&track=sph")
		st.write('---')
		st.write("""
		Click the button below to to get a round up of the latest news in chimate change and global warming from the web.
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
			plt.suptitle("Climate Change Hashtags by label", fontsize = 100)
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
     <input type="text" name="message" placeholder = "enter a message" required>
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
	if selection == 'models':
		st.title('Models')

		st.write("Let us know more about our team of predictors")
		
		st.write('---')
		with st.container():
			left_column, right_column = st.columns((3,1))
			with left_column:
				st.write("""
				A few models were used to try and come up with the best solution. Some of the models are discussed in this section,\n
				They include;
				- Support Vector classifier (SVC)
				- Linear SVC
				- Random Forest Classifier
				""")
			
			with right_column:
				st.image(function_image)
		
		st.write('---')

		st.subheader('Support vector Classifier')
		
		st.write("""
		A class of support vector machines that is used for classification problems. It works best with problems
		that have well defined 'borders' betweenn the target variable classes. It performed fairly well for this project.
		Learn more about Support Vector Classifiers [here](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
		"""
		)
		st.write('---')

		st.subheader('Linear SVC')

		st.write("""
		Our best performimg model. It is a type of support vector classifier model that uses liblinear to scale features.\n It takes both linear and sparse inputs
		and therefore it is effective on outputs from countvectorizer and tfidf vectorizer. It also allows flexibility on the loss function and penalties.
		Learn more about Linear Support Vector Classifiers [here](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
		"""
		)
		
		st.write('---')

		st.subheader('Random Forest Classifier')

		st.write("""
		An ensemble method made up of decision tress. It  is a non-parametric model hence
		it does not make assumptions about the data. This allows it to take any functional form
		from the data. It is also not very susceptible to overfitting. Learn more about
		Random Forest Classifiers [Here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
		
		""")

		st.write('---')
		st.subheader('Conclusion')
		st.write("""
		All the discussed model were developed and evaluated. The best performing model was the linear SVC model
		dathering an f1-score of 0.74. The other models are also available for the user to try out on the predictions page

		""")



# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
