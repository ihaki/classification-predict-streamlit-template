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
image = Image.open(r'Climate-change.jpg')
news_image = Image.open(r'news_img.jpg')
function_image = Image.open(r'function.png')
logo = Image.open(r'logo.PNG')
all_tags_img = Image.open(r'all_tags.PNG')
all_mentions_img = Image.open(r'usernames.PNG')
tags_by_label = Image.open(r'tags_by_mentions.PNG')
mentions_by_label = Image.open(r'mentions_by_label.PNG')
label_dist = Image.open(r'lable_dist.PNG')
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
	st.image(logo, width=150)


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
				the climate change debate. We aim to help companies to correctly predict 
				user's sentiments before running ad campaigns as part of their market research process
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


    - To use Natural language processing and machine learning models
	  to correctly classify a tweet

    - To help marketing teams to plan marketing strategies and run
	  successfull add campaigns

    - To help marketing teams to correctly make their stance on climate
	 change get well known to clients

    - To help companies build good customer relations with clients and
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
		tweet_text = st.text_area("Please enter the tweet as text")


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
				- General wordcloud: displays two wordclouds, a hashtags wordcloud and a mentions wordcloud
				- Hashtags by label: displays four setsof wordclouds of hashtags for each sentiment 
				- Mentions by label: displays four sets of wordclouds of mentions for each sentiment
				- Label distribution: displays  a bar graph showing the distribution of the four classes of sentiments
				"""
				)
			with right_column:
				st_lottie(data_lottie)
			st.write('##')

		#create a list of all visuals
		visuals = ['general wordcloud', 'hashtags by label', 'mentions by label', 'label distribution']

		


		st.sidebar.markdown("### Select a visual")
		visual = st.sidebar.selectbox('visuals', visuals)

		if visual == 'general wordcloud':
			with st.container():
				left_column, right_column = st.columns((1,1))
				with left_column:
					st.write("""
					Words such as: climate, climatechange, environment, actonclimate and globalwarming constitute the most popular hashtags in 
					this data. Hashtags are used to index and group tweets around a particular topic and the aforementioned hashtags would be the most appropriate tags for the climate change topic.
					""")
					
					st.image(all_tags_img, caption='all hashtags by label')
				
				with right_column:
					st.write("""The most frequently mentioned users are either politicians or
					 celebrities who have made remarks on climate change that have been met by criticism, support or both by the general public.
					
					"""
					)
					st.write('###')
					
					st.image(all_mentions_img, caption="all mentions by label")
		
		elif visual =='hashtags by label':
			st.write("""
			Trump is a popular hashtag across the labels in the climate change tweets. Trump's administration saw alot of controversial 
			moves and statements around climate change. 
			
			"""
			)
			st.write('---')
			st.image(tags_by_label)

		elif visual == "mentions by label":
			st.write("""
			Donald Trump is the most mentioned person throughout the labels. This could be because of his strong 
			opinions on climate change that are met by equally strong opposition or support by Twitter users.
			"""
			)
			st.write('---')
			st.image(mentions_by_label)

		elif visual=='label distribution':
			st.write("""
			The pro label is the most frequent category in this dataset; with 8,530 tweets labeled as 1 for supporting belief in man-made climate change, while the least; 1,296 tweets are labeled as -1, for tweets
			 that do not believe in man-made climate change
			"""
			)
			st.write('---')
			st.image(label_dist)

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
			st.subheader("Hi :wave:, we are Data cloud")
			st.write('---')
			st.title('A Market Research team focused on creating real-world solutions')
			st.write(""" \n We are passionate about the use of data to help
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
					 - Leveraging available data to analyse the market trends
					 - Creating machine learning models to analyse the data
					 - Using classification to accurately predict a user's opinion on a product
					 - Building ready to use web applications that clients can use to get a user's sentiment
					 - Deploying our web applications to make them available to a wide array of users
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

		st.write("Know more about our 'team' of predictors")
		
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
