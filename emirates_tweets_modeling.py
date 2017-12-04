import os, csv, re, glob, nltk
import argparse, textwrap
import numpy as np
import pandas as pd
import sklearn.feature_extraction.text as text
from dateutil.parser import parser
from sklearn import decomposition
from nltk import word_tokenize
from nltk.corpus import stopwords
from custom_stopword_tokens import custom_stopwords

from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import emoji
import tweepy

# Original Working Directory
owd = os.getcwd()


# Twitter App Credentials
consumer_key = "N6EHubErC6jwd5eDqDBoJ3iW1"
consumer_secret = "oOmJLOmlaEk7bR6R6KYsFguS2yeRmducUOKIWGZ8wmRuQ70nB0"
access_key = "598773124-jLVnqszY1MMYDbeD1vjBYeq6rx5O2QxCxtCm3IFM"
access_secret = "URaUh3VzdJJ6jgLejtny5U4I5uo4wlKGCpDgOwANn0ixZ"


en_stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

def write_csv(file_name):
    with open(file_name, 'w') as f:
        writer = csv.writer(f)


def get_tweets(searchQuery, lang, tweets_max=1000):
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_key, access_secret)
	api = tweepy.API(auth)
	maxTweets = tweets_max # Some arbitrary large number
	tweetsPerQry = 100  # this is the max the API permits
	fName = 'emirates_mentioned_tweets' # We'll store the tweets in a CSV file.
	sinceId = None
	max_id = -1L
	tweetCount = 0

	if not os.path.exists("data"):
		os.makedirs("data")
	os.chdir("data")
	with open("{0}_{1}_fil.CSV".format(fName, lang) , 'a') as file:
		writer = csv.writer(file)
		writer.writerow(['TWEET'])
	while tweetCount < maxTweets:
		try:
			if (max_id <= 0):
				if (not sinceId):
					new_tweets = api.search(q=searchQuery, tweet_mode='extended', lang = lang, count=tweetsPerQry)
				else:
					new_tweets = api.search(q=searchQuery, tweet_mode='extended', lang = lang, count=tweetsPerQry,
											since_id=sinceId)
			else:
				if (not sinceId):
					new_tweets = api.search(q=searchQuery, tweet_mode='extended', lang = lang, count=tweetsPerQry,
											max_id=str(max_id - 1))
				else:
					new_tweets = api.search(q=searchQuery, tweet_mode='extended', lang = lang, count=tweetsPerQry,
											max_id=str(max_id - 1),
											since_id=sinceId)
			if not new_tweets:
				print("No more tweets found")
				break
			for tweet in new_tweets:
				tweetFullText = tweet.full_text.encode("utf-8")
				tweetFilteredFullText = ''
				words = tweetFullText.split()

				for r in words:
					if not r in en_stop_words:
						if not r in emoji.UNICODE_EMOJI:
							try:
								# w = stemmer.stem(r)
								w = lemmatizer.lemmatize(r.decode('utf-8'))
							except UnicodeDecodeError:
								try:
									w = lemmatizer.lemmatize(r)
									# w = stemmer.stem(r.decode('utf-8'))
									print ("ERROR: UnicodeDecodeError " + tweetFullText)
								except UnicodeEncodeError:
									w = r
									print ("ERROR: UnicodeEncodeError " + tweetFullText)

							tweetFilteredFullText = tweetFilteredFullText + ' ' + w

				with open("{0}_{1}_fil.CSV".format(fName, lang) , 'a') as file:
					writer = csv.writer(file)
					writer.writerow([tweetFilteredFullText.encode("utf-8")])
			tweetCount += len(new_tweets)
			print("Downloaded {0} tweets".format(tweetCount))
			max_id = new_tweets[-1].id
		except tweepy.TweepError as e:
			print("some error : " + str(e))
			break
    	print ("Downloaded {0} tweets, Saved to {1}".format(tweetCount, fName))
	# Return to Original Directory
	os.chdir(owd)

# imports custom stop words list
def nltk_tokenize(text):
	tokens = word_tokenize(text)
	text = nltk.Text(tokens)
	stop_words = set(stopwords.words('english'))
	stop_words.update(custom_stopwords)
	words = [w.lower() for w in text if w.isalpha() and w.lower() not in stop_words]
	return words


def read_files(tweets_max=1000):

	get_tweets("@emirates", "en", tweets_max)
	file_path = "data/"
	#Set Data Directory
	os.chdir(file_path)

	try:
		rawcsv = glob.glob("*.CSV")[0]
		# print ("rawcsv: "+ rawcsv)
		twitter_data = pd.read_table(rawcsv)
		# print ("twitter_data: "+ twitter_data)
		filenames = twitter_data['TWEET']
		# print("filenames: " + filenames)
		print("Selecting {} files from {}...".format(len(filenames), file_path))
		return filenames

	except:
		print("please check your file path specification..." + os.getcwd())
		print (os.getcwd())


# SPECIFY VECTORIZER ALGORITHM
def select_vectorizer(vectorizer_type, req_ngram_range=[1,2]):

	# @ tfidf_std
	# @ tfidf_custom

	ngram_lengths = req_ngram_range

	if vectorizer_type == "tfidf_std":
		# Standard TFIDF Vectorizer (Content)
		vectorizer = text.TfidfVectorizer(input='content', analyzer='word', ngram_range=(ngram_lengths), stop_words='english', min_df=2)
		return vectorizer
	elif vectorizer_type == "tfidf_custom":
		# Standard TFIDF Vectorizer (Content)
		vectorizer = text.TfidfVectorizer(input='content', analyzer='word', ngram_range=(ngram_lengths), stop_words='english', min_df=2, tokenizer=nltk_tokenize)
		print("User specified custom stopwords: {} ...".format(str(custom_stopwords)[1:-1]))
		return vectorizer
	else:
		print("error in vectorizer specification...")
		pass


# topic modeling method
def topic_modeler(vectorizer_type, topic_modlr, n_topics, n_top_terms, req_ngram_range=[1,2], tweets_max=1000):

	# Reads Files to Analyze
	filenames = read_files(tweets_max)

	# Specify Number of Topics (max 25 topic), Ngram Structure, and Terms per Topic
	if n_topics > 25:
		n_topics = 25
	num_topics = n_topics
	num_top_words = n_top_terms
	ngram_lengths = req_ngram_range


	# SPECIFY VECTORIZER ALGORITHM
	vectorizer = select_vectorizer(vectorizer_type, ngram_lengths)


	# Vectorizer Results
	dtm = vectorizer.fit_transform(filenames).toarray()
	vocab = np.array(vectorizer.get_feature_names())
	print("Evaluating vocabulary...")
	print("Found {} terms in {} files...".format(dtm.shape[1], dtm.shape[0]))


	# DEFINE and BUILD MODEL
	if topic_modlr == "lda":

		#Define Topic Model: LatentDirichletAllocation (LDA)
		clf = decomposition.LatentDirichletAllocation(n_topics=num_topics+1, random_state=3)

	elif topic_modlr == "nmf":

		#Define Topic Model: Non-Negative Matrix Factorization (NMF)
		clf = decomposition.NMF(n_components=num_topics+1, random_state=3)

	elif topic_modlr == "pca":

		#Define Topic Model: Principal Components Analysis (PCA)
		clf = decomposition.PCA(n_components=num_topics+1)

	else:
		pass


	#Fit Topic Model
	doctopic = clf.fit_transform(dtm)
	topic_words = []
	for topic in clf.components_:
	    word_idx = np.argsort(topic)[::-1][0:num_top_words]
	    topic_words.append([vocab[i] for i in word_idx])


	# Show the Top Topics
	if not os.path.exists("results"):
		os.makedirs("results")
	os.chdir("results")
	results_file = "clf_results_{}_model.CSV".format(topic_modlr)
	write_csv(results_file)
	print("writing topic model results in {}...".format("/results/"+results_file))
	for t in range(len(topic_words)):
		print("Topic {}: {}".format(t, ', '.join(topic_words[t][:])))
		with open(results_file, 'a') as f:
			writer = csv.writer(f)
			writer.writerow(["Topic {}, {}".format(t, ', '.join(topic_words[t][:]))])

	# Return to Original Directory
	os.chdir(owd)




#topic_modeler("tfidf_std", "lda", 15, 10, [1,4], 5000)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Prepare input file',
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('vectorizer_type', type=str,
        help=textwrap.dedent("""\
        	Select the desired vectorizer for either text or tweet
        	@ tfidf_std      | TFIDF Vectorizer (for tweets)
        	@ tfidf_custom   | TFIDF Vectorizer with Custom Tokenizer (for tweets)

            """
            ))
    parser.add_argument('topic_modlr', type=str,
        help=textwrap.dedent("""\
        	Select the desired topic model classifier (clf)
        	@ lda     | Topic Model: LatentDirichletAllocation (LDA)
        	@ nmf     | Topic Model: Non-Negative Matrix Factorization (NMF)
        	@ pca     | Topic Model: Principal Components Analysis (PCA)

            """
            ))
    # parser.add_argument('search_query', type=str,
    #     help=textwrap.dedent("""\
    #     	Specify the word you need to search
    #     	@ @emirates | returns all tweets containing the passed word
    #
    #         """
    #         ))
    # parser.add_argument('lang', type=str,
    #     help=textwrap.dedent("""\
    #     	Select Language:
    #     	@ en | returns English tweets only
		# 	@ ar | returns Arabic tweets only
    #         """
    #         ))
    parser.add_argument('n_topics', type=int,
        help=textwrap.dedent("""\
            Select the number of topics to return (as integer)
            Note: requires n >= number of text files or tweets

            Consider the following examples:

            @ 10     | Example: Returns 5 topics
            @ 15     | Example: Returns 10 topics

            """
            ))
    parser.add_argument('n_top_terms', type=int,
        help=textwrap.dedent("""\
            Select the number of top terms to return for each topic (as integer)
            Consider the following examples:

            @ 10     | Example: Returns 10 terms for each topic
            @ 15     | Example: Returns 15 terms for each topic

            """
            ))
    parser.add_argument('req_ngram_range', nargs='+', type=int,
        help=textwrap.dedent("""\
            Select the requested 'ngram' or number of words per term
            @ NG-1:  | ngram of length 1, e.g. "pay"
            @ NG-2:  | ngram of length 2, e.g. "fair share"
            @ NG-3:  | ngram of length 3, e.g. "pay fair share"

            Consider the following ngram range examples:

            @ [1, 2] | Return ngrams of lengths 1 and 2
            @ [2, 5] | Return ngrams of lengths 2 through 5

            """
            ))
    parser.add_argument('tweets_max', type=int,
        help=textwrap.dedent("""\
			Select max number of returned tweets
			
			Consider the following max examples:
			
			@ 10000 | returns 10000 max tweets if available with the passed search query
			@ .
			
			"""
            ))

    args = parser.parse_args()
    topic_modeler(args.vectorizer_type, args.topic_modlr, args.n_topics, args.n_top_terms, args.req_ngram_range, args.tweets_max)
