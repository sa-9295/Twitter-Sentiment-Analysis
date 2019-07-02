# Twitter-Sentiment-Analysis

Sentiment analysis (also known as opinion mining) is one of the many applications of Natural Language Processing. It is a set of methods and techniques used for extracting subjective information from text or speech, such as opinions or attitudes. In simple terms, it involves classifying a piece of text as positive, negative or neutral.

The objective of this task is to detect hate speech in tweets. For the sake of simplicity, we say a tweet contains hate speech if it has a racist or sexist sentiment associated with it. So, the task is to classify racist or sexist tweets from other tweets.

Formally, given a training sample of tweets and labels, where label ‘1’ denotes the tweet is racist/sexist and label ‘0’ denotes the tweet is not racist/sexist, the objective is to predict the labels on the given test dataset.

We started with reading the files and then merging the train and test dataset. We have done text processing - Transform each word to lower case, Remove punctuations, stopwords, numbers, hastags, URLs, Stemming. Create bag of words after all the above steps snd then visualizing it with Word cloud for most frequesnt words. 

Algorithms used:
Regularized Logistic Regression
SVM with cross validation and grid search
Random forest with cross validation and grid search
XGBoost with cross validation and grid search
