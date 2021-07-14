# Sentiment Analysis on Tweets
Fine-tuning XLNet model sentiment classification.
This work was done for a course project as hackathon this is already completed in the year 2020, I took this project just to get expereinced in ML model, This is was conducted in the time of lockdowns, remote work, and general uncertainty, by #ZindiWeekendz offer data scientists the opportunity to continue to develop their skills while contributing to practical, open-source AI solutions to help in the battle against COVID-19.

# Understand the Problem Statement

The goal of this project was to predict sentiment for the given Twitter post using Python. Sentiment analysis can predict many different emotions attached to the text, but in this report only 3 major were considered: positive, negative and neutral. The training dataset was small (just over 5900 examples) and the data within it was highly skewed, which greatly impacted on the difficulty of building good classifier. After creating a lot of custom features, utilizing both bag-of-words and word2vec representations and applying the XGBoost algorithm and XLNet, the classification accuracy at level of 96% was achieved.

`Note: The evaluation metric from this practice problem is F1-Score`

## Requirements

There are some general library requirements for the project and some which are specific to individual methods. The general requirements are as follows.  
* `numpy`
* `scikit-learn`
* `scipy`
* `nltk`

# Simple Transformers - Roberta + Bert
# Link to the competition : https://zindi.africa/hackathons/covid-19-tweet-classification-challenge


## Information about other files

* dataset 1: List of positive words.
* dataset 2: List of negative words.
* dataset 3: GloVe words vectors from StanfordNLP which match our dataset for seeding word embeddings.
* Sentiment_Analysis_Tweets.ipynb: IPython notebook used to generate plots present in report.

# Results:

Test Accuracy : 96%
