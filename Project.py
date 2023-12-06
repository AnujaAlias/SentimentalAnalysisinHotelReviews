# import necessary libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Define the base URL and user agent

#base_url = 'https://www.tripadvisor.in/Hotel_Review-g304554-d307116-Reviews-';
base_url= 'https://www.tripadvisor.in/Hotel_Review-g304554-d307116-Reviews-';


headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
    'Accept-Language': 'en-US, en;q=0.5'
}

# Create an empty list to store the reviews.
all_reviews = []

# Loop over the page numbers.
for page_number in range(0, 600, 10):
    # Construct the URL for the current page.
    url = base_url + 'or' + str(page_number) + '-Grand_Hyatt_Mumbai_Hotel_Residences-Mumbai_Maharashtra.html'
    # Send a request to the URL and create a BeautifulSoup object.
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all review elements and extract the text.
    reviews = []
    for review in soup.find_all('div', {'class': '_T FKffI bmUTE'}):
        reviews.append(review.text.strip())

    # Add the reviews to the list of all reviews.
    all_reviews.extend(reviews)

# Create a dictionary and a dataframe from the reviews.
data = {'Reviews': all_reviews}
df = pd.DataFrame(data)

# Save the dataframe to a CSV file.
df.to_csv('data.csv', index=False)
print('Reviews saved to grand-Hyatt-Mumbai.csv')


df = pd.DataFrame(data)
print(df)


import re
import string

# this function converts to lowercase,removes sqaure brackets, removes numbers and punctuations

def text_clean(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '' , text)
    text = re.sub('[%s]' % re.escape(string.punctuation) , '', text)
    text = re.sub('\w*\d\w*' , '' , text)
    text = re.sub('[''""]' , '' , text)
    text = re.sub('\n' , '' , text)
    return text

cleaned = lambda x : text_clean(x)


#Let's take a look at the updated text

df['Reviews'] = pd.DataFrame(df.Reviews.apply(cleaned))
df.head(10)


import nltk
nltk.download('vader_lexicon')


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
#nltk.download('vader_lexicon')

sentiments = SentimentIntensityAnalyzer()

df["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in df["Reviews"]]
df["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in df["Reviews"]]
df["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in df["Reviews"]]
df = df[["Reviews", "Positive", "Negative", "Neutral"]]
print(df.head())


x = (df["Positive"].sum())
y = (df["Negative"].sum())
z = (df["Neutral"].sum())

def sentiment_score(a, b, c):
    if (a>b) and (a>c):
        print("Positive")
    elif (b>a) and (b>c):
        print("Negative")
    else:
        print("Neutral")
sentiment_score(x, y, z)

df.to_csv('reviews.csv', index=False)
print('Reviews.csv')

import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('reviews.csv')


import pandas as pd
import numpy as np
var = np.array([x, y, z])
mylabels = ["Positive", "Negative", "Neutral"]

plt.pie(var, labels = mylabels)
plt.title("Sentiments")
plt.show()

import re
import nltk
from nltk.corpus import stopwords


# nltk.download('stopwords')
def clean(review):
    review = review.lower()
    review = re.sub('[^a-z A-Z 0-9-]+', '', review)
    review = " ".join([word for word in review.split() if word not in stopwords.words('english')])

    return review

df['Reviews'] = df['Reviews'].apply(clean)
df.head(10)


def corpus(text):
    text_list = text.split()
    return text_list


df['Review_lists'] = df['Reviews'].apply(corpus)
df.head(10)


from tqdm import trange

corpus = []
for i in trange(df.shape[0], ncols=150, nrows=10, colour='green', smoothing=0.8):
    corpus += df['Review_lists'][i]
len(corpus)

print(corpus)


from collections import Counter
mostCommon = Counter(corpus).most_common(20)
mostCommon

words = []
freq = []
for word, count in mostCommon:
    words.append(word)
    freq.append(count)


import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(x=freq, y=words)
plt.title('Top 10 Most Frequently Occuring Words')
plt.show()


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(2,2))
bigrams = cv.fit_transform(df['Reviews'])


count_values = bigrams.toarray().sum(axis=0)
ngram_freq = pd.DataFrame(sorted([(count_values[i], k) for k, i in cv.vocabulary_.items()], reverse = True))
ngram_freq.columns = ["frequency", "ngram"]


sns.barplot(x=ngram_freq['frequency'][:10], y=ngram_freq['ngram'][:10])
plt.title('Top 10 Most Frequently Occuring Bigrams')
plt.show()


cv1 = CountVectorizer(ngram_range=(3,3))
trigrams = cv1.fit_transform(df['Reviews'])
count_values = trigrams.toarray().sum(axis=0)
ngram_freq = pd.DataFrame(sorted([(count_values[i], k) for k, i in cv1.vocabulary_.items()], reverse = True))
ngram_freq.columns = ["frequency", "ngram"]


sns.barplot(x=ngram_freq['frequency'][:10], y=ngram_freq['ngram'][:10])
plt.title('Top 10 Most Frequently Occuring Trigrams')
plt.show()