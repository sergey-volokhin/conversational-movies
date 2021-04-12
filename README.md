# Conversational Collaborative Filtering using External Data and MovieSent dataset

Code and data from the NAACL'21 short paper "You Sound Like Someone Who Watches Drama Movies: Towards Predicting Movie Preferences from Conversational Interactions" by Volokhin et al.

*MovieSent* - dataset containing 489 movie-related conversations with fine-grained user sentiment labels about each mentioned movie.
Conversations are in the [MovieSent.json](data/MovieSent.json) file.

Reviews were collected in April 2020. Initially a list of critics is compiled from more than 600 movies, their IDs are in [films_rt_ids.json](data/films_rt_ids.json). Then for those critics all their reviews are scraped and put into [reviews.tar.gz](data/reviews.tsv.gz) file. 

To run the model:

1) Install [requirements.txt](requirements.txt)
2) Run [index.py](index.py) to create an index of reviews based on the [reviews.tsv.gz](data/reviews.tsv.gz) file.
3) Run [sentiment_estimation.py](sentiment_estimation.py) to create a sentiment estimation model.
4) Run [main.py](main.py) for the final model. Training of CF model will occur at the same time, and can take a long time for a SVDpp model (KNN is much faster, ~20 seconds, if you just want to check if the code works).
