# Personalized Book Search
- **Author**: Minyang Wang
- **Date**: Sept. 14, 2022

## Instructions on Testing
Create a Python virtual environment, activate it, and install the requirements. 

Go to `main.py`, modify `user_id` (line 11) and `query` (line 12) to interested. Run the script.

There are also 3 optional parameters. You may simply leave them unchanged: 
- `K`: the number of book-search results retrieving; defaults to 20; 
- `to_read_boosting_factor`: the factor we multipy the final score with when the book is on the user's to-read list (further elaborated in the `Features and Models/Existence of Author(s) in Query Text` section); defaults to 1.5;
- `save_res`: whether to write search results to `.csv` under folder `saved_results`; defaults to `True`.

You may also take a look at `Demo.ipynb/html` for reference.

## Problem Formulation
Given a user's past book ratings and future reading plan and a text for book query, what books should a search engine show to the user?

## Goals
We want to build a book search engine that returns relevant books given a piece of text; we also want to build a recommender system that predicts the user's rating of a new book, given their past book ratings. We need to combine these two features to give the user a ranked list of personalized book search results.

## Non-Goals
?

## Features and Models
- **Rating**: to make the search results personalized, we need to predict the rating of a user given a new book. \
Thus, we need to train a Recommender System. Two popular approaches for Recommender System are matrix factorization and collaborative filtering. Matrix factorization can be easily realized using a package called [Surprise](https://surpriselib.com/); collaborative filtering is commonly realized with deep learning libraries such as TensorFlow. \
I experimented with both: a Singular-Value-Decomposition (SVD) model in Surprise and a collaborative filtering model with ResNet in TF - the former one not only had a significantly shorter training time but also reached a lower MAE and a higher $r^2$ score; thus, we would use `Surprise-SVD` for user-item pair rating predictions. \
We also save the model weights to local.\
The Model can be found at `Recommender.py` and training can be found at `RecommderTraining.ipynb/html`.

- **Sentence Similarity between Query Text and Book Titles**: in the case when a user does not type the exact name of a book, we need to infer what book they might be referring to. For instance, if a user searches for "movie", most logically, we should return books with titles containing `film` as well. Thus, we compute the sentence similarity score between query text and all book titles. \
 We apply a [Hugging Face sentence transformer](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for such a purpose. As the query a user types might be completely unrelated to any book titles in our dataset, a pre-trained model such as Hugging Face would do the trick. The model would return a similarity score between -1 and 1. We instead modify the minimum score to be 0.05 - a small positive value useful for later final score calculation.  \
 Code for this part can be found in the `calculate_word_sim` method in `SearchEngine.py`. 

- **Existence of Author(s) in Query Text**: a user might type an author name instead of a book name! Most logically, even if I don't like Harry Potter, when I search for J. K. Rowling, Harry Potter should still be the top result. Therefore, we need to take the authors' names into the modeling. \
 However, for names, we can no longer adapt the sentence similarity strategy: what the transformer looks at is the semantic similarity between sentences, so it would yield low similarity scores on unrelated sentences, such as people's names. For example, say Book 1 is coauthored by Pam Beesly and Joy Nguyen, and Book 2 is written by Happy Nguyen; when we search for `Joy Nguyen`, its similarity score with `Happy Richardson` would be higher than the one with `Pam Beesly, Joy Richardson`, which doesn't help our searching.\
Thus, for authors, we created an inverted index engine for the author-book pairs. If we found the query text includes an author, or the query is a substring of an author (for instance, if we type `rowling`, we should be able to tell it refers to `J. K. Rowling`), we would give a boosting factor 30 (multiply the final score by 30). We deliberately picked this value to make sure that books by the given author would show up at the top of the list.\
Of course, cleaning on author names is required, as some authors have non-English names, and we should not require, or expect, the user to type the full name. However, such a method does not allow for typo correction, which we will discuss more in the subsequent section. \
Code for this part can be found at `AuthorParser.py` and in the `author_boost` method in `SearchEngine.py`.

- **To-Read List**: if a book is on the user's to-read list, naturally, we should move its ranking up to encourage the reader to start reading it! \
This part of modeling is rather straightforward, if a book is on one's to-read list, we give it a 1.5 boosting factor. Of course, we give the option of changing this boosting factor: a larger to-read boosting factor means the search engine really wants the user to begin reading the books on their to-read list! \
Code for this part can be found in the `to_read_boost` method in `SearchEngine.py`.
- **Ranking**: from the previous 4 parts, we get 4 values: `rating`, `sentence similarity score`, `author boosting factor`, and `to-read boosting factor`. We simply multiply these 4 values together and call the result `final score`. We can sort by `final score` in non-descending order and take the first `K` results as our outputs.

## Evaluation

## Features for Future Development

## Other Things to Try 
- jkrolling (which I unfortunately did when I tested my model)
- use of ""
- 
- f
- f
- 









