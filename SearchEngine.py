import numpy as np, pandas as pd
import Recommender as Recom
import AuthorParser as AP
from sentence_transformers import SentenceTransformer, util
import pickle
import os


class SearchEngine:
    """
    A class for Search Engine
    @author: Minyang Wang
    @date: 9/14/2022
    """

    def __init__(self) -> None:
        """
        Initialize a previously-trained Recommender System, a Sentence Similarity Calculator, previously-saved 
        helper variables, and read essential files.
        """
        self.rs = Recom.load_model("model")
        self.sen_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        with open('model/author_to_book.pkl', 'rb') as handle:
            b = pickle.load(handle)
        self.se = AP.AuthorParser(author_to_book=b)
        self.books_cleaned = pd.read_csv("goodbooks-10k/books_cleaned.csv")
        self.to_read = pd.read_csv("goodbooks-10k/to_read.csv")

    def query(self, user_id, text, to_read_boosting_factor=1.5, K=20, save_res=True):
        """
        Main function of the class, given a user_id, query, we return the top books recommended

        Args:
            user_id: user id, integer between 1 and 53424 (inclusive)
            text: query
            to_read_boosting_factor (optional): to-read boost: if a reader really wants to read books on
                his/her/their to-read list first, this value could be set larger. Defaults to 1.5
            K (optional): how many book recommendations to return. Defaults to 20
            save_res(optional): whether writes results to csv. Defaults to True
        Returns:
            top K books, with final_score and some book info, for convenience
        """
        assert int(user_id) == user_id and 0 < user_id <= 53424, "Please input a valid user-id: integer between 1 and" \
                                                                 " 53424 (inclusive)"
        assert to_read_boosting_factor > 1, "Please enter a number > 1 for to_read_boosting_factor"

        res = pd.DataFrame()
        res["user_id"] = [user_id] * int(1e4)
        res["book_id"] = self.books_cleaned["book_id"]
        res["similarity_score"] = self.calculate_word_sim(self.books_cleaned.title, text)  # calculate similarity score
        res["to_read_boost"] = self.to_read_boost(user_id, to_read_boosting_factor)  # check if in user's to_read list
        res["rating"] = self.rs.predict(np.array(res.iloc[:, :2]))  # calculate user's rating of the books
        res["author_boost"] = self.author_boost(text)  # check if input contains any authors
        res = res.merge(self.books_cleaned, left_on="book_id", right_on="book_id")  # merge to show book info
        res["in_to_read_list"] = res.to_read_boost.apply(lambda a: a > 1) # add a col to show if in to-read
        res["final_score"] = res.similarity_score * res.to_read_boost * res.rating * res.author_boost  # final score
        res = res.sort_values("final_score", ascending=False)  # sort by final score
        res = res.head(K)
        res.index = range(K)
        res.drop(columns="user_id to_read_boost similarity_score rating author_boost".split(), inplace=True)

        # Save to local
        if save_res:
            path = "saved_results/"
            if not os.path.exists(path):
                os.makedirs(path)
            filename = f"{user_id} - {text}.csv"
            res.head(K).to_csv(path + filename)

        return res

    def calculate_word_sim(self, targets, text):
        """
        Calculate word similarity score

        Args:
            targets: an N * 1 pd.df, with col = book titles
            text: query text

        Returns:
            similarity scores
        """
        if text == "":
            return [0.05] * 10_000

        embedding_1 = self.sen_model.encode(np.array(targets), convert_to_tensor=True)
        embedding_2 = self.sen_model.encode(text, convert_to_tensor=True)
        res = util.pytorch_cos_sim(embedding_1, embedding_2)
        return np.maximum(res.numpy().reshape(-1), 0.05)

    def to_read_boost(self, user_id, to_read_boost):
        """
        If a book is in user's to read list, we increase the likelihood that it appears at the top.

        Args:
            user_id (int): user id
            to_read_boost(float): boosting factor

        Returns:
            a list of len 1e4: l[i] = 1 if not in to_read, else = to_read_boost
        """
        to_read = self.to_read[self.to_read.user_id == user_id].book_id
        res = [1 for _ in range(int(1e4))]
        for i in to_read:
            res[i - 1] = to_read_boost
        return res

    def author_boost(self, text):
        """
        If the input text contains an author name, we would give it an author_boost score, since it is most likely 
        searching for authors instead of book titles.

        Args:
            text: input query

        Returns:
            author_boost scores, by default 1 if no author not in query, else 30 (we deliberately picked
            this value to make sure that books by the given author would show up in the top of the list)
        """
        res = [1 for _ in range(int(1e4))]
        idx = self.se.search(text)
        for i in idx:
            res[i - 1] = 30
        return res
