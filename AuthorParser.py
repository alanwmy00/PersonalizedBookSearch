import re
import pickle
import unidecode

class AuthorParser:
    """
    A class for Aurthor Parser
    @author: Minyang Wang
    @date: 9/14/2022
    """

    def __init__(self, author_to_book=None) -> None:
        """
        Initialize

        Args:
            author_to_book (optional): author to book indexing dictionary. Defaults to None
        """
        self.author_to_book = {} if author_to_book is None else author_to_book
        
    def process(self, authors, book_ids, save_res=True):
        """
        populate three dictionaries

        Args:
            authors: list of all authors
            book_ids: list of unique ids
            save_res: whether writes result to local. Defaults to True
        """
        for i in range(len(book_ids)):
            id = book_ids.iloc[i]
            authors_ = authors[i].lower()
            authors_ = unidecode.unidecode(authors_) # ASCII transliterations of Unicode text

            # author to book
            for author in authors_.split(", "):
                author = re.sub(r'[^\w ]', ' ', author) # remove punctuations
                author = author.replace(" ", "") # remove space
                if author in self.author_to_book:
                    self.author_to_book[author].append(id)
                else:
                    self.author_to_book[author] = [id]
        
        # save to local
        if save_res:
            f = open("model/author_to_book.pkl", "wb")
            pickle.dump(self.author_to_book, f)
            f.close()


    def search(self, text):
        """
        Given a text, check if any author name contained in it. If yes, return all books by the authors mentioned

        Args:
            text: query text

        Returns: set of all relevant book ids
        """
        text = text.lower()
        text = unidecode.unidecode(text)
        text = re.sub(r'[^\w ]', ' ', text)
        text = text.replace(" ", "")

        # only if the input str is long enough that we consider it could be a substring of an author name
        if len(text) > 7:
            authors = list(filter(lambda x: (len(x) > 7) and (x in text or text in x), list(self.author_to_book)))
        else:
            authors = list(filter(lambda x: len(x) > 7 and x in text, list(self.author_to_book)))

        res = set()
        for author in authors:
            for j in self.author_to_book[author]:
                res.add(j)
        
        return res