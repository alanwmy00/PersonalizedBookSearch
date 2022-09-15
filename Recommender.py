import os
import numpy as np
import pandas as pd
import surprise
from surprise import Dataset, Reader, SVD


MODEL_NAME = "Surprise_SVD"


class Recommender:
    """
    A class for Recommender
    @author: Minyang Wang
    @date: 9/14/2022
    """
    
    def __init__(self):
        """
        Initialize a SVD model
        """
        self.model = SVD()


    def fit(self, train_X, train_y):
        """
        Fitting the model
        Args:
            train_X: X
            train_y: y
        """
        reader = Reader(rating_scale=(1, 5))

        train = pd.DataFrame(train_X.copy())
        train["rating"] = train_y
        data = Dataset.load_from_df(train, reader)
        self.model.fit(data.build_full_trainset())


    def predict(self, X):
        """
        Predict, given user-item pair

        Args:
            X: pd.DataFrame with shape (N, 2); first col = user_id, second col = item_id
        """
        test = pd.DataFrame(X)
        test.columns = ['u', 'i']
        test["rating"] = 1
        test_ = Dataset.load_from_df(test, reader=Reader(rating_scale=(1, 5))).build_full_trainset()
        test_set = test_.build_testset()
        predictions = pd.DataFrame(self.model.test(test_set))
        temp2 = pd.merge(test, predictions, left_on=['u', 'i'], right_on=['uid', 'iid'], how="left")
        return np.array(temp2.est).reshape(-1, 1)


    def save(self, model_path):
        """
        Save to local
        Args:
            model_path: path to write to
        """
        surprise.dump.dump(os.path.join(model_path, "model.save"), predictions=None, algo=self.model, verbose=1)


    @classmethod
    def load(cls, model_path):
        """
        Helper for loading model
        Args:
            model_path: path to model

        Returns:
            model, with weights loaded
        """
        mf = cls()
        mf.model = surprise.dump.load(os.path.join(model_path, "model.save"))[1]
        return mf


def save_model(model, model_path):
    model.save(model_path)


def load_model(model_path):
    """
    Helper for loading model
    Args:
        model_path: path to model

    Returns:
        model
    """
    try:
        model = Recommender.load(model_path)
    except:
        raise Exception(f'''Error loading the trained {MODEL_NAME} model. 
            Do you have the right trained model in path: {model_path}?''')
    return model