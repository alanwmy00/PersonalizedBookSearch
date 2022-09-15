# @author: Minyang Wang
# @data: 9/14/2022

import warnings
import SearchEngine as SE
warnings.filterwarnings("ignore")


search_eng = SE.SearchEngine()

user_id = 907
query = "i love you"

if __name__ == "__main__":
    res = search_eng.query(user_id=user_id, text=query, K=20, save_res=True)
