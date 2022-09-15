import warnings
warnings.filterwarnings("ignore")

import SearchEngine as SE
search_eng = SE.SearchEngine()

user_id = 984
query = "i love you"

if __name__ == "__main__":
    res = search_eng.query(user_id=user_id, text=query, save_res=True)