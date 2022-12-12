# @author: Minyang Wang
# @data: 9/14/2022

import warnings
import SearchEngine as SE
warnings.filterwarnings("ignore")


search_eng = SE.SearchEngine()

user_id = 907
query = "i love you"
save_res = True

if __name__ == "__main__":
    print("Searching...")
    res = search_eng.query(user_id=user_id, text=query, K=20, save_res=save_res)
    print(res)
    if save_res:
        print("Results saved to local.")
    else:
        print("Complete")
