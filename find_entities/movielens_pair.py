# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import imdb
import re
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ind", type=int, required=True, help="echo the string you use here")
args = parser.parse_args()

movielens = pd.read_csv("../_old/movie_movielens.csv")

ml = [x.lower() for x in list(movielens["title"])]

# print(movielens["title"])

movielens_ids = [x for x in list(movielens["movieId"])]

assert len(ml) == len(movielens_ids)

imdb = imdb.Cinemagoer(accessSystem='http')
# f = open("movie_7.txt", "r", encoding="utf8")
# sentence_list = f.read()
# sentence_list = sentence_list.lower().split("\n")

def replace_vowels(input_string):
    vowels = "aeiouAEIOU"
    result = ''.join([char if char not in input_string else "a" for char in input_string if char not in vowels])
    return result

# entity_list = []
# for sent in sentence_list:
#     new_entities = sent.split("//")
#     if new_entities != [""]:
#         if len(new_entities) <= 11:
#             entity_list += new_entities
#         else:
#             print(f"Too many in: {sent}")
#             entity_list += new_entities[:11]

# entity_list = sorted(list(set(entity_list)))
# # entity_list = [e.replace("(tv series)", "") for e in entity_list]
# # entity_list = [e.replace("tv series", "") for e in entity_list]
# entity_list = [e.replace("ﾃｩ", "e") for e in entity_list]

# real = 0
# real_entities = {}
# fake_not_movie = 0
# fake_not_movie_entities = {}
# fake_invented = 0
# fake_invented_entities = []

with open("movie_meta_fullv8_v3.json",'r') as f:
    film_meta = json.load(f)

# film_meta = {k: v for k,v in film_meta.items() if not isinstance(v, str)}

meta_titles = {}
meta_origs = {}
for key in tqdm(film_meta):
    # print(key)
    try:
        val = film_meta[key]["title"]
    except:
        continue
    meta_titles[val] = key
    meta_origs[key] = val

    # meta_origs[i] = i
# entity_list[1000*args.ind:1000*(args.ind+1)])
for film_ind, e in enumerate(tqdm(ml[1000*args.ind:1000*(args.ind+1)])):
    if movielens_ids[(1000*args.ind)+film_ind] == 5103:
        print(e)
    # print(e)
    worked = False
    fails = 0
    while worked == False:
        try:
            #print(f"Trying {e}...")
    # print(f"searching {e}...")
            items = imdb.search_movie(e.replace("_", " ").replace("(", "").replace(")","").strip())
            #items = [i for i in items if len([w for w in e.split(" ") if w in i["title"].lower().split(" ")])>0]
            if len(items) == 0:
                items = imdb.search_movie(re.sub(r'\([^)]*\)', '', e).strip())
                items = [i for i in items if len([w for w in e.split(" ") if w in i["title"].lower().split(" ")])>0]
            if len(items) == 0:
                items = imdb.search_movie(re.sub(r'\([^)]*\)', '', e).strip())
                items = [i for i in items if len([w for w in replace_vowels(e).split(" ") if w in replace_vowels(i["title"].lower()).split(" ")])>0]
            if items == []:
                worked = True
                #film_meta[e] = "NO SUCH FILM"
                print(f"{e} is NOT A FILM!")
            found = False
            for i in items:
                # print(f"{e} is {i['title']}?")
                if "movie" in i["kind"]:
                    movie = imdb.get_movie(i.movieID)
                    # print(i.movieID)
                    if "kind" in movie.keys():
                        if "movie" in movie["kind"]:
                            # print(e, "is movie")
                            if "title" in movie.data.keys():
                                title = movie.data["title"]
                                # print(title)
                                if title in list(meta_titles.keys()):
                                    film_meta[meta_titles[title]]["ID"] = movielens_ids[(1000*args.ind)+film_ind]
                                    print(e,":",movielens_ids[(1000*args.ind)+film_ind])
                                    worked = True
                                    break
                            else:
                                print("no title")
                    #print(f"{e} not found...\n\n")
            if worked == False:
                worked = True
                print(f"Movielens thinks {e} is a film, but IMDB doesn't...")
            #film_meta[e] = ""
        #     #print(f"So far {real} Real and {fake_not_movie + fake_invented} Fake ({fake_not_movie} non-movies, {fake_invented} invented)")
        #     #print(f"{round(100 * real / (fake_not_movie + fake_invented + real), 2)}% Accuracy so far...")
        except:
            print(f"failed on {film_ind} - being throttled?")
            #film_meta[e] = "API FAIL"
            fails += 1
            if fails > 10:
                worked = True

with open(f"movie_meta_id_{args.ind}.json", "w") as outfile:
    json.dump(film_meta, outfile)


