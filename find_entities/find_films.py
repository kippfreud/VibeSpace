import imdb
import re
from tqdm import tqdm
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ind", type=int, required=True, help="echo the string you use here")
args = parser.parse_args()

imdb = imdb.Cinemagoer(accessSystem='http')
f = open("movie_8.txt", "r", encoding="utf8")
sentence_list = f.read()
sentence_list = sentence_list.lower().split("\n")

f = open('movie_meta_fullv8_v2.json')
movie_meta = json.load(f)
f.close()

def replace_vowels(input_string):
    vowels = "aeiouAEIOU"
    result = ''.join([char if char not in input_string else "a" for char in input_string if char not in vowels])
    return result

entity_list = []
for sent in sentence_list:
    new_entities = sent.split("//")
    if new_entities != [""]:
        if len(new_entities) <= 11:
            entity_list += new_entities
        else:
            print(f"Too many in: {sent}")
            entity_list += new_entities[:11]

# entity_list = [e.replace("(tv series)", "") for e in entity_list]
# entity_list = [e.replace("tv series", "") for e in entity_list]
entity_list = [e.replace("ﾃｩ", "e") for e in entity_list]
entity_list = [e.strip() for e in entity_list]
entity_list = sorted(list(set(entity_list)))
print(f"{len(entity_list)} movies")
print(f"{len([e for e in [ent for ent in entity_list if ent in movie_meta.keys()] if movie_meta[e]!='API FAIL'])} with meta info")
failed_entity_list = [e for e in [ent for ent in entity_list if ent in movie_meta.keys()] if movie_meta[e]=='API FAIL'] + [ent for ent in entity_list if ent not in movie_meta.keys()]
print(f"We have {len(failed_entity_list)} entities to search...")
real = 0
real_entities = {}
fake_not_movie = 0
fake_not_movie_entities = {}
fake_invented = 0
fake_invented_entities = []

film_meta = {}

for film_ind, e in enumerate(tqdm(failed_entity_list[100*args.ind:100*(args.ind+1)])):
    worked = False
    fails = 0
    while worked == False:
        try:
            #print(f"Trying {e}...")
            SEARCH_STR = e.replace("_", " ").replace("%","").replace("'","").strip()
            try:
                items = imdb.search_movie(SEARCH_STR.replace("(", "").replace(")",""))
            except:
                items = imdb.search_movie("film " + SEARCH_STR.replace("(", "").replace(")", ""))
            #items = [i for i in items if len([w for w in e.split(" ") if w in i["title"].lower().split(" ")])>0]
            if len(items) == 0:
                items = imdb.search_movie(re.sub(r'\([^)]*\)', '', SEARCH_STR).strip())
                items = [i for i in items if len([w for w in e.split(" ") if w in i["title"].lower().split(" ")])>0]
            if len(items) == 0:
                items = imdb.search_movie(re.sub(r'\([^)]*\)', '', SEARCH_STR).strip())
                items = [i for i in items if len([w for w in replace_vowels(e).split(" ") if w in replace_vowels(i["title"].lower()).split(" ")])>0]
            found = False
            for i in items:
                if "movie" in i["kind"]:
                    movie = imdb.get_movie(i.movieID)
                    if "kind" in movie.keys():
                        if "movie" in movie["kind"]:
                            if "title" in movie.data.keys():
                                title = movie.data["title"]
                            if "genres" in movie.keys():
                                genres = movie["genres"]
                            else:
                                genres = None
                            if "director" in movie.keys():
                                directors = [d["name"] for d in movie["director"] if "name" in d.keys()]
                            else:
                                directors = []
                            if "writer" in movie.keys():
                                writers = [w["name"] for w in movie["writer"] if "name" in w.keys()]
                            else:
                                writers = []
                            if "languages" in movie.keys():
                                languages = movie["languages"]
                            else:
                                languages = None
                            if "cast" in movie.keys():
                                cast = [c["name"] for c in movie["cast"] if "name" in c.keys()]
                            else:
                                cast = None
                            #print(f"{e} -> {i}:\nGenres: {genres}\nDirectors: {directors}\nWriters: {writers}\nLanguages {languages}\n Top 5 Cast: {cast[:5]}\n\n")
                            #print(f"{e} -> {i}:\nGenres: {genres}\nDirectors: {directors}\nLanguages {languages}\nTop 5 Cast: {cast[:5] if cast is not None else []}\n\n")
                            found = True
                            real += 1
                            real_entities[e] = movie
                            break
            if found:
                film_meta[e] = {
                    "title": title,
                    "genres": genres,
                    "directors": directors,
                    "writers": writers,
                    "languages": languages,
                    "cast": cast
                }
                print(f"{SEARCH_STR} ---> {title}")
            else:
                if len(items) == 0:
                    fake_invented +=1
                    fake_invented_entities.append(e)
                    film_meta[e] = "SUSPECTED INVENTED"
                    print(f"{SEARCH_STR} was INVENTED")
                    worked = True
                else:
                    fake_not_movie += 1
                    fake_not_movie_entities[e] = items
                    film_meta[e] = "SUSPECTED NON MOVIE"
                    print(f"{SEARCH_STR} was SUSPECTED NON MOVIE")
                    worked = True
                #print(f"{e} not found...\n\n")
            worked = True

            #print(f"So far {real} Real and {fake_not_movie + fake_invented} Fake ({fake_not_movie} non-movies, {fake_invented} invented)")
            #print(f"{round(100 * real / (fake_not_movie + fake_invented + real), 2)}% Accuracy so far...")
        except:
            print(f"failed on {film_ind} - being throttled?")
            film_meta[e] = "API FAIL"
            fails += 1
            if fails > 100:
                worked = True

with open(f"f_movie_meta_{args.ind}.json", "w") as outfile:
    json.dump(film_meta, outfile)