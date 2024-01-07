
from tqdm import tqdm
import json
import requests
import re

def handler(signum, frame):
    raise TimeoutError("Operation timed out!")


f = open("song_7.txt", "r", encoding="utf8")
sentence_list = f.read()
sentence_list = sentence_list.lower().split("\n")
entity_list = []
for sent in sentence_list:
    new_entities = sent.split("//")
    if new_entities != [""]:
        if len(new_entities) <= 11:
            entity_list += new_entities
        else:
            print(f"Too many in: {sent}")
            entity_list += new_entities[:11]
entity_list = sorted(list(set(entity_list)))
print(f"{len(sentence_list)} Sentences")
print(f"{len(entity_list)} Songs")


# f = open('song_meta.json')
# song_meta = json.load(f)
# f.close()
# entity_list = [k for k,v in song_meta.items() if v=="API error"]
song_meta = {}

# You have to have your own unique two values for API_KEY and API_SECRET
# Obtain yours from https://www.last.fm/api/account/create for Last.fm
API_KEY = "fa0dd8874d2ac5231b78eddf76165bae"  # this is a sample key
API_SECRET = "8dbbf8e4941b4e4d38382e29b4cf1673"

#song_meta = {}
try:
    for e in tqdm(entity_list):
        SEARCH_STR = re.sub("\(.*?\)","",e)
        SEARCH_STR = SEARCH_STR.strip().replace("|", " ").replace("(", " ").replace(")","")
        SEARCH_STR = SEARCH_STR.replace("&", "and")
        x = requests.get(f'http://ws.audioscrobbler.com/2.0/?method=track.search&track={SEARCH_STR}&api_key={API_KEY}&format=json')
        if x.status_code == 200:
            met = json.loads(x.content.decode('utf8'))
            if len(met["results"]["trackmatches"]["track"]) == 0:
                print(f"{e} had no search results...")
                song_meta[e] = "No Results..."
                continue
            track = met["results"]["trackmatches"]["track"][0]
            tname = track["name"]
            tartist = track["artist"]
            x = requests.get(
                f'http://ws.audioscrobbler.com/2.0/?method=track.getTopTags&track={tname}&artist={tartist}&api_key={API_KEY}&format=json')
            # x = requests.get(
            #     f'http://ws.audioscrobbler.com/2.0/?method=track.getTags&api_key={API_KEY}&artist={tartist}&track={tname}&format=json'
            # )
            if x.status_code == 200:
                try:
                    tags = [T["name"] for T in json.loads(x.content.decode('utf8'))["toptags"]["tag"][:10]]
                except:
                    tags = []
                song_meta[e] = {
                    "track": tname,
                    "artist": tartist,
                    "tags": tags
                }
            else:
                song_meta[e] = "API error"
                continue
            x = requests.get(
                f"http://ws.audioscrobbler.com/2.0/?method=track.getsimilar&artist={tartist}&track={tname}&api_key={API_KEY}&format=json"
            )
            if x.status_code == 200:
                try:
                    simtracks = [{"track": T["name"], "artist": T["artist"]["name"]} for T in json.loads(x.content.decode('utf8'))["similartracks"]["track"][:10]]
                except:
                    simtracks = []
                song_meta[e] = {
                    "track": tname,
                    "artist": tartist,
                    "tags": tags,
                    "similar": simtracks
                }
            else:
                song_meta[e] = "API error"
            print(f"{e} --> {tname} by {tartist}: {tags}")
        else:
            song_meta[e] = "API error"
except:
    print(f"Finished {len(list(song_meta.keys()))} songs out of {len(entity_list)} then broke...")
    with open(f"song_meta.json", "w") as outfile:
        json.dump(song_meta, outfile, indent=4)

with open(f"song_meta.json", "w") as outfile:
    json.dump(song_meta, outfile, indent=4)