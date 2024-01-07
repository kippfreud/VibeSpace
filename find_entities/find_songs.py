import spotipy
from _old import spot_cred
from spotipy.oauth2 import SpotifyOAuth
from tqdm import tqdm
import json
import signal
import requests
from spotipy import SpotifyException
import time

def handler(signum, frame):
    raise TimeoutError("Operation timed out!")

#
# birdy_uri = 'spotify:artist:2WX2uTcsvV5OnS0inACecP'
#
# results = sp.artist_albums(birdy_uri, album_type='album')
# albums = results['items']
# while results['next']:
#     results = sp.next(results)
#     albums.extend(results['items'])
#
# for album in albums:
#     print(album['name'])

# Opening JSON file
f = open('song_meta.json')
meta_dict = json.load(f)
f.close()

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

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=spot_cred.client_ID,
                                               client_secret=spot_cred.client_SECRET,
                                               redirect_uri=spot_cred.redirect_url),
                     retries=0,
                     status_retries=0)
# sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id='7eda05abfe8b45c587cf619c63edd074',
#                                                client_secret='075130841b2840a79c665d32d6b77476',
#                                                redirect_uri='http://localhost/'))
# sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id='300da9568dda4289b30557e901483dcf',
#                                                client_secret='364de393f9b54485a26b8911e7ce0a2d',
#                                                redirect_uri='http://localhost/'))

def get_meta(e):
    r = sp.search(e)
    track = r["tracks"]["items"][0]
    name = track['name']
    genres = sp.album(track["album"]["uri"])["genres"]
    if genres == []:
        for art in track["artists"]:
            genres = sp.artist(art["uri"])["genres"]
            if genres != []:
                break
    genres_str = ", ".join(genres)
    artists = ", ".join([t["name"] for t in track["artists"]])
    album = track["album"]['name']
    ret = {
        "track_name": name,
        "genres": genres,
        "artists": [t["name"] for t in track["artists"]]
    }
    return ret

real = 0
fake = 0
throttled = False


for i, e in enumerate(tqdm(entity_list)):
    #time.sleep(1)
    if throttled:
        break
    if i>0 and i%1000==0:
        print(f"Got meta info for {len(list(meta_dict.keys()))} out of {len(entity_list)} songs")
        with open("song_meta.json", "w") as outfile:
            json.dump(meta_dict, outfile)
    meta_dict[e] = get_meta(e)
    try:
        meta_dict[e] = get_meta(e)
        real += 1
        # print(f"So far {real} Real and {fake} Fake: {round(100 * real / (fake + real), 2)}% Accuracy so far...")
        # print(f"{e} -> {track['name']}\nArtists: {artists}\nGenres: {genres_str}\nAlbum: {album}\nRelease Date: { track['album']['release_date']}\n\n")
    except requests.exceptions.ConnectionError:
        print("The operation took too long to complete. Trying again...")
        time.sleep(30)
        try:
            signal.alarm(30)
            meta_dict[e] = get_meta(e)
            real += 1
        except requests.exceptions.ConnectionError:
            throttled = True
            print("Didn't work, being throttled? exiting..")
            break
        except:
            print("Didn't work type 109, being throttled? exiting..")
            break
    except SpotifyException as err:
        if err.http_status == 429:
            print("429 error: leaving...")
            time.sleep(30)
            try:
                signal.alarm(10)
                meta_dict[e] = get_meta(e)
            except SpotifyException as err:
                print("Waited 30s and got another error :(, exiting...")
                break
            except:
                print("Waited 30s and got another error type 324 :(, exiting...")
                break
        else:
            print(f"Spotify exception with unknown error code for {e} exiting...")
            print(err)
            break
    except:
        print(f"{e} failed")
        fake += 1
        meta_dict[e] = False

print("FINISHED!")
print(f"Got meta info for {len(list(meta_dict.keys()))} out of {len(entity_list)} songs")
with open("song_meta.json", "w") as outfile:
    json.dump(meta_dict , outfile, indent=4)
