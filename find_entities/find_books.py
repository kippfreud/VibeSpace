books_api_key = "AIzaSyBpj5U_PKChOFr7pSQrTfgryH-MGww_8sQ"

import urllib.request
import json
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
from gensim.parsing.preprocessing import remove_stopwords


f = open("book_5.txt", "r", encoding="utf8")
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

entity_list = [e.strip() for e in entity_list]
entity_list = [e.replace("⬥", "") for e in entity_list]
entity_list = sorted(list(set(entity_list)))

print(f"{len(sentence_list)} Sentences")
print(f"{len(entity_list)} Books")

fake = 0
real = 0
cats = 0
no_cats = 0

for e in entity_list:

    payload = {
    #    'q': 'white whale:keyes',
        'q': e.replace("|", " ").replace("(", " ").strip(),
        'key': 'AIzaSyDtjvhKOniHFwkIcz7-720bgtnubagFxS8',
        #'key': books_api_key,
        'maxResults': 20
    }

    if "|" in e:
        raw_tit = e.split("|")[0]
    else:
        raw_tit = e

    query = urllib.parse.urlencode(payload)

    url = 'https://www.googleapis.com/books/v1/volumes?' + query

    response = urllib.request.urlopen(url)
    text = response.read()
    data_first = json.loads(text)

    found = False
    for item in data_first["items"]:
        item_tit = item['volumeInfo']['title']
        shared_words = [w for w in [w.replace(":","").replace("(","").replace(")","") for w in raw_tit.split(" ")] if w in [i.replace(":","").replace("(","").replace(")","") for i in item_tit.lower().split(" ")]]
        if remove_stopwords(raw_tit) == "" or remove_stopwords(raw_tit) == "":
            pass
        else:
            shared_words = [s for s in remove_stopwords(" ".join(shared_words)).split(" ") if s!= ""]
        if len(shared_words) > 0:
            found = True
            break
    if found:
        response = urllib.request.urlopen(item["selfLink"])
        text = response.read()
        data = json.loads(text)
        print(data)
    else:
        payload = {
            #    'q': 'white whale:keyes',
            'q': raw_tit.strip(),
            'key': 'AIzaSyDtjvhKOniHFwkIcz7-720bgtnubagFxS8',
            #'key': books_api_key,
            'maxResults': 10
        }
        query = urllib.parse.urlencode(payload)

        url = 'https://www.googleapis.com/books/v1/volumes?' + query

        response = urllib.request.urlopen(url)
        text = response.read()
        data_first = json.loads(text)

        found = False
        for item in data_first["items"]:
            item_tit = item['volumeInfo']['title']
            shared_words = [w for w in
                            [w.replace(":", "").replace("(", "").replace(")", "") for w in raw_tit.split(" ")] if
                            w in [i.replace(":", "").replace("(", "").replace(")", "") for i in item_tit.lower().split(" ")]]
            if remove_stopwords(raw_tit) == "" or remove_stopwords(raw_tit) == "":
                pass
            else:
                shared_words = [s for s in remove_stopwords(" ".join(shared_words)).split(" ") if s!= ""]

            if len(shared_words) > 0:
                found = True
                break
        if found:
            response = urllib.request.urlopen(item["selfLink"])
            text = response.read()
            data = json.loads(text)
            print(data)
        else:
            print(f"{e} was unfound :(")
            fake += 1
            continue

    if "categories" in data["volumeInfo"].keys():
        categories = ", ".join(data["volumeInfo"]["categories"])
        cats += 1
    else:
        categories = "NA"
        no_cats += 1

    name = data["volumeInfo"]["title"]
    real += 1
    print(f"So far {real} Real ({cats} with genres, {no_cats} without) and {fake} Fake: {round(100 * real / (fake + real), 2)}% Accuracy so far...")
    print(f"{e} -> {name}")
    authors = ", ".join(data["volumeInfo"]["authors"])
    print(f"Authors: {authors}\nCatagories: {categories}\n\n")
