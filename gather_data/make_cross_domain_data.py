import openai
import json
import pickle as pkl
import random
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--domain_to", help="whats the domain to map to!?!?!?!!", required=True)
parser.add_argument("--domain_from", help="whats the domain to map from!?!?!?!!", required=True)
args = parser.parse_args()
openAI_key = "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"
openai.api_key = openAI_key

domain_to = args.domain_to
domain_from = args.domain_from

if domain_to == "movie":
    formatting_instructions = "Movies should be in the form name_of_movie (release date)."
elif domain_to == "song":
    formatting_instructions = "Songs should be in the form name_of_song|artist_name(release date)."
elif domain_to == "book":
    formatting_instructions = "Books should be in the form name_of_book|author_name(release date)."
else:
    print("Unknown domain, ERRORORORORRRRR!!!")
    exit(0)

num_sim = 5
prompt = f"You are a both a {domain_to} and a {domain_from} expert system. You will be given names of {domain_from}s, and you will respond with a list of "+\
f"{num_sim} {domain_to}s which are most similar to that {domain_from}s . You will respond in JSON format. E.g."+\
f" if given the input '{domain_from} A', you will respond {{'{domain_from} A':"+\
f" [list of similar {domain_to}s ...]}}. returned {domain_to}s"+\
" should be in similarity order. You will return just the JSON with no surrounding text."
if formatting_instructions is not None:
    prompt += f" {formatting_instructions}"


f = open(f"{domain_from}_{8 if domain_from=='movie' else 7}.txt", "r", encoding="utf8")
corpus = f.read()
sentence_list = [lst.replace(" ", "_").replace("&", "and") for lst in corpus.lower().split("\n")]
sentence_list = [e for e in sentence_list if "similar" not in "".join(e)]
entity_list = []
for sent in sentence_list:
    entity_list += sent.split("//")
entity_list = sorted(list(set(entity_list)))
print(f"Corpus is {len(sentence_list)} sentences long")
print(f"There are {len(entity_list)} unique {domain_from}s...")

import random
random.shuffle(entity_list)

ret = {}
print(f"{len(ret.keys())} sentences so far ===============")
I = 0
while len(ret.keys()) < 10000:
    e = entity_list[I]
    I += 1
    try:
        worked = False
        while worked == False:
            try:
                messages = [{"role": "system", "content":
                    prompt}]
                message = e
                if message:
                    messages.append(
                        {"role": "user", "content": message},
                    )
                    chat = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo", messages=messages
                    )
                reply = chat.choices[0].message.content
                messages.append({"role": "assistant", "content": reply})
                worked = True
            except:
                print(f"{e} didn't work (api issue), trying again...")
        # print(reply)
        d = json.loads(reply)
        print(f"{e} worked: {list(d.keys())[0]} --> {d[list(d.keys())[0]]}")
        ret.update(d)
    except:
        print(f"{e} didn't work (un-parseable), not trying again...")

# Serializing json
json_object = json.dumps(ret, indent=4)
# Writing to sample.json
with open(f"Mapping_from_{domain_from}_to_{domain_to}.json", "w") as outfile:
    outfile.write(json_object)