import openai
import json
import pickle as pkl
import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("domain", help="whats the domain!?!?!?!!", required=True)
args = parser.parse_args()
openAI_key = "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"
openai.api_key = openAI_key

def make_initial_dataset(domain, num, formatting_instructions=None):
    initial_generation_prompt = f"Generate me a list of {num} {domain}s. Each item in the list should be separated by '||', and '||' should be placed at the beginning and end of the list as well. The list should not be numbered. Do not include linebreaks or quotation marks. Do not repeat {domain}s."
    if formatting_instructions is not None:
        initial_generation_prompt += f" {formatting_instructions}"
    messages = [ {"role": "system", "content":
                  f"You are a {domain} expert"}]
    messages.append(
        {"role": "user", "content": initial_generation_prompt},
    )
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
    )
    reply = chat.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    entities = [r for r in reply.split("||") if len(r)>1]
    if entities[0][0] == " ":
        entities = [r[1:-1] for r in entities]

    return entities

domain = args.domain
if domain == "movie":
    formatting_instructions = "Movies should be in the form name_of_movie (release date)."
    num = 50
    entities = make_initial_dataset(domain, num, formatting_instructions)
elif domain == "song":
    formatting_instructions = "Songs should be in the form name_of_song|artist_name(release date)."
    num = 50
    entities = make_initial_dataset(domain, num, formatting_instructions)
elif domain == "book":
    formatting_instructions = "Books should be in the form name_of_book|author_name(release date)."
    num = 50
    entities = make_initial_dataset(domain, num, formatting_instructions)
else:
    print("Unknown domain, ERRORORORORRRRR!!!")
    exit(0)

num_sim = 10
prompt = f"You are a {domain} expert system. You will be given names of {domain}s, and you will respond with a list of "+\
f"{num_sim} {domain}s which are most similar to those {domain}s . You will respond in JSON format. E.g."+\
f" if given the input '{domain} A, {domain} B, {domain} C', you will respond {{'{domain} A':"+\
f" [list of similar {domain}s ...], '{domain}s B': [list of similar {domain}s ...], '{domain}s C':"+\
f" [list of similar {domain}s ...]}}. returned {domain}s"+\
" should be in similarity order. You will return just the JSON with no surrounding text."
if formatting_instructions is not None:
    prompt += f" {formatting_instructions}"

sentences = []
searched_entities = []
cpass = 1
while len(sentences) < 50000:
    print(f"Pass {cpass} ============================")
    print(f"{len(sentences)} sentences so far ===============")
    print(f"Running {len(entities)} entities this pass ===============")

    new_entities = []
    for e in entities:
        try:
            print(e)
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
                    print(f"{e} didn't work, trying again...")

            # print(reply)
            d = json.loads(reply)
            print(d)
            for i in range(len(d.keys())):
                s = "//".join([list(d.keys())[i]] + list(d.values())[i])
                sentences.append(s)
                new_entities += [D for D in list(d.values())[i] if
                                 D not in entities and D not in searched_entities and D not in new_entities]
                if len(sentences) >= 50000:
                    break
            searched_entities.append(e)
            if len(sentences) >= 50000:
                break
        except:
            print(f"{e} didn't work, NOT trying again...")

    entities = new_entities
    random.shuffle(entities)

    with open(f'{domain}_{cpass}.txt', 'w') as f:
        for line in sentences:
            f.write(f"{line}\n")

    # open a file, where you ant to store the data
    file = open(f'{domain}_{cpass}_params.pkl', 'wb')
    # dump information to that file
    pkl.dump({
        "entities": entities,
        "searched_entities": searched_entities,
        "sentences": sentences,

    }, file)
    # close the file
    file.close()

    if entities == []:
        break

    cpass += 1