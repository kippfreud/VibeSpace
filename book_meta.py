import requests
import json
from tqdm import tqdm

def load_json(path):
    print(path)
    with open(path) as f:
        js = json.load(f)
    return js

def get_book_genre_from_openlibrary(isbn):
    # Open Library API endpoint for ISBN lookup
    url = f"https://openlibrary.org/api/books"

    # Parameters: use jscmd=data to get more detailed information
    params = {
        "bibkeys": f"ISBN:{isbn}",
        "format": "json",
        "jscmd": "data"
    }

    error_count = 0
    max_errors = 20

    while error_count < max_errors:
        # Send a request to the Open Library API
        response = requests.get(url, params=params)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            key = f"ISBN:{isbn}"

            # Check if the ISBN exists in the response data
            if key in data:
                book_info = data[key]

                # Check for 'subjects' field which often contains genre tags
                if "subjects" in book_info:
                    genres = [subject['name'] for subject in book_info["subjects"]]
                    return ', '.join(genres) if genres else "No genre information available."
                else:
                    return "No genre information available."
            else:
                return "Book not found."
        else:
            error_count += 1
            print(f"Error: {response.status_code} - Retrying... ({error_count}/{max_errors})")

    # If we hit the max error count, return an error message
    return f"Failed after {max_errors} attempts with error code: {response.status_code}"


book_meta = load_json("meta/book_meta.json")
reduced_book_meta = {k: v for k, v in book_meta.items() if not(v["isbn_10"] == "" and v["isbn_13"] == "")}

# for i, k in enumerate(tqdm(book_meta.keys())):
#     if "ol_genre" not in book_meta[k].keys():
#         if book_meta[k]["isbn_10"] != "":
#             genre = get_book_genre_from_openlibrary(book_meta[k]["isbn_10"])
#             if genre == "No genre information available." or genre == "book not found.":
#                 genre = None
#             else:
#                 genre = [g.strip() for g in genre.lower().split(",")]
#                 book_meta[k]["ol_genre"] = genre
#                 # print(f"{i}: {k}: \n     Genre was {book_meta[k]['subjects']} \n     Genre is now {genre}")
#             if i % 1000 == 0 and i > 0:
#                 with open("meta/book_meta_updated.json", 'w') as json_file:
#                     json.dump(book_meta, json_file, indent=4)  # indent=4 makes the file pretty-printed

for i, k in enumerate(tqdm(book_meta.keys())):
    if "ol_genre" not in book_meta[k].keys():
        book_meta[k]["ol_genre"] = []

with open("meta/book_meta.json", 'w') as json_file:
    json.dump(book_meta, json_file, indent=4)  # indent=4 makes the file pretty-printed