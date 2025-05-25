from FlagEmbedding import BGEM3FlagModel # type: ignore
import json
import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
USED_FEATURES = ["card_faces" , "cmc", "color_identity", "defense", "edhrec_rank", "game_changer",
                   "keywords",  "life_modifier", "loyalty", "mana_cost", "name", "oracle_text",
                   "power", "produced_mana", "toughness", "type_line", "rulings"]

#Downloads or returns the cached Sentence transformer model
def download_model(model_name):
    os.makedirs('local_models', exist_ok=True)
    local_model_path = os.path.join(os.getcwd(), 'local_models', model_name)

    try:
        model = SentenceTransformer(model_name, cache_folder=local_model_path)
        print(f"Cached and or loaded'{model_name}' from {local_model_path}\n")
        return model
    except Exception as e:
        print(f"An error occurred during model download or loading: {e}")

def get_embeddings(redo_embeddings=False, saved_embeddings="Embedded_Magic_cards.pkl", model="paraphrase-MiniLM-L6-v2"):
    if redo_embeddings:
        print(f"Building data frame and doing embeddings")
        with open("Data/oracle-cards-.json", "r") as file:
            data = json.load(file)
        with open("Data/rulings-.json", "r") as file:
            rulings = json.load(file)

        oracle_id_loc = {}
        for i in range(len(data)):
            oracle_id_loc[data[i]["oracle_id"]] = i

        for r in rulings:
            if 'rulings' not in data[oracle_id_loc[r["oracle_id"]]]:
                data[oracle_id_loc[r["oracle_id"]]]['rulings'] = r['comment']
        # Process the data
        df = pd.DataFrame(data, columns=USED_FEATURES)
        df = df[df['type_line'] != "Card // Card"]
        df = df.fillna('')
        df.set_index('name', inplace=True)
        # Create Text card object to encode as sentences
        text_cards = gen_text_cards(df)

        embedding_model = download_model(model)

        # Encode the text cards to get embeddings
        card_embeddings = embedding_model.encode(text_cards)

        print(card_embeddings)
        print(df.shape, card_embeddings.shape)
        print(card_embeddings[-1])

        df['embeddings'] = card_embeddings.tolist()
        df.to_pickle(saved_embeddings)
        print(f"Build data frame and saved to {saved_embeddings}")
    else:
        print(f"Dataframe read from {saved_embeddings}")
        df = pd.read_pickle(saved_embeddings)

    return df

def gen_text_cards(df, summarize=True):
    samples = []
    for card_name, row in df.iterrows():
        # row_strings = [f"{card_name};; "]
        row_strings = [f""]
        for column in df.columns:
            if column != 'name' or column != card_name:
                value = row[column]
                # if column == 'oracle_text':
                #    #Replace the name on the card with [CARD_NAME]
                #    if np.char.find(value, card_name) != -1:
                #        value = np.char.replace(value, card_name, '[CARD_NAME]')

                #    name_split = re.split(' |,', card_name)
                #    if np.char.find(value, name_split[0]) != -1:
                #        value = np.char.replace(value, card_name[0], '[CARD_NAME]')
                if value == '':
                    continue
                row_strings.append(f"{column}; {value}")
        # Combine into single string
        # sample = f"{card_name}; " + ", ".join(row_strings)
        sample = f"".join(row_strings)
        samples.append(sample)

    return samples

# Better to compute as a matrix instead
def single_soft_cosine(vAt, vBt):
    vA = vAt.reshape(1, -1)
    vB = vBt.reshape(1, -1)
    s = cosine_similarity(vA,vB)[0]
    neu = np.dot(vA, vB.reshape(-1,1)) * s
    d1 = np.sqrt(np.dot(vA, vA.reshape(-1,1))*s)
    d2 = np.sqrt(np.dot(vB, vB.reshape(-1,1))*s)
    return neu/np.dot(d1, d2.reshape(-1,1))

def soft_cosine(target, all_embeddings):
    return [single_soft_cosine(target, e) for e in all_embeddings]

def cosine_similarity_search(df, card_name, all_embeddings, soft = True):
    if card_name not in df.index:
        print(f"{card_name} is not a card present within dataframe")
        return None
    else:
        target_embedding = np.array(df.loc[card_name, "embeddings"]).reshape(1, -1)
        if not soft:
            similarities = cosine_similarity(target_embedding, all_embeddings)[0]
        else:
            similarities = soft_cosine(target_embedding, all_embeddings)

        similarities_df = pd.DataFrame({'card_name': df.index, 'similarity': similarities})
        similarities_df = similarities_df[similarities_df['card_name'] != card_name]
        return similarities_df.sort_values(by='similarity', ascending=False)

def perform_search(df, search_vect="", n=10, all_embeddings = None, contains = ""):
    if all_embeddings is None:
        all_embeddings = np.stack(df["embeddings"].values)

    cos_search_results = cosine_similarity_search(df, search_vect, all_embeddings)
    if cos_search_results is not None:
        print(f"Top {n} most similar cards to search")
        print("=" * 60, "\n")
        #print(cos_search_results.head(n))
        #for k in cos_search_results.head(n)["card_name"]:
        i = 0
        j = 0
        while j < n and i < len(cos_search_results)-1:
            card_name = cos_search_results["card_name"].iloc[i]
            if contains == "" or contains in df.loc[card_name]["oracle_text"]:
                #Python 11 doesn't support these being nested within f strings
                mana_cost = df.loc[card_name]["mana_cost"]
                edhrec_rank = df.loc[card_name]["edhrec_rank"]
                type_line = df.loc[card_name]["type_line"]
                rulings = df.loc[card_name]["rulings"]
                print(f"{card_name}:  {mana_cost}  {type_line}  edhrec_rank: {edhrec_rank}  \n")
                print(df.loc[card_name]["oracle_text"], "\n")
                if rulings != "":
                    print(f"Special rulings: \t {rulings}\n")
                print("=" * 60, "\n")
                j += 1

            i += 1

        print(f"Found {j} cards containing '{contains}' in oracle text. Search complete.")