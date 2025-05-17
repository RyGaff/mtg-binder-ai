from FlagEmbedding import BGEM3FlagModel # type: ignore
import json
import os
import numpy as np
import pandas as pd
import faiss
from sklearn.metrics.pairwise import cosine_similarity
useful_features = ["card_faces" , "cmc", "color_identity", "defense", "edhrec_rank", "game_changer", 
                   "keywords",  "life_modifier", "loyalty", "mana_cost", "name", "oracle_text", 
                   "power", "produced_mana", "toughness", "type_line", "rulings"]

saved_embeddings = "Embedded_Magic_cards.pkl"
redo_embeddings = True


#TODO # Need to be able to define the importance of words. 
# For example we get a lot of cards that are similar because they share the same "name " for example all of the urzas are more similar than other cards
# I want cards to similar soly their attributes. 
# (It may be better for now to just do the oracle text and ignore everything else? But these you still ahve the problem where cards reference themselves and the model thinks that means they are similar)
# Which yes they are however the name of the card should not matter. Only Some aspects of card text we care about
def gen_text_cards(df):
    samples = []
    for card_name, row in df.iterrows():
        row_strings = [f"{card_name};; "]
        for column in df.columns:
            if column != 'name':
                value = row[column]
                if value == '':
                    continue
                row_strings.append(f"{column}; {value}")
        # Combine into single string
        sample = f"{card_name}; " + ", ".join(row_strings)
        samples.append(sample)
    
    return samples

def cosine_similarity_search(df, card_name, all_embeddings):
    if card_name not in df.index:
        print(f"{card_name} is not a card present within dataframe")
        return None
    else:
        target_embedding = np.array(df.loc[card_name, "embeddings"]).reshape(1, -1)
        similarities = cosine_similarity(target_embedding, all_embeddings)[0]

        similarities_df = pd.DataFrame({'card_name': df.index,
                                        'similarity': similarities})
        similarities_df = similarities_df[similarities_df['card_name'] != card_name]

        return similarities_df.sort_values(by='similarity', ascending=False)

def k_most_similar(search_term, df, k=5):
    cos_search_results = cosine_similarity_search(df, search_term, np.stack(df["embeddings"].values))
    if cos_search_results is not None:
        print(f"Top {k} most similar cards to search")
        print(cos_search_results.head(k))
        for i in cos_search_results.head(k)["card_name"]:
            print(i + ":", df.loc[i]["oracle_text"], df.loc[i])

if __name__ == "__main__":
    read_scryfall_data = False
    redo_embeddings = False
    search_term = "Urza, Lord High Artificer" 
    # Read the oracle and rulings if saved or if user indicates
    if not os.path.exists(saved_embeddings) or read_scryfall_data:
        print(f"Building data frame and doing embeddings")
        with open("Data/oracle-cards-20250425090226.json", "r") as file:
            data = json.load(file)
        with open("Data/rulings-20250425090033.json", "r") as file:
            rulings = json.load(file)

        oracle_id_loc = {}
        for i in range(len(data)):
            oracle_id_loc[data[i]["oracle_id"]] = i

        for r in rulings:
            if 'rulings' not in data[oracle_id_loc[r["oracle_id"]]]:
                data[oracle_id_loc[r["oracle_id"]]]['rulings'] = r['comment'] 
            else: 
                data[oracle_id_loc[r["oracle_id"]]]['rulings'] + " " + r['comment']
        #Process the data
        df = pd.DataFrame(data, columns=useful_features)
        df = df[df['type_line'] != "Card // Card"]
        df = df.fillna('')
        df.set_index('name', inplace=True)
    else:
        print(f"Dataframe read from {saved_embeddings}")
        df = pd.read_pickle(saved_embeddings)

    if redo_embeddings:
        #Create Text card object to encode as sentences 
        text_cards = gen_text_cards(df)
        embedding_model = BGEM3FlagModel('BAAI/bge-m3') 
        #card_embeddings = embedding_model.encode(df.index.tolist(), return_dense=True)
        card_embeddings = embedding_model.encode(text_cards, return_dense=True)
        print(card_embeddings)
        print(df.shape, card_embeddings['dense_vecs'].shape)
        print(card_embeddings['dense_vecs'][-1])
        df['embeddings'] = card_embeddings['dense_vecs'].tolist()
        df.to_pickle(saved_embeddings)
        print(f"Build data frame and saved do {saved_embeddings}")

    #embeddings = np.array(df["embeddings"].tolist()).astype('float32')
    #string_to_int_id = {str_id: i for i, str_id in enumerate(df.index)}

    k_most_similar(search_term, df, 5)
