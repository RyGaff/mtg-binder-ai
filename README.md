# Usage
```
python3 cli.py -s "Urza, Lord High Artificer"

usage: cli.py [-h] [-u] [-e] [-s SEARCH] [-m MODEL]
options:
  -h, --help            show this help message and exit
  -u, --update          Update the scryfall bulk data
  -e, --embeddings      Redo the vector embeddings
  -s SEARCH, --search SEARCH
                        Search through similar cards to the given card name
  -m MODEL, --model MODEL
                        Model to use for vector embeddings. NOTE: This it is software is highly model dependent
                        the current default model is paraphrase-MiniLM-L6-v
```
# Features:
Scryfall Bulk Data Integration: 
  Query and update stored Scryfall bulk data for access to every Magic: The Gathering card.
  
Vector Embeddings for Semantic Search and Customizable Embedding Models: 
  Generate numerical vector embeddings from card oracle text, to allow the user to quickly search for similar cards based on the one provided.
  IMPORTANT NOTE! You can use any sentence transformer model for this but keep in mind this will greatly impact card similarity.
      (creating a specific model and tuning it for the magic the gathering language will be an on going project)

# Todo:
Card Analysis tool:
  Search for "synergistic" cards
  Parameter to use regular languge to search instead of card name 
  Custom SentenceTransformer model specialized in the "language" of mtg

Collection Manager:
  Manage your collection from one place.
  Easily sort through bulk.
  Import and export decks to multiple sites.
  Check value of cards. 
  Create decks.
  Goldfish!?
  Integration with analysis tool to find sneaky cards that go well with your deck. 
  
Card Scanner:
  Scan physical cards to integrate integrate into personal collection or just to find some cool stats!
