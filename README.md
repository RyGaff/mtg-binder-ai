# Usage
```
usage: cli.py [-h] [-u] [-e] [-q QUERY] [-c CONTAINS] [-k NUMRET] [-m MODEL] [-s SCAN]

Cli tool to manage and search Magic: the Gathering cards

options:
  -h, --help            show this help message and exit
  -u, --update          Update the scryfall bulk data
  -e, --embeddings      Redo the vector embeddings
  -q QUERY, --query QUERY
                        Search through similar cards to the given card name
  -c CONTAINS, --contains CONTAINS
                        Ensure the similar cards contain the given string
  -k NUMRET, --numRet NUMRET
                        Adjust the number of cards to filter
  -m MODEL, --model MODEL
                        Model to use for vector embeddings. NOTE: This it is software is highly model
                        dependent
  -s SCAN, --scan SCAN  Scan card
```
# Features:
**Scryfall Bulk Data Integration:**
  * Query and update stored Scryfall bulk data for access to every Magic: The Gathering card.
  
**Vector Embeddings for Semantic Search and Customizable Embedding Models:** 
  * Generate numerical vector embeddings from card oracle text, to allow the user to quickly search for similar cards based on the one provided.
  * IMPORTANT NOTE! You can use any sentence transformer model for this but keep in mind this will greatly impact card similarity. (creating a specific model and tuning it for the magic the gathering language will be an on going project)

# Todo:
**Card Analysis tool:**
  * Search for "synergistic" cards
  * Parameter to use regular languge to search instead of card name 
  * Custom SentenceTransformer model specialized in the "language" of mtg

**Collection Manager:**
  * Manage your collection from one place.
  * Easily sort through bulk.
  * Import and export decks to multiple sites.
  * Check value of cards. 
  * Create decks.
  * Goldfish!?
  * Integration with analysis tool to find sneaky cards that go well with your deck. 
  
**Card Scanner:**
  * Scan physical cards to integrate integrate into personal collection or just to find some cool stats!
