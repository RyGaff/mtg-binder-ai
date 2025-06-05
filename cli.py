import helpers
import argparse
import requests
import json
import time
import os
import re
import sys
from helpers import get_embeddings, perform_search
from cardReader import scan

saved_embeddings = "Embedded_Magic_cards.pkl"

def get_scryfall_bulk_data(data_type="default_cards", output_dir="Data"):
   bulk_data_url = "https://api.scryfall.com/bulk-data"
   print(f"Fetching bulk data index from: {bulk_data_url}")

   try:
      response = requests.get(bulk_data_url)
      response.raise_for_status()
      bulk_data_index = response.json()

      for data_object in bulk_data_index.get("data", []):
         download_url = data_object.get("download_uri")
         data_type = data_object.get("type")

         # All cards is two 2Gb so skip
         if data_type == "all_cards":
            continue

         print(f"Downloading '{data_type}': {download_url}")
         os.makedirs(output_dir, exist_ok=True)
         output_path = os.path.join(output_dir, re.sub(r'\d', '',os.path.basename(download_url)))

         print(f"Downloading bulk data to: {output_path}")
         data_response = requests.get(download_url, stream=True)
         data_response.raise_for_status()

         with open(output_path, 'wb') as f:
            for chunk in data_response.iter_content(chunk_size=8192):
               f.write(chunk)
         print(f"Successfully downloaded '{data_type}' bulk data to {output_path}\n{'=' * 60}\n")

   except requests.exceptions.RequestException as e:
      print(f"Network or HTTP error occurred: {e}")
   except json.JSONDecodeError as e:
      print(f"Error decoding JSON response: {e}")
   except Exception as e:
      print(f"An unexpected error occurred: {e}")


#Downloads artwork to output_dir and returns missing image_uris
def download_artwork(unique_artwork, output_dir="Data/images/", set_id=None):
   print(f"Downloading unique artwork to: {output_dir}... This may take a while")
   if set_id is not None:
      print(f"Downloading normal images from set {set_id}")
      return
   else :
      print("Downloading all artwork")
      with open(unique_artwork, 'r') as f:
         data = json.load(f)

      #Extract and download the image uris
       # Attempt to download large, if that does not exist try normal if not, then try small

      missing = []
      i = 0
      total = len(data)
      for item in data:
         i += 1
         i_toString = f"{i}/{total}"
         if "image_uris" in item:
            if item['image_uris']["large"] is not None:
               image_url = item['image_uris']["large"]
            elif item['image_uris']["normal"] is not None:
               image_url = item['image_uris']["normal"]
            elif item['image_uris']["small"] is not None:
               image_url = item['image_uris']["small"]
            else:
               print("No image found... Should not happen")
               missing.append(item)
               continue
            try:
               response = requests.get(image_url, stream=True)
               response.raise_for_status()  # Raise an exception for HTTP errors
               clean_name = re.sub(r'[^\w\- ]', '', item['name'])
               clean_name = re.sub(r'\s+', '-', clean_name)
               save_name = item['set'] + "-" + item['collector_number'] + "-" + clean_name + '.jpg'
               save_path = os.path.join(output_dir, save_name)

               with open(save_path, 'wb') as file:
                  for chunk in response.iter_content(1024):  # Download in chunks
                     file.write(chunk)

               print(f"{i_toString}  Downloaded: {save_path}")
            except requests.exceptions.RequestException as error:
               print(f"{i_toString}  Failed to download image: {error}")
               print(f"Image url: {image_url}")
               missing.append(item)
         else:
            missing.append(item)

   return missing

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description=
                                    'Cli tool to manage and search Magic: the Gathering cards')

   parser.add_argument('-u', '--update', action='store_true', help='Update the scryfall bulk data')
   parser.add_argument('-e', '--embeddings', action='store_true', help='Redo the vector embeddings')
   parser.add_argument('-q', '--query', type=str, help='Search through similar cards to the given card name')
   parser.add_argument('-c', '--contains', type=str, help='Ensure the similar cards contain the given string', default="")
   parser.add_argument('-i', '--identity', type=str, help='Ensure the similar cards are within the specified color identity', default="")
   parser.add_argument('-k', '--numRet', type=str, help='Adjust the number of cards to filter', default=10)
   parser.add_argument('-m', '--model', type=str, help='Model to use for vector embeddings. NOTE: This it is software is highly model dependent', default="paraphrase-MiniLM-L6-v2")
   parser.add_argument('-s', '--scan', type=str, help='Card image to scan and query scryfall for')
   parser.add_argument('--Art', action='store_true', help='Download All Artwork')
   args = parser.parse_args()

   print(args)

   update_embeddings = args.embeddings
   if args.update or os.path.exists('Data/oracle-cards-.json') is False:
      # Update all bulk data
      print("Updating bulk data...")
      get_scryfall_bulk_data()
      update_embeddings = True

   if args.query:
      # Redo embeddings since we have new data or if user asked
      # Otherwise just fetched the stored dataframe
      if update_embeddings:
         print("Updating embeddings...")
      else:
         print("Getting embeddings from file")
      df = get_embeddings(update_embeddings, model=args.model)
      perform_search(df, search_vect=args.query, n=int(args.numRet), contains=args.contains, identity=args.identity)

   if args.scan:
      scan(args.scan)
      sys.exit(0)

   if args.Art:
      download_artwork(unique_artwork="Data/unique-artwork-.json", output_dir="Data/images")
