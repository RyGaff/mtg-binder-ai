import cv2
import pytesseract
import argparse
import sys
import time
import numpy as np
import requests
import re

ESC_KEY_CODE = 27
DEFAULT_TIMEOUT = 60
def scan_card(img_path, show_scan=False):
    print(f"Scanning card: {img_path}")
    processed_card = preprocess_card(img_path)
    if show_scan:
        show_card(processed_card)
    return extract_card_text(processed_card)

def extract_card_text(img):
    print("Extracting card text")
    custom_config = r'--oem 1 --psm 6'
    text = pytesseract.image_to_string(img, lang='eng', config=custom_config)
    #text = pytesseract.image_to_string(img)
    return text.lower()

def find_roi_bottom_left(image, scale_factor=2):
    height, width, _ = image.shape
    roi = image[int(height*0.93):int(height*0.97), int(width*0.05):int(width*0.187)]
    return  roi

def preprocess_card(img):
    return process_bottom_left(img)

def process_bottom_left(img):
    # Step 1: Load the image
    image = cv2.imread(img)
    cv2.imwrite("step_1_original.png", image)  # Save the original image for reference

    image = find_roi_bottom_left(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("step_2_grayscale.png", gray)  # Grayscale removes color information, focusing on intensity
    print("Gray", extract_card_text(gray))

    kernel = np.ones((1, 1), np.uint8)
    noise_removal = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    cv2.imwrite("step_3_noise_removal.png", noise_removal)
    print("Noise", extract_card_text(noise_removal))

    return noise_removal

def close_window(window_name):
    """Handle the cleanup of an OpenCV window."""
    cv2.destroyAllWindows()
    print(f"{window_name} window closed")

def show_card(card_image, timeout=DEFAULT_TIMEOUT):
    window_name = "Card"
    start_time = time.time()
    cv2.imshow(window_name, card_image)

    while True:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            close_window(window_name)
            break

        if cv2.waitKey(1) & 0xFF == ESC_KEY_CODE:  # ESC key pressed
            print("User closed the card window with ESC key")
            close_window(window_name)
            break

        if time.time() - start_time > timeout:  # Timeout exceeded
            print("Card window timed out")
            close_window(window_name)
            break

def query_scryfall(cn = None, set = None, card_name = None):
    if card_name is not None:
        response = requests.get(f"https://api.scryfall.com/cards/named?fuzzy={card_name}")
    else:
        cn = cn.strip()
        set = set.strip()
        print(">>>", cn, set, "<<<")
        response = requests.get(f"https://api.scryfall.com/cards/search?q=set%3A{set}+cn%3A{cn}")
        print(response.json())
    response.raise_for_status()
    return response.json()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image processing')
    parser.add_argument('image_path', type=str, help='Card to process')
    args = parser.parse_args()
    card = preprocess_card(args.image_path)
    extracted_text = re.split(r"[ \-\n]", extract_card_text(card))
    #show_card(card)
    print(extracted_text)
    # Search up scryfall for the card
    query_scryfall(cn=extracted_text[1], set=extracted_text[2])

    # If offline use saved data

    # If online query

    sys.exit(0)
