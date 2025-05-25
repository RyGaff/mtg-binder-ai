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
    # Todo Algorithmically format the card
    # Search for largest contour in image
    # Warp the image such that it is right side up and cropped properly
    # Proceed with the following
    height, width, _ = image.shape
    roi = image[int(height*0.93):int(height*0.98), int(width*0.05):int(width*0.187)]
    return  roi

def preprocess_card(img, scale_factor=4):
    # Todo Handle cards that are missing set id and card number
    # Calculate new dimensions
    image = cv2.imread(img)
    new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
    # Resize the image
    enlarged_image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    return process_bottom_left(enlarged_image)

def process_bottom_left(image):
    #Load the iamge
    cv2.imwrite("step_1_original.png", image)  # Save the original image for reference

    #Crop to find set id and card number
    image = find_roi_bottom_left(image)

    # Clean up image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("step_2_grayscale.png", gray)  # Grayscale removes color information, focusing on intensity
    print("Gray", extract_card_text(gray))

    gauss = cv2.GaussianBlur(gray, (5, 5), 0)

    kernel = np.ones((1, 1), np.uint8)
    noise_removal = cv2.morphologyEx(gauss, cv2.MORPH_OPEN, kernel)
    cv2.imwrite("step_3_noise_removal.png", noise_removal)
    print("Noise", extract_card_text(noise_removal))

    threshold = cv2.threshold(noise_removal, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    #cv2.imwrite("step_4_threshold.png", noise_removal)
    #print("Thresh", extract_card_text(threshold))
    #threshold = cv2.adaptiveThreshold(noise_removal, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_OTSU, 5, 3)
    # Save and display the results
    cv2.imwrite("thresholded_image.png", threshold)

    # Todo. Handle different lighting from images in world

    return threshold

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

def query_scryfall(cn = None, set_id = None, card_name = None):
    if card_name is not None:
        response = requests.get(f"https://api.scryfall.com/cards/named?fuzzy={card_name}")
    else:
        cn = cn.strip()
        set_id = set_id.strip()
        print(f"using set_id id {set_id} and using card number {cn}")
        response = requests.get(f"https://api.scryfall.com/cards/search?q=set%3A{set_id}+cn%3A{cn}")
        print(response.json())
    response.raise_for_status()
    return response.json()

def scan(image_path):
    card = preprocess_card(image_path)
    extracted_text = re.split(r"[^a-zA-Z0-9/]", extract_card_text(card))
    #show_card(card)
    print(extracted_text)
    # Loop through to make more reliable
    card_id = ""
    set_id = ""
    for i in range(len(extracted_text)):
        # Find card_id - find numeric number or some division operator
        if extracted_text[i].isnumeric() and card_id == "":
            card_id = extracted_text[i]
        if '/' in extracted_text[i] and card_id == "":
            card_id = extracted_text[i].split('/')[0]
        # Find set_id - find 3 character code
        if len(extracted_text[i]) == 3  and extracted_text[i].isalpha() and set_id == "":
            set_id = extracted_text[i]

        # They should be the first returned if ocr read properly
        if card_id != "" and set_id != "":
            break

    print(card_id, set_id)
    query_scryfall(cn=card_id, set_id=set_id)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image processing')
    parser.add_argument('image_path', type=str, help='Card to process')
    args = parser.parse_args()
    scan(args.image_path)
    sys.exit(0)
