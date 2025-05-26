import cv2
import pytesseract
import argparse
import sys
import time
import numpy as np
import requests
import re
import os

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
    height = image.shape[0]
    width = image.shape[1]
    roi = image[int(height*0.93):int(height*0.98), int(width*0.05):int(width*0.187)]
    return  roi

def find_roi_title(image, scale_factor=2):
    # Todo Algorithmically format the card
    # Search for largest contour in image
    # Warp the image such that it is right side up and cropped properly
    # Proceed with the following
    #height, width, _ = image.shape
    height = image.shape[0]
    width = image.shape[1]
    roi = image[int(height*0.93):int(height*0.98), int(width*0.05):int(width*0.187)]
    return  roi

def find_roi_bottom(image, scale_factor=2):
    # Todo Algorithmically format the card
    # Search for largest contour in image
    # Warp the image such that it is right side up and cropped properly
    # Proceed with the following
    print(image.shape, len(image.shape))
    height = image.shape[0]
    width = image.shape[1]
    roi = image[int(height*0.92):int(height*0.97), int(width*0.05):int(width*0.95)]
    return  roi

def detect_card_layout(image, templates_folder="Data/templates/"):
    print("Detecting card layout...")

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Ensure input image is resized consistently
    input_image = cv2.resize(image, (500, 700), interpolation=cv2.INTER_CUBIC)

    # Detect keypoints and descriptors for input image
    kp1, des1 = orb.detectAndCompute(input_image, None)

    if des1 is None:
        raise ValueError("No features detected in the input image. Check the image quality or content.")

    layout_matches = {}

    # Process each template
    for template_file in os.listdir(templates_folder):
        template_path = os.path.join(templates_folder, template_file)
        template_image = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

        if template_image is None:
            print(f"Could not load template: {template_file}. Skipping...")
            continue

        # Resize template for consistency
        template_image = cv2.resize(template_image, (500, 700), interpolation=cv2.INTER_CUBIC)

        # Get keypoints and descriptors for the template
        kp2, des2 = orb.detectAndCompute(template_image, None)

        if des2 is None:
            print(f"No features detected in template {template_file}. Skipping...")
            continue

        # Check descriptor compatibility
        if des1.shape[1] != des2.shape[1]:
            print(f"Descriptor shape mismatch: des1.shape = {des1.shape}, des2.shape = {des2.shape}. Skipping...")
            continue

        # Match features using Brute Force Matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        try:
            matches = bf.match(des1, des2)
            layout_matches[template_file] = len(matches)
        except cv2.error as e:
            print(f"OpenCV error during matching: {e}. Skipping template...")
            continue

    if not layout_matches:
        raise ValueError("No matches found with any templates. Input layout may not match existing templates.")

    # Find the template with the most matches
    best_match = max(layout_matches, key=layout_matches.get)
    print(f"Best layout match: {best_match} with {layout_matches[best_match]} matches.")
    print(layout_matches)

    return best_match

def preprocess_card(img, scale_factor=4):
    # Todo Handle cards that are missing set id and card number
    # Calculate new dimensions
    image = cv2.imread(img)
    if image is None:
        raise ValueError("Image not found")


    detected_layout = detect_card_layout(image.copy())
    print(f"Detected layout: {detected_layout}")
    exit(1)
    new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
    # Resize the image
    enlarged_image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    return process_bottom_left(enlarged_image)

# Great for front facing "Regular/Newer" cards with constant lighting that has
# number and set symbol in the bottom left. Will work in most cases
def process_bottom_left(image, debug=False):
    #Crop to find set id and card number
    image = find_roi_bottom_left(image)

    # Clean up image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray, (5, 5), 0)

    kernel = np.ones((1, 1), np.uint8)
    noise_removal = cv2.morphologyEx(gauss, cv2.MORPH_OPEN, kernel)
    threshold = cv2.threshold(noise_removal, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    if debug:
        cv2.imwrite("step_1_original.png", image)  # Save the original image for reference
        cv2.imwrite("step_2_grayscale.png", gray)  # Grayscale removes color information, focusing on intensity
        print("Gray", extract_card_text(gray))
        cv2.imwrite("step_3_noise_removal.png", noise_removal)
        print("Noise", extract_card_text(noise_removal))
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
