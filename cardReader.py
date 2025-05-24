import cv2
import pytesseract
import argparse
import sys
import time

ESC_KEY_CODE = 27
DEFAULT_TIMEOUT = 5
def scan_card(img_path, show_scan=False):
    print(f"Scanning card: {img_path}")
    processed_card = preprocess_card(img_path)
    if show_scan:
        show_card(processed_card)
    return extract_card_text(processed_card)

def extract_card_text(img):
    text = pytesseract.image_to_string(img)
    return text

#OCR does not seem to be the way to go
def preprocess_card(img):
    image = cv2.imread(img)
    #Convert to greyscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 30, 150)
    return edged

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image processing')
    parser.add_argument('image_path', type=str, help='Card to process')
    args = parser.parse_args()
    card = preprocess_card(args.image_path)
    show_card(card)
    print(extract_card_text(card))
    sys.exit(0)
