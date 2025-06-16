import cv2
import requests
import time
from datetime import datetime
import os # Import the os module for path manipulation
from colorama import Fore, Style, init
from rich.console import Console
from rich.panel import Panel
from rich import box
from utils.utils import print_framed_output


# --- Configuration ---
# URL of your FastAPI server on Ubuntu
# IMPORTANT: Replace with the actual IP address of your Ubuntu server
UBUNTU_SERVER_IP = "10.147.17.37"
UBUNTU_SERVER_PORT = 8000

SERVER_ENDPOINT = f"http://{UBUNTU_SERVER_IP}:{UBUNTU_SERVER_PORT}/process_frame_with_prompt"

# Desired frames per second to process.
# BASED ON YOUR SERVER LOGS (average ~1.8s processing time per frame),
# setting TARGET_FPS to 0.5 means sending a frame every 2 seconds (1/0.5).
# This gives the server enough time to process and avoid building up a queue.
TARGET_FPS = 0.5

# Calculate the ideal delay needed between sending requests
IDEAL_SEND_INTERVAL = 1.0 / TARGET_FPS

# Timeout for the HTTP request to the server. Should be longer than your server's VLM processing time.
REQUEST_TIMEOUT_SECONDS = 5

# --- Output File Configuration ---
# Define the directory to save the output file
OUTPUT_DIR = "vlm_outputs"
# Define the name of the output text file
OUTPUT_FILENAME = "vlm_responses.txt"
# Full path for the output file
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
# --- End Configuration ---

# --- Customizable Prompts ---
BANK_ASSISTANT_SYSTEM_PROMPT ="""You are a helpful and professional bank assistant. Focus your descriptions 
on financial documents, payment methods, banking activities, or anything related to banking. Your response should be as short and concise as possible within 200 characters more or less"""
BANK_ASSISTANT_USER_QUERY = "What financial documents or activities are visible?"

GENERAL_SCENE_SYSTEM_PROMPT = None # No specific role, just general instructions
GENERAL_SCENE_USER_QUERY = "Describe the overall scene in detail. Your response should be as short and concise as possible within 200 characters more or less"

# Choose which prompt combination to use for testing
CURRENT_SYSTEM_PROMPT = GENERAL_SCENE_SYSTEM_PROMPT
CURRENT_USER_QUERY = GENERAL_SCENE_USER_QUERY
# --- End Customizable Prompts ---


def main():
    """
    Client that captures frames from the webcam and sends them to the VLM server.
    The server processes the frames and returns a response.
    The response is printed to the console and saved to a file.
    The client can be stopped by pressing Ctrl+C.
    """
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Open video capture (e.g., webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream. Make sure your webcam is available or video path is correct.")
        return

    print(f"Client configured to target {TARGET_FPS} frame(s) per second (every {IDEAL_SEND_INTERVAL:.2f} seconds).")
    print(f"Server Endpoint: {SERVER_ENDPOINT}")
    print(f"VLM responses will be saved to: {OUTPUT_FILE_PATH}")

    frame_count = 0
    last_successful_send_time = time.time()
    
    output_file = None # Initialize to None
    try:
        output_file = open(OUTPUT_FILE_PATH, "w", encoding="utf-8")
        output_file.write(f"--- VLM Responses from {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n\n")

        while True:
            # --- Pacing Logic ---
            time_to_wait = IDEAL_SEND_INTERVAL - (time.time() - last_successful_send_time)
            if time_to_wait > 0:
                time.sleep(time_to_wait)
            # --- End Pacing Logic ---

            ret, frame = cap.read()
            if not ret:
                print("End of stream or error reading frame.")
                break

            frame_count += 1
            current_send_time = time.time()

            # print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Capturing and sending frame {frame_count}...")
            # if CURRENT_SYSTEM_PROMPT:
                # print(f"  System Prompt: '{CURRENT_SYSTEM_PROMPT}...'")
            # print(f"  User Query:    '{CURRENT_USER_QUERY}'")

            _, img_encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            img_bytes = img_encoded.tobytes()

            files = {'image': ('frame.jpg', img_bytes, 'image/jpeg')}
            data = {'system_prompt': CURRENT_SYSTEM_PROMPT, 'user_query': CURRENT_USER_QUERY}

            try:
                response = requests.post(SERVER_ENDPOINT, files=files, data=data, timeout=REQUEST_TIMEOUT_SECONDS)
                response.raise_for_status()

                response_data = response.json()
                end_time = time.time()
                latency = end_time - current_send_time

                llm_response_text = response_data.get("llm_response", "No LLM response key found.")
                
                # print(f"[{datetime.now().strftime('%H:%M:%S')}] Server response (Latency: {latency:.2f}s): {llm_response_text}")
                # print(f"[{datetime.now().strftime('%H:%M:%S')}] {llm_response_text}")
                print_framed_output(llm_response_text)
                # --- Save to file ---
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                output_file.write(f"--- Frame {frame_count} ({timestamp}, Latency: {latency:.2f}s) ---\n")
                output_file.write(f"User Query: {CURRENT_USER_QUERY}\n")
                output_file.write(f"VLM Response:\n{llm_response_text}\n\n")
                output_file.flush()
                # --- End Save to file ---

                last_successful_send_time = end_time

            except requests.exceptions.Timeout:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Request for frame {frame_count} timed out after {REQUEST_TIMEOUT_SECONDS} seconds. Server busy or model loading?")
                last_successful_send_time = time.time()
            except requests.exceptions.ConnectionError as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Connection error: {e}. Is the server running and accessible?")
                break
            except requests.exceptions.RequestException as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] An unexpected request error occurred for frame {frame_count}: {e}")
                last_successful_send_time = time.time()
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] An unexpected error occurred during client processing for frame {frame_count}: {e}")
                last_successful_send_time = time.time()

    except KeyboardInterrupt:
        print("\nStopping stream due to user interrupt.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if output_file:
            output_file.write(f"\n--- End of Stream ---")
            output_file.close() # Ensure the file is properly closed
            print(f"VLM responses saved to: {OUTPUT_FILE_PATH}")
        print("Client gracefully shut down.")

if __name__ == "__main__":
    main()
