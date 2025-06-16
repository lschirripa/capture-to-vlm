import requests
import json
import time
from datetime import datetime
import os
from colorama import Fore, Style, init
from rich.console import Console
from rich.panel import Panel
from rich import box
from utils.utils import print_framed_output

# --- Configuration ---
# IMPORTANT: Replace with the actual IP address of your Ubuntu server
UBUNTU_SERVER_IP = "10.147.17.37"
UBUNTU_SERVER_PORT = 8000

# The endpoint for summarization
SERVER_ENDPOINT = f"http://{UBUNTU_SERVER_IP}:{UBUNTU_SERVER_PORT}/summarize_vlm_output"

# Path to the VLM output file (video frame descriptions)
VLM_OUTPUT_DIR = "vlm_outputs" # Must match the directory in the VLM client script
VLM_OUTPUT_FILENAME = "vlm_responses.txt" # Must match the filename in the VLM client script
VLM_OUTPUT_FILE_PATH = os.path.join(VLM_OUTPUT_DIR, VLM_OUTPUT_FILENAME)

# NEW: Path to the audio transcription file
AUDIO_TRANSCRIPTION_FILE_PATH = "audio_transcription.txt"

# Adjust based on your model and hardware.
REQUEST_TIMEOUT_SECONDS = 300

# --- Customizable Prompts for Summarization ---
SUMMARIZATION_SYSTEM_PROMPT = """You are an AI assistant tasked with summarizing a chronological log of 
events described by a Vision Language Model (VLM) and an accompanying audio transcription.
The log contains descriptions of individual frames from a video stream, including timestamps and VLM 
observations. The audio transcription provides spoken content with timestamps.
Your goal is to provide a concise, coherent, and chronological summary of what occurred across all 
frames, integrating relevant information from the audio transcription.
Focus on identifying key actions, changes in the scene, significant objects or interactions, and important 
spoken events.
Do not invent information. If an action or spoken event is repeated, note its duration or recurrence.
Your summary should be presented as a flowing narrative, connecting the observations and spoken content naturally.
You should be as short and concise as possible within 400 characters more or less
"""

# The user query will implicitly be the content of the VLM output file + audio transcription.
# We don't need a separate user_query variable here as the file content serves as the main input.
# --- End Configuration ---

def main():
    # --- Read VLM output file ---
    if not os.path.exists(VLM_OUTPUT_FILE_PATH):
        print(f"Error: VLM output file not found at '{VLM_OUTPUT_FILE_PATH}'.")
        print("Please run the VLM client script (`main3.py`) first to generate the descriptions.")
        return

    print(f"Reading VLM outputs from: {VLM_OUTPUT_FILE_PATH}")
    try:
        with open(VLM_OUTPUT_FILE_PATH, "r", encoding="utf-8") as f:
            vlm_output_content = f.read()
    except Exception as e:
        print(f"Error reading VLM output file: {e}")
        return

    audio_transcription_content = ""
    if not os.path.exists(AUDIO_TRANSCRIPTION_FILE_PATH):
        print(f"Warning: Audio transcription file not found at '{AUDIO_TRANSCRIPTION_FILE_PATH}'.")
        print("Proceeding with VLM output only.")
    else:
        print(f"Reading audio transcription from: {AUDIO_TRANSCRIPTION_FILE_PATH}")
        try:
            with open(AUDIO_TRANSCRIPTION_FILE_PATH, "r", encoding="utf-8") as f:
                audio_transcription_content = f.read()
        except Exception as e:
            print(f"Error reading audio transcription file: {e}")
            print("Proceeding with VLM output only.")

    combined_user_query = f"""
    VLM Observations (Video Frame Descriptions):
    {vlm_output_content}

    ---

    Audio Transcription:
    {audio_transcription_content}
    """

    # Prepare the payload for the summarization endpoint
    headers = {"Content-Type": "application/json"}
    payload = {
        "system_prompt": SUMMARIZATION_SYSTEM_PROMPT,
        "user_query": combined_user_query
    }

    start_time = time.time()
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Sending VLM output and audio transcription for summarization...")
    print(f"  To Endpoint: {SERVER_ENDPOINT}")
    print(f"  Using System Prompt: {SUMMARIZATION_SYSTEM_PROMPT[:100]}...") # Print a truncated version
    print(f"  User Query (first 100 chars): {combined_user_query[:100]}...") # Print a truncated version

    try:
        response = requests.post(SERVER_ENDPOINT, headers=headers, data=json.dumps(payload), timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()

        response_data = response.json()
        end_time = time.time()
        latency = end_time - start_time # Calculate client-side RTT

        summary_text = response_data.get("llm_response", "No summary text from LLM.")

        # print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Received summary (Latency: {latency:.2f}s):")
        print(f"--- Summarization Result ---")
        print(summary_text)
        print_framed_output(summary_text)
        
        # print(f"--- End Summarization Result ---")
        # print(f"Server message: {response_data.get('message')}")
        # print(f"GPU Status: {response_data.get('gpu_status')}")

    except requests.exceptions.Timeout:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Request timed out after {REQUEST_TIMEOUT_SECONDS} seconds.")
        print("The summarization model (llama3-70B-cool:latest) might be very slow or requires more resources.")
    except requests.exceptions.ConnectionError as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Connection error: {e}. Is the server running and accessible?")
    except requests.exceptions.RequestException as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] An unexpected request error occurred: {e}")
    except json.JSONDecodeError:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error decoding JSON response from server. Raw text: {response.text}")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()