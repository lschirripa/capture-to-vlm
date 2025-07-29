import requests
import json
import time
from datetime import datetime
from utils.utils import print_framed_output

# --- Configuration ---
# IMPORTANT: Replace with the actual IP address of your Ubuntu server
UBUNTU_SERVER_IP = "10.147.17.37"
UBUNTU_SERVER_PORT = 8000
SERVER_ENDPOINT = f"http://{UBUNTU_SERVER_IP}:{UBUNTU_SERVER_PORT}/chat_text_only"

# Timeout for the HTTP request to the server.
# Text-only LLM inference is usually faster than VLM, but keep it generous.
REQUEST_TIMEOUT_SECONDS = 60
# --- End Configuration ---

def send_text_prompt(user_query: str, system_prompt: str = None):
    """
    Sends a text prompt to the server.
    Prompts are fixed but the user query can be changed.
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "user_query": user_query
    }
    if system_prompt:
        payload["system_prompt"] = system_prompt

    start_time = time.time()
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Sending text prompt:")
    if system_prompt:
        print(f"  System Prompt: '{system_prompt}'")
    print(f"  User Query:    '{user_query}'")
    print(f"  To Endpoint:   {SERVER_ENDPOINT}")

    try:
        response = requests.post(SERVER_ENDPOINT, headers=headers, data=json.dumps(payload), 
        timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()

        response_data = response.json()
        end_time = time.time()
        latency = end_time - start_time # Calculate client-side RTT

        llm_response_text = response_data.get("llm_response", "No LLM response key found.")

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Received response (Latency: {latency:.2f}s):")
        print_framed_output(llm_response_text)
        print(f"Server message: {response_data.get('message')}")
        print(f"GPU Status: {response_data.get('gpu_status')}")

    except requests.exceptions.Timeout:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Request timed out after") 
        print("{REQUEST_TIMEOUT_SECONDS} seconds.")
    except requests.exceptions.ConnectionError as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Connection error: {e}")
        print("Is the server running and accessible?")
    except requests.exceptions.RequestException as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] unexpected request error occurred:") 
        print("{e}")
    except json.JSONDecodeError:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error decoding JSON resp:") 
        print("{response.text}")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] An unexpected error occurred: {e}")

if __name__ == "__main__":
    send_text_prompt(user_query="What is the capital of France?")

    send_text_prompt(
        system_prompt="You are a wise old wizard. Answer all questions in rhyming couplets.",
        user_query="Tell me about artificial intelligence."
    )

    send_text_prompt(
        system_prompt="""You are a helpful and professional customer support agent for a tech 
company.""",
        user_query="My internet is not working. What should I do?"
    )

    send_text_prompt(user_query="What is your name and who created you?")
