from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
import numpy as np
import cv2
import torch
import base64
import requests
import json
from starlette.concurrency import run_in_threadpool
import time
from datetime import datetime



app = FastAPI()

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_VLM_MODEL = "qwen2.5vl:7b"  # change model as needed
OLLAMA_LLM_MODEL = "llama3:8b"

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_model = torch.nn.Linear(10, 1).to(device)
    print(f"CUDA is available and dummy model loaded on GPU. device: {device}" if device.type == 'cuda' else f"CUDA not available, using CPU for dummy model. device: {device}")
except Exception as e:
    print(f"Error initializing GPU/PyTorch: {e}")
    device = "cpu"
    print("Falling back to CPU for dummy model due to error.")

class ProcessingResponse(BaseModel):
    message: str
    llm_response: str
    gpu_status: str = "N/A"

class TextPromptRequest(BaseModel):
    system_prompt: str = None # Optional system prompt
    user_query: str # Mandatory user query

def _call_ollama_api_sync(model: str, prompt: str, image_base64: str = None):
    """
    Helper function for Ollama API call (unified, accepts optional image_base64)
    model: str - The Ollama model to use
    prompt: str - The prompt to send to the model
    image_base64: str - The base64 encoded image to send to the model
    Returns:
        dict - The response from the Ollama API
    """

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if image_base64:
        payload["images"] = [image_base64]

    try:
        ollama_response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            headers=headers,
            data=json.dumps(payload),
            timeout=300
        )
        ollama_response.raise_for_status()
        return ollama_response.json()
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Ollama request timed out. Model might be loading or busy.")
    except requests.exceptions.RequestException as e:
        status_code = getattr(e.response, "status_code", 500)
        detail = f"Error calling Ollama: {e}"
        if hasattr(e, 'response') and e.response is not None:
             detail += f" - Ollama Response: {getattr(e.response, 'text', 'No response body')}"
        raise HTTPException(status_code=status_code, detail=detail)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Error decoding Ollama response. Raw text: {ollama_response.text if 'ollama_response' in locals() else 'N/A'}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error during Ollama call: {e}")


@app.post("/process_frame", response_model=ProcessingResponse)
async def process_frame(image: UploadFile = File(...)):
    """
    Endpoint for processing a single image. It describes what is happening in the image in detail.
    Args:
        image: UploadFile - The image to process
    Returns:
        ProcessingResponse - The response from the Ollama API
    """

    request_start_time = time.time()
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    try:
        contents = await image.read()
        img_np = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Received frame for /process_frame (shape: {img.shape})")
        gpu_status_msg = "No specific GPU processing performed for this image (CPU fallback)."
        try:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)
            gpu_status_msg = "Image tensor moved to GPU."
        except Exception as e:
            print(f"GPU processing error: {e}")
            gpu_status_msg = f"GPU processing failed: {e}"
        _, buffer = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        prompt = "Describe what is happening in this image in detail. Focus on objects, actions, and the overall scene."
        llm_response_text = "VLM processing failed."
        ollama_call_start_time = time.time()
        try:
            ollama_response_json = await run_in_threadpool(_call_ollama_api_sync, OLLAMA_VLM_MODEL, prompt, img_base64)
            ollama_call_end_time = time.time()
            llm_response_text = ollama_response_json.get("response", "No response text from VLM.")
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Ollama VLM Response for /process_frame (Ollama Time: {ollama_call_end_time - ollama_call_start_time:.2f}s): {json.dumps(ollama_response_json, indent=2)[:500]}...")
        except HTTPException as e:
            ollama_call_end_time = time.time()
            llm_response_text = f"Ollama VLM error: {e.detail}"
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] FastAPI caught Ollama HTTPException for /process_frame (Ollama Time: {ollama_call_end_time - ollama_call_start_time:.2f}s): {e.detail}")
        except Exception as e:
            ollama_call_end_time = time.time()
            llm_response_text = f"An unexpected error occurred during Ollama call for /process_frame: {e}"
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] FastAPI caught unexpected error for /process_frame (Ollama Time: {ollama_call_end_time - ollama_call_start_time:.2f}s): {e}")
        request_end_time = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Total request processing time for /process_frame: {request_end_time - request_start_time:.2f}s")
        return ProcessingResponse(
            message="Frame processed successfully",
            llm_response=llm_response_text,
            gpu_status=gpu_status_msg
        )
    except HTTPException:
        request_end_time = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Total request processing time (HTTPError): {request_end_time - request_start_time:.2f}s")
        raise
    except Exception as e:
        request_end_time = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Total request processing time (Unexpected Error): {request_end_time - request_start_time:.2f}s")
        raise HTTPException(status_code=500, detail=f"Internal server error in frame processing: {e}")

@app.post("/process_frame_with_prompt", response_model=ProcessingResponse)
async def process_frame_with_prompt(
    image: UploadFile = File(...),
    system_prompt: str = Form(None),
    user_query: str = Form(...)
):
    """
    Endpoint for processing a single image with a custom prompt.
    Args:
        image: UploadFile - The image to process
        system_prompt: str - The system prompt to use
        user_query: str - The user query to use
    Returns:
        ProcessingResponse - The response from the Ollama API
    """
    request_start_time = time.time()
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    try:
        contents = await image.read()
        img_np = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Received frame for /process_frame_with_prompt (shape: {img.shape})")
        gpu_status_msg = "No specific GPU processing performed for this image (CPU fallback)."
        try:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)
            gpu_status_msg = "Image tensor moved to GPU."
        except Exception as e:
            print(f"GPU processing error: {e}")
            gpu_status_msg = f"GPU processing failed: {e}"
        _, buffer = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        full_prompt = ""
        if system_prompt:
            full_prompt += f"{system_prompt}\n\n"
        full_prompt += user_query
        llm_response_text = "VLM processing failed."
        ollama_call_start_time = time.time()
        try:
            ollama_response_json = await run_in_threadpool(_call_ollama_api_sync, OLLAMA_VLM_MODEL, full_prompt, img_base64)
            ollama_call_end_time = time.time()
            llm_response_text = ollama_response_json.get("response", "No response text from VLM.")
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Ollama VLM Response for /process_frame_with_prompt (Ollama Time: {ollama_call_end_time - ollama_call_start_time:.2f}s): {json.dumps(ollama_response_json, indent=2)[:500]}...")
        except HTTPException as e:
            ollama_call_end_time = time.time()
            llm_response_text = f"Ollama VLM error: {e.detail}"
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] FastAPI caught Ollama HTTPException for /process_frame_with_prompt (Ollama Time: {ollama_call_end_time - ollama_call_start_time:.2f}s): {e.detail}")
        except Exception as e:
            ollama_call_end_time = time.time()
            llm_response_text = f"An unexpected error occurred during Ollama call for /process_frame_with_prompt: {e}"
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] FastAPI caught unexpected error for /process_frame_with_prompt (Ollama Time: {ollama_call_end_time - ollama_call_start_time:.2f}s): {e}")
        request_end_time = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Total request processing time for /process_frame_with_prompt: {request_end_time - request_start_time:.2f}s")
        return ProcessingResponse(
            message="Frame processed successfully with custom prompt",
            llm_response=llm_response_text,
            gpu_status=gpu_status_msg
        )
    except HTTPException:
        request_end_time = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Total request processing time (HTTPError): {request_end_time - request_start_time:.2f}s")
        raise
    except Exception as e:
        request_end_time = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Total request processing time (Unexpected Error): {request_end_time - request_start_time:.2f}s")
        raise HTTPException(status_code=500, detail=f"Internal server error in frame processing with prompt: {e}")

@app.post("/chat_text_only", response_model=ProcessingResponse)
async def chat_text_only(request: TextPromptRequest):
    """
    Endpoint for processing a text-only prompt.
    Args:
        request: TextPromptRequest - The request containing the system prompt and user query
    Returns:
        ProcessingResponse - The response from the Ollama API
    """
    request_start_time = time.time()
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Received text-only request for /chat_text_only")
    full_prompt = ""
    if request.system_prompt:
        full_prompt += f"{request.system_prompt}\n\n"
    full_prompt += request.user_query
    llm_response_text = "LLM processing failed."
    ollama_call_start_time = time.time()
    try:
        ollama_response_json = await run_in_threadpool(_call_ollama_api_sync, OLLAMA_VLM_MODEL, full_prompt, image_base64=None)
        ollama_call_end_time = time.time()
        llm_response_text = ollama_response_json.get("response", "No response text from LLM.")
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Ollama LLM Response for /chat_text_only (Ollama Time: {ollama_call_end_time - ollama_call_start_time:.2f}s): {llm_response_text[:500]}...")
    except HTTPException as e:
        ollama_call_end_time = time.time()
        llm_response_text = f"Ollama LLM error: {e.detail}"
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] FastAPI caught Ollama HTTPException for /chat_text_only (Ollama Time: {ollama_call_end_time - ollama_call_start_time:.2f}s): {e.detail}")
    except Exception as e:
        ollama_call_end_time = time.time()
        llm_response_text = f"An unexpected error occurred during Ollama call for /chat_text_only: {e}"
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] FastAPI caught unexpected error for /chat_text_only (Ollama Time: {ollama_call_end_time - ollama_call_start_time:.2f}s): {e}")
    request_end_time = time.time()
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Total request processing time for /chat_text_only: {request_end_time - request_start_time:.2f}s")
    return ProcessingResponse(
        message="Text-only prompt processed successfully",
        llm_response=llm_response_text,
        gpu_status="N/A (Text-only request)"
    )

@app.post("/summarize_vlm_output", response_model=ProcessingResponse)
async def summarize_vlm_output(request: TextPromptRequest):
    """
    Endpoint for summarizing the frame-to-frame output of the VLM model.
    Args:
        request: TextPromptRequest - The request containing the system prompt and the user query with the VLM output of the frames captured during the frame-to-frame processing.
    Returns:
        ProcessingResponse - The response from the Ollama API
    """
    request_start_time = time.time()
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Received summarization request for /summarize_vlm_output")

    full_prompt = ""
    if request.system_prompt:
        full_prompt += f"{request.system_prompt}\n\n"
    full_prompt += request.user_query # The VLM output content

    llm_summary_text = "Summarization failed."
    ollama_call_start_time = time.time()
    try:
        ollama_response_json = await run_in_threadpool(_call_ollama_api_sync, OLLAMA_LLM_MODEL, full_prompt, image_base64=None)
        ollama_call_end_time = time.time()
        llm_summary_text = ollama_response_json.get("response", "No response text from LLM.")
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Ollama LLM Summary Response (Ollama Time: {ollama_call_end_time - ollama_call_start_time:.2f}s): {llm_summary_text[:500]}...")
    except HTTPException as e:
        ollama_call_end_time = time.time()
        llm_summary_text = f"Ollama LLM error: {e.detail}"
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] FastAPI caught Ollama HTTPException for summarization (Ollama Time: {ollama_call_end_time - ollama_call_start_time:.2f}s): {e.detail}")
    except Exception as e:
        ollama_call_end_time = time.time()
        llm_summary_text = f"An unexpected error occurred during Ollama summarization call: {e}"
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] FastAPI caught unexpected error for summarization (Ollama Time: {ollama_call_end_time - ollama_call_start_time:.2f}s): {e}")

    request_end_time = time.time()
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] Total request processing time for /summarize_vlm_output: {request_end_time - request_start_time:.2f}s")

    return ProcessingResponse(
        message="VLM output summarized successfully",
        llm_response=llm_summary_text,
        gpu_status="N/A (Text summarization)"
    )

@app.get("/")
async def read_root():
    return {"message": "FastAPI server is running and ready for VLM/LLM processing!"}