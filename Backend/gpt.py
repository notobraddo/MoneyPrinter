import re
import os
import json
import requests

from dotenv import load_dotenv
from logstream import log
from typing import Tuple, List, Optional
from utils import ENV_FILE

# Load environment variables
load_dotenv(ENV_FILE)

# Groq Configuration
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Fallback: Gemini
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")

# Fallback: OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = "mistralai/mistral-7b-instruct:free"


def _call_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY tidak diset di .env")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 1024,
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise ValueError("GOOGLE_API_KEY tidak diset di .env")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.8, "maxOutputTokens": 1024},
    }

    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def _call_openrouter(prompt: str) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY tidak diset di .env")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/FujiwaraChoki/MoneyPrinter",
        "X-Title": "MoneyPrinter",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 1024,
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def generate_response(prompt: str, ai_model: str = "") -> str:
    """
    Generate a response using Groq (primary) with fallback to Gemini and OpenRouter.
    """
    providers = [
        ("Groq",       _call_groq),
        ("Gemini",     _call_gemini),
        ("OpenRouter", _call_openrouter),
    ]

    last_error = None
    for name, fn in providers:
        try:
            log(f"[AI] Mencoba {name}...", "info")
            result = fn(prompt)
            if result and result.strip():
                log(f"[AI] {name} berhasil!", "success")
                return result.strip()
        except ValueError as e:
            log(f"[AI] {name} dilewati: {e}", "warning")
            continue
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "?"
            log(f"[AI] {name} HTTP {status}: {e}", "error")
            last_error = e
            if status == 429:
                log(f"[AI] Quota {name} habis, pindah provider...", "warning")
            continue
        except Exception as e:
            log(f"[AI] {name} error: {e}", "error")
            last_error = e
            continue

    raise RuntimeError(
        f"Semua AI provider gagal. Error terakhir: {last_error}\n"
        "Pastikan minimal satu API key diset di .env:\n"
        "  GROQ_API_KEY, GOOGLE_API_KEY, atau OPENROUTER_API_KEY"
    )


# Kept for compatibility — tidak dipakai tapi jangan dihapus
def list_ollama_models() -> Tuple[List[str], str]:
    return (["groq/llama-3.3-70b-versatile"], "groq/llama-3.3-70b-versatile")


def generate_script(
    video_subject: str,
    paragraph_number: int,
    ai_model: str,
    voice: str,
    customPrompt: str,
) -> Optional[str]:
    # Build prompt
    if customPrompt:
        prompt = customPrompt
    else:
        prompt = """
            Generate a script for a video, depending on the subject of the video.

            The script is to be returned as a string with the specified number of paragraphs.

            Here is an example of a string:
            "This is an example string."

            Do not under any circumstance reference this prompt in your response.

            Get straight to the point, don't start with unnecessary things like, "welcome to this video".

            Obviously, the script should be related to the subject of the video.

            YOU MUST NOT INCLUDE ANY TYPE OF MARKDOWN OR FORMATTING IN THE SCRIPT, NEVER USE A TITLE.
            YOU MUST WRITE THE SCRIPT IN THE LANGUAGE SPECIFIED IN [LANGUAGE].
            ONLY RETURN THE RAW CONTENT OF THE SCRIPT. DO NOT INCLUDE "VOICEOVER", "NARRATOR" OR SIMILAR INDICATORS OF WHAT SHOULD BE SPOKEN AT THE BEGINNING OF EACH PARAGRAPH OR LINE. YOU MUST NOT MENTION THE PROMPT, OR ANYTHING ABOUT THE SCRIPT ITSELF. ALSO, NEVER TALK ABOUT THE AMOUNT OF PARAGRAPHS OR LINES. JUST WRITE THE SCRIPT.

        """

    prompt += f"""
    
    Subject: {video_subject}
    Number of paragraphs: {paragraph_number}
    Language: {voice}

    """

    # Generate script
    response = generate_response(prompt, ai_model)

    log(response, "info")

    if response:
        response = response.replace("*", "")
        response = response.replace("#", "")
        response = re.sub(r"\[.*\]", "", response)
        response = re.sub(r"\(.*\)", "", response)

        paragraphs = response.split("\n\n")
        selected_paragraphs = paragraphs[:paragraph_number]
        final_script = "\n\n".join(selected_paragraphs)

        log(f"Number of paragraphs used: {len(selected_paragraphs)}", "success")
        return final_script
    else:
        log("[-] AI returned an empty response.", "error")
        return None


def get_search_terms(
    video_subject: str, amount: int, script: str, ai_model: str
) -> List[str]:
    prompt = f"""
    Generate {amount} search terms for stock videos,
    depending on the subject of a video.
    Subject: {video_subject}

    The search terms are to be returned as
    a JSON-Array of strings.

    Each search term should consist of 1-3 words,
    always add the main subject of the video.
    
    YOU MUST ONLY RETURN THE JSON-ARRAY OF STRINGS.
    YOU MUST NOT RETURN ANYTHING ELSE. 
    YOU MUST NOT RETURN THE SCRIPT.
    
    The search terms must be related to the subject of the video.
    Here is an example of a JSON-Array of strings:
    ["search term 1", "search term 2", "search term 3"]

    For context, here is the full text:
    {script}
    """

    response = generate_response(prompt, ai_model)
    log(response, "info")

    search_terms = []

    try:
        search_terms = json.loads(response)
        if not isinstance(search_terms, list) or not all(
            isinstance(term, str) for term in search_terms
        ):
            raise ValueError("Response is not a list of strings.")

    except (json.JSONDecodeError, ValueError):
        log("[*] AI returned an unformatted response. Attempting to clean...", "warning")

        match = re.search(r"\[[\s\S]*\]", response)
        if match:
            try:
                search_terms = json.loads(match.group())
            except json.JSONDecodeError:
                search_terms = []

        if not search_terms:
            search_terms = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', response)
            search_terms = [term.strip() for term in search_terms if term.strip()]

    log(f"\nGenerated {len(search_terms)} search terms: {', '.join(search_terms)}", "info")
    return search_terms


def generate_metadata(
    video_subject: str, script: str, ai_model: str
) -> Tuple[str, str, List[str]]:
    title_prompt = f"""  
    Generate a catchy and SEO-friendly title for a YouTube shorts video about {video_subject}.  
    """
    title = generate_response(title_prompt, ai_model).strip()

    description_prompt = f"""  
    Write a brief and engaging description for a YouTube shorts video about {video_subject}.  
    The video is based on the following script:  
    {script}  
    """
    description = generate_response(description_prompt, ai_model).strip()
    keywords = get_search_terms(video_subject, 6, script, ai_model)

    return title, description, keywords