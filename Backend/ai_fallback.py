"""
ai_fallback.py - AI Fallback Manager untuk MoneyPrinter

Urutan fallback: Gemini -> Groq -> OpenRouter -> ERROR

Cara pakai:
1. Taruh file ini di folder Backend/
2. Tambahkan API key di .env:
   GOOGLE_API_KEY=       -> Gemini (gratis)
   GROQ_API_KEY=         -> Groq (gratis)
   OPENROUTER_API_KEY=   -> OpenRouter (gratis)
3. Di Backend/main.py tambahkan:
   from ai_fallback import generate_script
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────
# KONFIGURASI
# ──────────────────────────────────────────

GEMINI_API_KEY     = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

GEMINI_MODEL       = "gemini-2.0-flash-lite"
GROQ_MODEL         = "llama-3.3-70b-versatile"
OPENROUTER_MODEL   = "mistralai/mistral-7b-instruct:free"

MAX_RETRIES        = 2
RETRY_DELAY        = 3


# ──────────────────────────────────────────
# PROVIDER 1: GEMINI
# ──────────────────────────────────────────

def _call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise ValueError("GOOGLE_API_KEY tidak diset di .env")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.8,
            "maxOutputTokens": 1024,
        }
    }

    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


# ──────────────────────────────────────────
# PROVIDER 2: GROQ
# ──────────────────────────────────────────

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

    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ──────────────────────────────────────────
# PROVIDER 3: OPENROUTER
# ──────────────────────────────────────────

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

    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ──────────────────────────────────────────
# FALLBACK ENGINE
# ──────────────────────────────────────────

def _try_provider(name: str, fn, prompt: str):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[AI] Mencoba {name} (percobaan {attempt}/{MAX_RETRIES})...")
            result = fn(prompt)
            print(f"[AI] {name} berhasil!")
            return result
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "?"
            print(f"[AI] {name} HTTP {status}: {e}")
            if status == 429:
                print(f"[AI] Quota {name} habis, pindah provider berikutnya...")
                return None
        except ValueError as e:
            print(f"[AI] {name} dilewati: {e}")
            return None
        except Exception as e:
            print(f"[AI] {name} error: {e}")

        if attempt < MAX_RETRIES:
            print(f"[AI] Retry dalam {RETRY_DELAY} detik...")
            time.sleep(RETRY_DELAY)

    return None


# ──────────────────────────────────────────
# FUNGSI UTAMA
# ──────────────────────────────────────────

def generate_script(topic: str, language: str = "Indonesian") -> str:
    prompt = f"""
Kamu adalah penulis skrip video YouTube Shorts yang profesional.

Buat skrip video pendek (maksimal 60 detik) tentang: "{topic}"

Ketentuan:
- Bahasa: {language}
- Durasi: 45-60 detik saat dibaca keras
- Gaya: santai, menarik, informatif
- Mulai langsung dengan hook yang kuat (tanpa intro "Halo teman-teman")
- Akhiri dengan call-to-action singkat
- Tulis HANYA skrip narasi saja, tanpa keterangan seperti [OPENING] atau [HOOK]
- Tidak perlu tanda kurung atau petunjuk sutradara

Tulis skripnya sekarang:
"""

    providers = [
        ("Gemini",     _call_gemini),
        ("Groq",       _call_groq),
        ("OpenRouter", _call_openrouter),
    ]

    for name, fn in providers:
        result = _try_provider(name, fn, prompt)
        if result and result.strip():
            return result.strip()

    raise RuntimeError(
        "Semua AI provider gagal!\n"
        "Pastikan minimal satu API key diset di .env:\n"
        "  GOOGLE_API_KEY        -> Gemini\n"
        "  GROQ_API_KEY          -> Groq\n"
        "  OPENROUTER_API_KEY    -> OpenRouter"
    )


# ──────────────────────────────────────────
# TEST — jalankan: python ai_fallback.py
# ──────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("TEST AI Fallback Manager")
    print("=" * 50)

    test_topic = "5 Fakta Mengejutkan Tentang Luar Angkasa"

    try:
        script = generate_script(test_topic)
        print("\nHASIL SKRIP:")
        print("-" * 50)
        print(script)
        print("-" * 50)
        print(f"Berhasil! ({len(script.split())} kata)")
    except RuntimeError as e:
        print(f"\n{e}")
