from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import epitran
import nltk
import numpy as np
import soundfile as sf
import noisereduce as nr
import httpx
import tempfile
import os

# ── CMU dict setup ────────────────────────────────────────────────────────────
try:
    nltk.data.find("corpora/cmudict")
except LookupError:
    nltk.download("cmudict")

from nltk.corpus import cmudict
CMU = cmudict.dict()

# ── Globals (loaded once at startup) ─────────────────────────────────────────
_whisper = None
_epi_hindi = None

def get_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        print("Loading Whisper large-v3...")
        _whisper = WhisperModel("large-v3", device="cpu", compute_type="int8")
        print("Whisper ready.")
    return _whisper

def get_epitran():
    global _epi_hindi
    if _epi_hindi is None:
        _epi_hindi = epitran.Epitran("hin-Deva")
    return _epi_hindi

# ── Lifespan: warm up models in background after server starts ────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    loop = asyncio.get_event_loop()
    # Load models in a thread so the server becomes available immediately
    loop.run_in_executor(None, get_whisper)
    loop.run_in_executor(None, get_epitran)
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── ARPAbet → IPA ─────────────────────────────────────────────────────────────
ARPA_TO_IPA = {
    "AA":"ɑ",  "AE":"æ",  "AH":"ʌ",  "AO":"ɔ",  "AW":"aʊ", "AY":"aɪ",
    "B":"b",   "CH":"tʃ", "D":"d",   "DH":"ð",  "EH":"ɛ",  "ER":"ɝ",
    "EY":"eɪ", "F":"f",   "G":"g",   "HH":"h",  "IH":"ɪ",  "IY":"iː",
    "JH":"dʒ", "K":"k",   "L":"l",   "M":"m",   "N":"n",   "NG":"ŋ",
    "OW":"oʊ", "OY":"ɔɪ", "P":"p",   "R":"r",   "S":"s",   "SH":"ʃ",
    "T":"t",   "TH":"θ",  "UH":"ʊ",  "UW":"uː", "V":"v",   "W":"w",
    "Y":"j",   "Z":"z",   "ZH":"ʒ",
}

def arpa_to_ipa(arpa_list):
    return [ARPA_TO_IPA.get(t.rstrip("012"), t.rstrip("012")) for t in arpa_list]

# ── Phoneme extraction ────────────────────────────────────────────────────────
def get_phonemes_english(text: str) -> list[str]:
    import eng_to_ipa as ipa_fb
    result = []
    for word in text.lower().split():
        w = word.strip(".,!?;:'\"")
        if w in CMU:
            result.extend(arpa_to_ipa(CMU[w][0]))
        else:
            ipa = ipa_fb.convert(w)
            result.extend([ch for ch in ipa if ch.strip()])
    return result

def get_phonemes_hindi(text: str) -> list[str]:
    return list(get_epitran().trans_list(text))

def get_phonemes(text: str, language: str) -> list[str]:
    return get_phonemes_hindi(text) if language == "hindi" else get_phonemes_english(text)

# ── Levenshtein ───────────────────────────────────────────────────────────────
def levenshtein(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1] if a[i-1]==b[j-1] else 1+min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])
    return dp[m][n]

# ── Rule-based fallback feedback (used if Ollama not running) ─────────────────
PAIR_TIPS = {
    ("b","w"):  "Press both lips together firmly and pop the sound — don't round them.",
    ("r","w"):  "Curl your tongue upward — the difference from 'w' is the tongue, not just the lips.",
    ("s","θ"):  "Your tongue needs to stay behind your teeth for 's', not between them.",
    ("θ","s"):  "For 'th', slide your tongue tip gently between or behind your front teeth.",
    ("k","t"):  "For 'k' the contact is at the back of your mouth — think of the back of your throat.",
    ("l","w"):  "For 'l', your tongue tip must touch the ridge behind your top teeth.",
    ("ʃ","s"):  "For 'sh', pull your tongue slightly back and push your lips forward slightly.",
    ("tʃ","ʃ"):"For 'ch', you need the 't' build-up first before releasing into 'sh'.",
    ("r","l"):  "For 'r', curl the tongue up without touching the roof. For 'l', the tip must touch.",
    ("v","b"):  "For 'v', upper teeth rest on the lower lip. For 'b', both lips press together.",
    ("f","θ"):  "For 'f', upper teeth on lower lip. For 'th', tongue goes between the teeth.",
}
PHONEME_TIPS = {
    "r":"Curl your tongue tip up toward the roof — don't touch it. Round lips slightly.",
    "l":"Press tongue tip firmly on the ridge behind your top teeth. Let the sides drop.",
    "s":"Tongue close to the ridge, not touching. Channel air down the middle groove.",
    "z":"Same as 's' — add voice. Feel your throat buzz.",
    "θ":"Tongue tip gently between front teeth. Blow air over it. No voice.",
    "ð":"Same as 'θ' but add voice — your throat should vibrate.",
    "ʃ":"Pull tongue slightly back from 's'. Push lips forward. Wider, fuller sound.",
    "tʃ":"Tongue on ridge like 't', then release into 'sh' — one smooth movement.",
    "dʒ":"Like 'ch' but voiced. Start at ridge like 'd', release into voiced 'zh'.",
    "v":"Upper teeth on lower lip. Blow air and add voice — feel the buzz.",
    "f":"Upper teeth on lower lip. Blow air through. No voice.",
    "k":"Back of tongue touches the soft palate — the back of the roof of your mouth.",
    "g":"Same as 'k' — back of tongue on soft palate — but add your voice.",
    "æ":"Open wide, push tongue flat and forward. It should feel stretched.",
    "ɪ":"Tongue high and forward, lips slightly spread. Keep it short.",
    "ʊ":"Round lips loosely, tongue high and back. Short and relaxed.",
    "ə":"Relax everything. Tongue in the middle. The softest, laziest sound.",
}

def rule_based_feedback(expected, detected) -> str:
    for e, d in zip(expected, detected):
        if e != d:
            tip = PAIR_TIPS.get((e,d)) or PAIR_TIPS.get((d,e)) or PHONEME_TIPS.get(e)
            if tip:
                return f"/{e}/ needs work: {tip}"
    return "Good effort! Practise slowly, one sound at a time."

def rule_based_phoneme_feedback(target: str, detected: list[str]) -> str:
    closest = detected[0] if detected else None
    if closest:
        tip = PAIR_TIPS.get((target, closest)) or PAIR_TIPS.get((closest, target))
        if tip: return tip
    return PHONEME_TIPS.get(target,
        "Study the diagram carefully and focus on where your tongue and lips are placed.")

# ── Ollama phi3:mini feedback ─────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"

SYSTEM_PROMPT = (
    "You are a warm, encouraging speech therapist working with a child. "
    "Give one specific, concrete instruction (1–2 sentences max) about how to physically "
    "produce the target sound correctly. Focus on tongue position, lip shape, or airflow. "
    "Use simple words a child can understand. Never say 'great job' or give generic praise. "
    "Never repeat the phoneme symbol back. Just give the physical instruction."
)

async def llm_word_feedback(
    target_word: str,
    expected: list[str],
    detected: list[str],
    transcript: str,
) -> str:
    # Find the first mismatch
    mismatches = [(e, d) for e, d in zip(expected, detected) if e != d]
    if not mismatches:
        return "Great pronunciation! Every sound came through clearly."

    e, d = mismatches[0]
    prompt = (
        f"The child was trying to say '{target_word}'. "
        f"They produced '{transcript}'. "
        f"The target sound was /{e}/ but they said /{d}/. "
        f"Give one specific physical instruction to help them produce /{e}/ correctly."
    )
    return await call_ollama(prompt)

async def llm_phoneme_feedback(target: str, detected: list[str], transcript: str) -> str:
    closest = detected[0] if detected else "an unclear sound"
    prompt = (
        f"The child was trying to produce the isolated sound /{target}/. "
        f"They said '{transcript}' and produced /{closest}/ instead. "
        f"Give one specific physical instruction to help them produce /{target}/ correctly."
    )
    return await call_ollama(prompt)

async def call_ollama(prompt: str) -> str | None:
    """Call Ollama local API. Returns None if Ollama isn't running."""
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": False,
                "options": {
                    "temperature": 0.3,   # low temp = more consistent clinical advice
                    "num_predict": 80,    # cap at ~2 sentences
                },
            })
            if resp.status_code == 200:
                text = resp.json().get("response", "").strip()
                # Strip any IPA symbols phi3 might hallucinate into the output
                return text if text else None
    except (httpx.ConnectError, httpx.TimeoutException):
        pass  # Ollama not running — fall back silently
    return None

# ── Audio preprocessing ───────────────────────────────────────────────────────
def preprocess_audio(path: str) -> str:
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    # Use first 0.5s as noise profile (background before speech starts)
    noise_sample = data[:int(sr * 0.5)] if len(data) > sr * 0.5 else data
    reduced = nr.reduce_noise(
        y=data, sr=sr, y_noise=noise_sample,
        prop_decrease=0.85, stationary=False,
    )
    # Peak normalise to 0.9
    peak = np.max(np.abs(reduced))
    if peak > 0.001:
        reduced = reduced / peak * 0.9
    out_path = path.replace(".wav", "_clean.wav")
    sf.write(out_path, reduced, sr)
    return out_path

def transcribe(path: str) -> str:
    clean = preprocess_audio(path)
    try:
        segs, _ = get_whisper().transcribe(
            clean, language=None, beam_size=5, vad_filter=True,
        )
        return " ".join(s.text.strip() for s in segs).strip().lower()
    finally:
        if os.path.exists(clean):
            os.unlink(clean)

def match_phonemes(expected, detected):
    return [
        {
            "expected": ep,
            "detected": detected[i] if i < len(detected) else None,
            "correct": i < len(detected) and ep == detected[i],
        }
        for i, ep in enumerate(expected)
    ]

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/phonemes")
async def phonemes_endpoint(
    word: str = Form(...),
    language: str = Form(default="english"),
):
    return {"word": word, "language": language, "phonemes": get_phonemes(word, language)}


@app.post("/compare")
async def compare(
    audio: UploadFile = File(...),
    target_word: str = Form(...),
    language: str = Form(default="english"),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        transcript = transcribe(tmp_path)
        expected = get_phonemes(target_word, language)
        detected = get_phonemes(transcript, language) if transcript else []
        dist = levenshtein(expected, detected)
        accuracy = max(0, round((1 - dist / max(len(expected), 1)) * 100))
        matches = match_phonemes(expected, detected)

        # Try LLM feedback, fall back to rule-based
        feedback = await llm_word_feedback(target_word, expected, detected, transcript)
        if not feedback:
            feedback = rule_based_feedback(expected, detected)

        return {
            "transcript": transcript,
            "target_word": target_word,
            "expected_phonemes": expected,
            "detected_phonemes": detected,
            "matches": matches,
            "accuracy": accuracy,
            "feedback": feedback,
        }
    finally:
        os.unlink(tmp_path)


@app.post("/compare_phoneme")
async def compare_phoneme(
    audio: UploadFile = File(...),
    target_phoneme: str = Form(...),
    language: str = Form(default="english"),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        transcript = transcribe(tmp_path)
        detected = get_phonemes(transcript, language) if transcript else []
        correct = target_phoneme in detected

        feedback = ""
        if not correct:
            feedback = await llm_phoneme_feedback(target_phoneme, detected, transcript)
            if not feedback:
                feedback = rule_based_phoneme_feedback(target_phoneme, detected)

        return {
            "transcript": transcript,
            "target_phoneme": target_phoneme,
            "detected_phonemes": detected,
            "correct": correct,
            "feedback": feedback,
        }
    finally:
        os.unlink(tmp_path)


@app.get("/")
def root():
    return {"status": "VaakSiddhi backend running"}
