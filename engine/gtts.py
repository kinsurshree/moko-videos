"""Google Cloud Text-to-Speech (Chirp 3 HD, Hindi) - one voice per character, one call per line.
Key is read from env GOOGLE_TTS_KEY (never stored in the repo)."""
import os, json, base64, urllib.request, subprocess, hashlib
import numpy as np

VOICES = {  # character -> Google voice (hi-IN)
    "moko": "hi-IN-Chirp3-HD-Leda",
    "drip": "hi-IN-Chirp3-HD-Puck",
    "sunny": "hi-IN-Chirp3-HD-Aoede",
    "wind": "hi-IN-Chirp3-HD-Charon",
}
RATE = {"moko": 0.95, "drip": 1.0, "sunny": 0.95, "wind": 0.88}   # speaking speed (kids need it slow and clear)

def synth_line(text, who, out_mp3, lang="hi-IN"):
    key = os.environ.get("GOOGLE_TTS_KEY")
    if not key: raise RuntimeError("GOOGLE_TTS_KEY not set")
    voice = VOICES.get(who, VOICES["moko"])
    body = {"input": {"text": text}, "voice": {"languageCode": lang, "name": voice},
            "audioConfig": {"audioEncoding": "MP3", "speakingRate": RATE.get(who, 1.0)}}
    req = urllib.request.Request(f"https://texttospeech.googleapis.com/v1/text:synthesize?key={key}",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=60)); break
        except Exception as e:
            if attempt == 2: raise
    open(out_mp3, "wb").write(base64.b64decode(d["audioContent"]))

def synth_episode(ep_dir, spec, decode, SR):
    """returns list of numpy segments (one per line), loudness-matched, silence trimmed"""
    segs = []
    for i, l in enumerate(spec["lines"]):
        p = os.path.join(ep_dir, f"line_{i:02d}.mp3")
        synth_line(l["text"], l["who"], p)
        x = decode(p)
        # trim leading/trailing silence
        hop = int(0.01 * SR); n = len(x) // hop
        r = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1))
        on = np.where(r > 0.01)[0]
        if len(on): x = x[max(0, on[0] - 3) * hop:(on[-1] + 6) * hop]
        # match loudness
        rms = np.sqrt(np.mean(x ** 2)) + 1e-9
        x = np.clip(x * (0.11 / rms), -0.98, 0.98).astype(np.float32)
        segs.append(x)
    return segs
