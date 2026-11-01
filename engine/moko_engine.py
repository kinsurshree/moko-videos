"""Moko episode engine.

python3 moko_engine.py <episode_dir>
  episode_dir must contain: spec.json, voice.mp3   (voice = all lines in order, one file)
  writes: episode_dir/final.mp4, timing.json, preview.png

spec.json:
{
  "id": "ep02",
  "cast": ["moko", "drip", "wind"],            # any of drip, sunny, wind (moko always)
  "start_mood": "calm",                         # moko mood at start
  "location": "garden",                         # where the episode happens (see list)
  "lines": [
    {"who": "moko", "text": "...", "mood": "excited", "do": ["show:apple"]},
    {"who": "drip", "text": "...", "drip": "sad", "do": ["go:park"]},
    ...
  ]
}
moods: happy, calm, sad, angry, scared, excited, shy, proud
locations: sky, park, town, farm, road, night, school, garden
   (match the topic: road for road safety, night for bedtime, garden/farm for nature and food,
    school for letters/numbers, town for helping/sharing. Do not use the same location in two
    episodes in a row.)
do actions: show:<prop>, give:<prop>:<drip|sunny|wind|moko>, hide:<prop>, mud:on, mud:off,
            bubbles, enter:sunny, enter:wind, rainbow, go:<location> (change place, max once)
props: apple, ball, soap, toothbrush, water, star, heart, flower, umbrella, banana, book, n1..n9
Music, sound effects and camera movement are added automatically (music differs per episode id).
"""
import math, sys, os, json, random, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS, SR, S = 1080, 1920, 30, 44100, 2
HERE = os.path.dirname(os.path.abspath(__file__))
FONT = "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf"
PITCH = {"moko": 0.0, "drip": 4.0, "sunny": -3.0, "wind": -5.5}
GAP = 0.4

def lerp(a, b, t): return a + (b - a) * t
def lerpc(c1, c2, t): return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))
def ease(t): t = max(0.0, min(1.0, t)); return t * t * (3 - 2 * t)

# ======================= AUDIO =======================
def decode(path, rate=SR):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(rate), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()

def silences(x, thr_db=-35, min_d=0.3):
    hop = int(0.01 * SR)
    n = len(x) // hop
    r = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-9
    db = 20 * np.log10(r)
    quiet = db < thr_db
    out, i = [], 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]: j += 1
            if (j - i) * 0.01 >= min_d and i > 0 and j < n:
                out.append((i * 0.01, j * 0.01))
            i = j
        else:
            i += 1
    return out

def split_lines(x, texts):
    """Split one voice file into len(texts) lines using the longest pauses, guided by text length."""
    n = len(texts)
    dur = len(x) / SR
    sil = silences(x)
    if n == 1:
        return [(0.0, dur)]
    # choose n-1 ordered pauses (dynamic programming): segment lengths should match
    # each line's share of characters, and longer pauses are preferred as cut points
    if len(sil) < n - 1:
        raise RuntimeError("could not split voice into lines")
    lens = np.array([max(4, len(t.replace(" ", ""))) for t in texts], dtype=float)
    speech_total = dur - sum(b - a for a, b in sil)
    rate = speech_total / lens.sum()
    expv = lens * rate
    mids = [(a + b) / 2 for a, b in sil]
    durs = [b - a for a, b in sil]
    def speech_between(t0, t1):
        q = t1 - t0
        for a, b in sil:
            q -= max(0, min(b, t1) - max(a, t0))
        return q
    m = len(sil)
    INF = -1e18
    # dp[k][j]: best score with k cuts made, last cut at silence j
    dp = [[INF] * m for _ in range(n)]
    back = [[-1] * m for _ in range(n)]
    for j in range(m):
        seg = speech_between(0, mids[j])
        dp[1][j] = durs[j] * 2 - abs(seg - expv[0]) / expv[0]
    for k in range(2, n):
        for j in range(m):
            for i in range(j):
                if dp[k - 1][i] == INF: continue
                seg = speech_between(mids[i], mids[j])
                sc = dp[k - 1][i] + durs[j] * 2 - abs(seg - expv[k - 1]) / expv[k - 1]
                if sc > dp[k][j]: dp[k][j], back[k][j] = sc, i
    best_j, best = -1, INF
    for j in range(m):
        if dp[n - 1][j] == INF: continue
        seg = speech_between(mids[j], dur)
        sc = dp[n - 1][j] - abs(seg - expv[n - 1]) / expv[n - 1]
        if sc > best: best, best_j = sc, j
    if best_j < 0:
        raise RuntimeError("could not split voice into lines")
    idxs, k, j = [], n - 1, best_j
    while k >= 1:
        idxs.append(j); j = back[k][j]; k -= 1
    chosen = [sil[i] for i in reversed(idxs)]
    cuts = [0.0] + [((a + b) / 2) for a, b in chosen] + [dur]
    segs = []
    for k in range(n):
        a, b = cuts[k], cuts[k + 1]
        seg = x[int(a * SR):int(b * SR)]
        # trim silence at both ends
        hop = int(0.01 * SR)
        m = len(seg) // hop
        if m == 0:
            segs.append((a, b)); continue
        r = np.sqrt(np.mean(seg[:m * hop].reshape(m, hop) ** 2, axis=1))
        on = np.where(20 * np.log10(r + 1e-9) > -38)[0]
        if len(on):
            segs.append((a + max(0, on[0] - 3) * 0.01, a + min(m, on[-1] + 4) * 0.01))
        else:
            segs.append((a, b))
    return segs

def pitch(seg, semis):
    if abs(semis) < 0.01: return seg
    r = 2 ** (semis / 12)
    p = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
                        "-af", f"asetrate={int(SR * r)},aresample={SR},atempo={1 / r:.5f}",
                        "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                       input=seg.astype(np.float32).tobytes(), capture_output=True, check=True).stdout
    return np.frombuffer(p, dtype=np.float32).copy()

def _pluck(freq, dur, vol):
    t = np.arange(int(dur * SR)) / SR
    w = np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(4 * np.pi * freq * t) + 0.12 * np.sin(6 * np.pi * freq * t)
    env = np.minimum(1, t / 0.008) * np.exp(-t * 5.5)
    return (vol * w * env).astype(np.float32)

def _pad(freqs, dur, vol):
    t = np.arange(int(dur * SR)) / SR
    w = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
    env = np.minimum(1, t / 0.4) * np.minimum(1, (dur - t) / 0.4)
    return (vol * w * env).astype(np.float32)

def make_music(N, ep_id, voices):
    """Gentle, original background tune; key, tempo and melody differ per episode id."""
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(ep_id))
    r = random.Random(seed)
    root = r.choice([261.63, 293.66, 329.63, 349.23, 392.0])          # C D E F G
    bpm = r.choice([92, 100, 108, 116])
    beat = 60 / bpm
    scale = [0, 2, 4, 7, 9, 12, 14, 16]                                # major pentatonic
    prog = r.choice([[0, 5, 3, 4], [0, 3, 4, 0], [0, 4, 5, 3]])        # I vi IV V style
    chord_iv = {0: [0, 4, 7], 3: [5, 9, 12], 4: [7, 11, 14], 5: [9, 12, 16]}
    out = np.zeros(N, dtype=np.float32)
    total = N / SR
    t, bar, idx = 0.0, 0, r.randrange(len(scale))
    while t < total:
        deg = prog[bar % 4]
        base = root / 2
        pad = _pad([base * 2 ** (s / 12) for s in chord_iv[deg]], beat * 4, 0.035)
        i0 = int(t * SR); out[i0:i0 + len(pad)] += pad[:max(0, N - i0)]
        bass = _pluck(base / 2 * 2 ** (chord_iv[deg][0] / 12), beat * 2, 0.07)
        out[i0:i0 + len(bass)] += bass[:max(0, N - i0)]
        for b in range(4):
            if r.random() < 0.8:
                idx = max(0, min(len(scale) - 1, idx + r.choice([-2, -1, -1, 0, 1, 1, 2])))
                n = _pluck(root * 2 ** (scale[idx] / 12), beat * 1.5, 0.05)
                j = int((t + b * beat) * SR); out[j:j + len(n)] += n[:max(0, N - j)]
        t += beat * 4; bar += 1
    # duck under voices and fade in/out
    spf = SR // 100
    nf = N // spf
    v = np.sqrt(np.mean(voices[:nf * spf].reshape(nf, spf) ** 2, axis=1))
    duck = np.where(v > 0.01, 0.45, 1.0)
    duck = np.convolve(duck, np.ones(25) / 25, mode="same")
    duck = np.repeat(duck, spf)
    duck = np.pad(duck, (0, N - len(duck)), constant_values=1.0)
    fade = np.minimum(1, np.arange(N) / (0.8 * SR)) * np.minimum(1, (N - np.arange(N)) / (1.2 * SR))
    return (out * duck * fade * 2.0).astype(np.float32)

def extra_sfx(spec, lines, N, rng):
    out = np.zeros(N, dtype=np.float32)
    # soft rain hiss while Moko is sad or angry
    cur = spec.get("start_mood", "calm")
    for l, (who, a, b) in zip(spec["lines"], lines):
        cur = l.get("mood", cur)
        if cur in ("sad", "angry"):
            L = int((b - a + 0.4) * SR)
            hiss = rng.normal(0, 1, L).astype(np.float32)
            hiss = np.convolve(hiss, np.ones(6) / 6, mode="same")
            env = np.minimum(1, np.arange(L) / (0.3 * SR)) * np.minimum(1, (L - np.arange(L)) / (0.3 * SR))
            i0 = int(a * SR); out[i0:i0 + L] += (0.018 * hiss * env)[:max(0, N - i0)]
        for act in l.get("do", []):
            p = act.split(":")
            if p[0] in ("show", "give"):        # little sparkle
                t0 = a if p[0] == "show" else a + (b - a) * .5
                for k, f in enumerate([1568.0, 2093.0]):
                    tone = _pluck(f, 0.4, 0.06)
                    j = int((t0 + k * 0.07) * SR); out[j:j + len(tone)] += tone[:max(0, N - j)]
            elif p[0] in ("enter", "go"):         # whoosh
                L = int(0.7 * SR)
                nz = rng.normal(0, 1, L).astype(np.float32)
                k = np.linspace(2, 40, L).astype(int)
                sm = np.array([nz[max(0, i - k[i]):i + 1].mean() for i in range(0, L)], dtype=np.float32)
                env = np.sin(np.linspace(0, np.pi, L)) ** 2
                j = int((a - 0.6) * SR)
                if j >= 0: out[j:j + L] += (0.12 * sm * env)[:max(0, N - j)]
    return out

def build_audio(ep_dir, spec):
    x = decode(os.path.join(ep_dir, "voice.mp3"))
    texts = [l["text"] for l in spec["lines"]]
    segs = split_lines(x, texts)
    t = 0.6
    pieces, lines = [], []
    for l, (a, b) in zip(spec["lines"], segs):
        seg = pitch(x[int(a * SR):int(b * SR)], PITCH.get(l["who"], 0))
        pieces.append((l["who"], t, seg))
        lines.append([l["who"], round(t, 3), round(t + len(seg) / SR, 3)])
        t += len(seg) / SR + GAP
    total = round(t - GAP + 1.8, 2)
    N = int(total * SR)
    tracks = {}
    for who, ts, seg in pieces:
        tr = tracks.setdefault(who, np.zeros(N, dtype=np.float32))
        i0 = int(ts * SR)
        tr[i0:i0 + len(seg)] += seg[:max(0, N - i0)]
    sfx = np.zeros(N, dtype=np.float32)
    rng = np.random.default_rng(7)
    for li, l in enumerate(spec["lines"]):
        if l.get("mood") == "angry":
            r0, rl = lines[li][1] - 0.15, int(1.6 * SR)
            br = np.cumsum(rng.normal(0, 1, rl)).astype(np.float32)
            br -= np.convolve(br, np.ones(2000) / 2000, mode="same"); br /= (np.abs(br).max() + 1e-6)
            sfx[int(r0 * SR):int(r0 * SR) + rl] += 0.22 * br * np.sin(np.linspace(0, np.pi, rl)) ** 2
        if "rainbow" in l.get("do", []) or l.get("mood") in ("happy", "proud") and li == len(spec["lines"]) - 1:
            c0 = lines[li][1] - 0.3
            for k, f in enumerate([1046.5, 1318.5, 1568.0, 2093.0]):
                st, L = int((c0 + k * 0.12) * SR), int(1.2 * SR)
                tt = np.arange(L) / SR
                tone = (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt)) * np.exp(-tt * 4)
                sfx[st:st + L] += 0.10 * tone[:max(0, N - st)].astype(np.float32)
    voices = sum(tracks.values())
    sfx += extra_sfx(spec, lines, N, rng)
    music = make_music(N, spec.get("id", "ep"), voices)
    mix = voices + sfx + music
    mix = mix / max(1.0, np.abs(mix).max() / 0.95)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
                    "-ar", "48000", os.path.join(ep_dir, "audio.wav")], input=mix.astype(np.float32).tobytes(), check=True)
    env, spf = {}, SR // FPS
    for who, tr in tracks.items():
        nf = N // spf
        rms = np.sqrt(np.mean(tr[:nf * spf].reshape(nf, spf) ** 2, axis=1))
        pk = np.percentile(rms[rms > 0.005], 90) if np.any(rms > 0.005) else 1
        e = np.clip(rms / pk, 0, 1); e[e < 0.12] = 0
        env[who] = np.convolve(e, [0.25, 0.5, 0.25], mode="same")
    return lines, total, env

# ======================= DRAWING =======================
def vgrad(top, bot):
    g = Image.new("RGB", (1, H))
    for y in range(H): g.putpixel((0, y), lerpc(top, bot, y / (H - 1)))
    return g.resize((W, H)).convert("RGBA")

SKY = {
    "grey": vgrad((92, 100, 122), (160, 166, 180)),
    "calm": vgrad((140, 196, 236), (212, 234, 250)),
    "happy": vgrad((110, 186, 246), (214, 238, 255)),
    "fog": vgrad((170, 178, 190), (205, 210, 218)),
    "night": vgrad((40, 52, 96), (98, 110, 160)),
    "pink": vgrad((236, 186, 214), (250, 226, 236)),
}
MOOD = {  # sky, greyness of Moko, anger, smile, rain, snow, fog, shrink, rays, pink
    "happy":   dict(sky="happy", grey=0, anger=0, smile=1.0, rain=0, snow=0, fog=0, shrink=1.0, rays=0, pink=0),
    "calm":    dict(sky="calm", grey=0, anger=0, smile=0.55, rain=0, snow=0, fog=0, shrink=1.0, rays=0, pink=0),
    "sad":     dict(sky="grey", grey=0.5, anger=0, smile=0.05, rain=0.6, snow=0, fog=0, shrink=0.97, rays=0, pink=0),
    "angry":   dict(sky="grey", grey=1.0, anger=1.0, smile=0.0, rain=1.0, snow=0, fog=0, shrink=1.0, rays=0, pink=0),
    "scared":  dict(sky="fog", grey=0.3, anger=0, smile=0.15, rain=0, snow=0, fog=1.0, shrink=0.88, rays=0, pink=0),
    "excited": dict(sky="happy", grey=0, anger=0, smile=1.0, rain=0, snow=1.0, fog=0, shrink=1.03, rays=0, pink=0),
    "shy":     dict(sky="pink", grey=0, anger=0, smile=0.45, rain=0, snow=0, fog=0, shrink=0.97, rays=0, pink=1.0),
    "proud":   dict(sky="happy", grey=0, anger=0, smile=1.0, rain=0, snow=0, fog=0, shrink=1.06, rays=1.0, pink=0),
}

def hills(a, b):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for col, y0, amp, sd in ((a, 1620, 40, 1.0), (b, 1740, 30, 2.7)):
        d.polygon([(0, H)] + [(x, y0 + amp * math.sin(x / 260 + sd) + amp * .5 * math.sin(x / 110 + sd * 2)) for x in range(0, W + 40, 40)] + [(W, H)], fill=col + (255,))
    return im
HILLS_GREY, HILLS_GREEN = hills((96, 120, 104), (78, 100, 86)), hills((150, 214, 140), (118, 194, 118))

# ---------------- locations (ground layer, drawn at 2x then reduced) ----------------
def _ground(d, s, col_a, col_b, y0=1620):
    for col, yy, amp, sd in ((col_a, y0, 40, 1.0), (col_b, y0 + 120, 30, 2.7)):
        d.polygon([(0, s(H))] + [(s(x), s(yy + amp * math.sin(x / 260 + sd) + amp * .5 * math.sin(x / 110 + sd * 2))) for x in range(0, W + 40, 40)] + [(s(W), s(H))], fill=col + (255,))

def _tree(d, s, x, y, h):
    d.rounded_rectangle([s(x - 14), s(y - h * .45), s(x + 14), s(y)], radius=s(6), fill=(140, 96, 64, 255))
    for dx, dy, r in ((0, -h * .62, h * .26), (-h * .2, -h * .48, h * .2), (h * .2, -h * .48, h * .2), (0, -h * .82, h * .18)):
        d.ellipse([s(x + dx - r), s(y + dy - r), s(x + dx + r), s(y + dy + r)], fill=(96, 176, 96, 255))
    d.ellipse([s(x - h * .12), s(y - h * .8), s(x + h * .02), s(y - h * .68)], fill=(140, 210, 130, 255))

def _house(d, s, x, y, w, h, wall, roof):
    d.rectangle([s(x), s(y - h), s(x + w), s(y)], fill=wall + (255,))
    d.polygon([(s(x - 16), s(y - h)), (s(x + w / 2), s(y - h - w * .5)), (s(x + w + 16), s(y - h))], fill=roof + (255,))
    d.rounded_rectangle([s(x + w * .4), s(y - h * .5), s(x + w * .6), s(y)], radius=s(8), fill=(150, 100, 70, 255))
    for wx in (x + w * .12, x + w * .68):
        d.rounded_rectangle([s(wx), s(y - h * .8), s(wx + w * .2), s(y - h * .55)], radius=s(6), fill=(200, 230, 255, 255))

def _flowers(d, s, y0, y1, n, seed):
    rr = random.Random(seed)
    for _ in range(n):
        x, y = rr.uniform(20, W - 20), rr.uniform(y0, y1)
        col = rr.choice([(255, 140, 180), (255, 220, 90), (190, 150, 255), (255, 255, 255)])
        for k in range(5):
            a = k * 2 * math.pi / 5
            d.ellipse([s(x + 9 * math.cos(a) - 7), s(y + 9 * math.sin(a) - 7), s(x + 9 * math.cos(a) + 7), s(y + 9 * math.sin(a) + 7)], fill=col + (255,))
        d.ellipse([s(x - 5), s(y - 5), s(x + 5), s(y + 5)], fill=(255, 200, 60, 255))

def make_location(name):
    im = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0)); d = ImageDraw.Draw(im); s = lambda v: int(v * S)
    G1, G2 = (150, 214, 140), (118, 194, 118)
    if name == "park":
        _ground(d, s, G1, G2)
        for x, h in ((110, 330), (960, 380), (820, 260)): _tree(d, s, x, 1680, h)
        d.rounded_rectangle([s(560), s(1640), s(760), s(1662)], radius=s(8), fill=(190, 130, 80, 255))
        for lx in (580, 740): d.rectangle([s(lx), s(1662), s(lx + 12), s(1700)], fill=(150, 100, 60, 255))
        _flowers(d, s, 1730, 1880, 22, 3)
    elif name == "town":
        d.rectangle([0, s(1640), s(W), s(H)], fill=(170, 210, 150, 255))
        d.rectangle([0, s(1800), s(W), s(H)], fill=(200, 196, 190, 255))
        for x, w, h, wall, roof in ((30, 230, 200, (255, 214, 170), (230, 110, 90)), (300, 200, 250, (190, 220, 255), (90, 120, 200)),
                                    (540, 240, 190, (255, 240, 180), (220, 150, 70)), (820, 220, 230, (220, 200, 255), (160, 100, 190))):
            _house(d, s, x, 1680, w, h, wall, roof)
    elif name == "farm":
        for i, col in enumerate([(176, 216, 120), (232, 210, 120), (150, 204, 110), (224, 196, 104)]):
            y = 1600 + i * 80
            d.polygon([(0, s(y)), (s(W), s(y - 30)), (s(W), s(y + 80)), (0, s(y + 110))], fill=col + (255,))
        d.rectangle([s(700), s(1440), s(960), s(1640)], fill=(200, 70, 60, 255))
        d.polygon([(s(680), s(1440)), (s(830), s(1330)), (s(980), s(1440))], fill=(150, 50, 45, 255))
        d.rectangle([s(790), s(1540), s(870), s(1640)], fill=(255, 245, 230, 255))
        for x in range(0, 640, 70):
            d.rectangle([s(x), s(1610), s(x + 12), s(1690)], fill=(190, 140, 90, 255))
        d.rectangle([0, s(1630), s(640), s(1642)], fill=(190, 140, 90, 255)); d.rectangle([0, s(1662), s(640), s(1674)], fill=(190, 140, 90, 255))
    elif name == "road":
        d.rectangle([0, s(1600), s(W), s(H)], fill=G1 + (255,))
        d.rectangle([0, s(1700), s(W), s(1720)], fill=(220, 220, 220, 255))
        d.rectangle([0, s(1720), s(W), s(1880)], fill=(110, 116, 128, 255))
        for x in range(380, 720, 56): d.rectangle([s(x), s(1730), s(x + 32), s(1870)], fill=(250, 250, 250, 255))
        d.rectangle([s(960), s(1380), s(976), s(1700)], fill=(80, 80, 90, 255))
        d.rounded_rectangle([s(930), s(1290), s(1006), s(1450)], radius=s(14), fill=(60, 60, 70, 255))
        for k, col in enumerate([(240, 80, 80), (250, 200, 70), (90, 210, 110)]):
            d.ellipse([s(946), s(1302 + k * 48), s(990), s(1346 + k * 48)], fill=col + (255,))
        _tree(d, s, 120, 1660, 300)
    elif name == "night":
        _ground(d, s, (60, 96, 110), (44, 76, 92))
        d.ellipse([s(470), s(90), s(630), s(250)], fill=(255, 246, 200, 255))
        d.ellipse([s(515), s(70), s(665), s(220)], fill=(0, 0, 0, 0))
        rr = random.Random(9)
        for _ in range(40):
            x, y, r = rr.uniform(0, W), rr.uniform(60, 1100), rr.uniform(3, 7)
            d.ellipse([s(x - r), s(y - r), s(x + r), s(y + r)], fill=(255, 250, 220, 230))
    elif name == "school":
        d.rectangle([0, s(1640), s(W), s(H)], fill=G1 + (255,))
        d.rectangle([s(560), s(1380), s(1040), s(1680)], fill=(255, 220, 140, 255))
        d.polygon([(s(540), s(1380)), (s(800), s(1260)), (s(1060), s(1380))], fill=(220, 100, 80, 255))
        d.ellipse([s(760), s(1300), s(840), s(1380)], fill=(255, 255, 255, 255))
        d.line([s(800), s(1340), s(800), s(1312)], fill=(60, 60, 70, 255), width=s(5)); d.line([s(800), s(1340), s(820), s(1340)], fill=(60, 60, 70, 255), width=s(5))
        for wx in (600, 690, 890, 960): d.rounded_rectangle([s(wx), s(1440), s(wx + 60), s(1510)], radius=s(6), fill=(200, 230, 255, 255))
        d.rounded_rectangle([s(770), s(1560), s(830), s(1680)], radius=s(8), fill=(150, 100, 70, 255))
        d.rectangle([s(520), s(1200), s(528), s(1680)], fill=(120, 120, 130, 255))
        d.polygon([(s(528), s(1205)), (s(600), s(1225)), (s(528), s(1245))], fill=(255, 150, 60, 255))
    elif name == "garden":
        _ground(d, s, G1, G2, 1600)
        for row, y in enumerate((1700, 1780, 1860)):
            d.rectangle([0, s(y), s(W), s(y + 24)], fill=(150, 108, 76, 255))
            for x in range(40 + row * 30, W, 110):
                d.ellipse([s(x - 26), s(y - 34), s(x + 26), s(y + 6)], fill=(90, 170, 90, 255))
                if (x // 110 + row) % 3 == 0: d.ellipse([s(x - 10), s(y - 26), s(x + 10), s(y - 6)], fill=(235, 80, 70, 255))
        _flowers(d, s, 1610, 1680, 16, 5)
        _tree(d, s, 980, 1640, 320)
    else:  # sky (default rolling hills)
        _ground(d, s, G1, G2)
    col = im.resize((W, H), Image.LANCZOS)
    grey = Image.merge("RGBA", (*Image.blend(col.convert("RGB"), col.convert("L").convert("RGB"), .6).point(lambda v: int(v * .78)).split(), col.split()[3]))
    return col, grey

_LOC_CACHE = {}
def location(name):
    if name not in _LOC_CACHE: _LOC_CACHE[name] = make_location(name)
    return _LOC_CACHE[name]

CW, CH = 900, 680
PUFFS = [(450, 380, 250), (250, 420, 175), (650, 420, 175), (330, 270, 165), (580, 260, 180),
         (450, 200, 150), (150, 470, 110), (750, 470, 110), (450, 500, 190)]
ARMS = [(70, 430, 55), (830, 430, 55)]
INK = (52, 48, 66, 255)

def moko_body():
    mask = Image.new("L", (CW * S, CH * S), 0); md = ImageDraw.Draw(mask)
    for x, y, r in PUFFS + ARMS: md.ellipse([(x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S], fill=255)
    sh = Image.new("RGB", mask.size); sd = ImageDraw.Draw(sh)
    for y in range(CH * S): sd.line([(0, y), (CW * S, y)], fill=lerpc((255, 255, 255), (214, 226, 242), ease((y / (CH * S) - .35) / .65)))
    hl = Image.new("L", mask.size, 0); hd = ImageDraw.Draw(hl)
    for x, y, r in PUFFS: hd.ellipse([(x - r * .8) * S, (y - r * .8) * S, (x + r * .2) * S, (y - r * .1) * S], fill=120)
    sh = Image.composite(Image.new("RGB", sh.size, (255, 255, 255)), sh, hl.filter(ImageFilter.GaussianBlur(30 * S)))
    edge = mask.filter(ImageFilter.GaussianBlur(18 * S))
    inv = Image.composite(Image.eval(edge, lambda v: 255 - v), Image.new("L", mask.size, 0), mask)
    sh = Image.composite(Image.new("RGB", sh.size, (190, 204, 226)), sh, Image.eval(inv, lambda v: int(v * .55)))
    b = sh.convert("RGBA"); b.putalpha(mask)
    return b.resize((CW, CH), Image.LANCZOS)
MOKO = moko_body()
MOKO_GREY = Image.blend(MOKO, Image.merge("RGBA", (*Image.new("RGB", MOKO.size, (150, 156, 172)).split(), MOKO.split()[3])), .55)
MOKO_PINK = Image.blend(MOKO, Image.merge("RGBA", (*Image.new("RGB", MOKO.size, (255, 196, 220)).split(), MOKO.split()[3])), .35)

def moko_face(anger, smile, blink, mouth, look=0.0, mud=0.0):
    img = Image.new("RGBA", (CW * S, CH * S), (0, 0, 0, 0)); d = ImageDraw.Draw(img); s = lambda v: int(v * S)
    for cx in (285, 615): d.ellipse([s(cx - 58), s(422), s(cx + 58), s(482)], fill=(150, 200, 245, int(lerp(170, 215, smile))))
    for i, cx in enumerate((345, 555)):
        ey, rx, ry, px = 365, 50, 62, look * 16
        if blink > .85: d.arc([s(cx - rx), s(ey - 20), s(cx + rx), s(ey + 24)], 200, 340, fill=INK, width=s(9))
        else:
            r2 = ry * (1 - blink)
            d.ellipse([s(cx - rx), s(ey - r2), s(cx + rx), s(ey + r2)], fill=INK)
            d.ellipse([s(cx - 22 + px), s(ey - r2 * .62), s(cx + 6 + px), s(ey - r2 * .62 + 28)], fill=(255, 255, 255, 255))
            d.ellipse([s(cx + 12 + px), s(ey + r2 * .25), s(cx + 26 + px), s(ey + r2 * .25 + 14)], fill=(255, 255, 255, 230))
        if anger > .02:
            inner = 1 if i == 0 else -1
            d.line([s(cx - 58 * inner), s(288), s(cx + 40 * inner), s(288 + 34 * anger)], fill=(*INK[:3], int(255 * min(1, anger * 1.5))), width=s(14))
    mx, my, mw = 450, 470, 66
    curve = lerp(-26, 30, smile)
    top = [(s(mx - mw + 2 * mw * k / 20), s(my + curve * (1 - (2 * k / 20 - 1) ** 2) * (.35 if mouth > .1 else 1))) for k in range(21)]
    if mouth > .1:
        dep = 18 + 46 * mouth
        bot = [(s(mx - mw * .8 + 1.6 * mw * k / 20), s(my + 6 + dep * math.sin(math.pi * k / 20) + max(curve, 0) * .4)) for k in range(21)]
        d.polygon(top + bot[::-1], fill=(120, 56, 78, 255))
        ty = my + 6 + dep * .75
        d.ellipse([s(mx - 26), s(ty - 12), s(mx + 26), s(ty + 10)], fill=(232, 120, 140, 255))
        d.line(top, fill=INK, width=s(8), joint="curve")
    else:
        d.line(top, fill=INK, width=s(10), joint="curve")
    if mud > .02:  # mud spots on the little arms
        for (x, y, r) in [(60, 420, 22), (90, 455, 15), (50, 460, 12), (840, 420, 22), (810, 452, 15), (850, 462, 12)]:
            d.ellipse([s(x - r), s(y - r), s(x + r), s(y + r)], fill=(130, 92, 60, int(235 * mud)))
    return img.resize((CW, CH), Image.LANCZOS)

DW, DH = 260, 330
def drip_img(happy, sad, mouth, blink, tear):
    im = Image.new("RGBA", (DW * S, DH * S), (0, 0, 0, 0)); s = lambda v: int(v * S)
    cx, cy, r = 130, 215, 95
    mask = Image.new("L", im.size, 0); md = ImageDraw.Draw(mask)
    md.ellipse([s(cx - r), s(cy - r), s(cx + r), s(cy + r)], fill=255)
    md.polygon([(s(cx - r * .86), s(cy - r * .5)), (s(cx), s(cy - r - 120)), (s(cx + r * .86), s(cy - r * .5))], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(3))
    g = Image.new("RGB", im.size); gd = ImageDraw.Draw(g)
    for y in range(im.size[1]): gd.line([(0, y), (im.size[0], y)], fill=lerpc((186, 226, 255), (92, 166, 236), y / im.size[1]))
    b = g.convert("RGBA"); b.putalpha(mask); im.alpha_composite(b); d = ImageDraw.Draw(im)
    d.ellipse([s(cx - 60), s(cy - 110), s(cx - 30), s(cy - 50)], fill=(255, 255, 255, 200))
    for fx in (cx - 34, cx + 34): d.ellipse([s(fx - 22), s(cy + r - 14), s(fx + 22), s(cy + r + 12)], fill=(80, 150, 225, 255))
    for ex in (cx - 34, cx + 34):
        ey = cy - 10
        if blink > .85: d.arc([s(ex - 20), s(ey - 8), s(ex + 20), s(ey + 12)], 200, 340, fill=INK, width=s(6))
        else:
            hg = 30 * (1 - blink)
            d.ellipse([s(ex - 22), s(ey - hg), s(ex + 22), s(ey + hg)], fill=INK)
            d.ellipse([s(ex - 12), s(ey - hg * .7), s(ex + 2), s(ey - hg * .7 + 14)], fill=(255, 255, 255, 255))
        if sad > .05:
            inner = 1 if ex > cx else -1
            d.line([s(ex - 16 * inner), s(ey - 44 + 10 * sad), s(ex + 14 * inner), s(ey - 44 - 6 * sad)], fill=(*INK[:3], int(255 * sad)), width=s(6))
    for ex in (cx - 58, cx + 58): d.ellipse([s(ex - 16), s(cy + 22), s(ex + 16), s(cy + 40)], fill=(255, 170, 190, 150))
    mx, my = cx, cy + 40
    if mouth > .1: d.ellipse([s(mx - 16), s(my - 4), s(mx + 16), s(my + 8 + 22 * mouth)], fill=(120, 56, 78, 255))
    else:
        cv = lerp(-12, 14, happy) * (1 - sad) - 12 * sad
        d.line([(s(mx - 22 + 44 * k / 10), s(my + cv * (1 - (2 * k / 10 - 1) ** 2))) for k in range(11)], fill=INK, width=s(5), joint="curve")
    if tear > .05:
        ty = cy + 8 + tear * 50
        d.ellipse([s(cx + 43), s(ty - 12), s(cx + 61), s(ty + 12)], fill=(150, 210, 255, int(230 * min(1, tear * 2))))
    return im.resize((DW, DH), Image.LANCZOS)

SW = 420
def sunny_img(rot, mouth, blink):
    im = Image.new("RGBA", (SW * S, SW * S), (0, 0, 0, 0)); d = ImageDraw.Draw(im); s = lambda v: int(v * S); c = SW / 2
    for k in range(12):
        a = rot + k * math.pi / 6; x, y = c + math.cos(a) * 160, c + math.sin(a) * 160
        d.ellipse([s(x - 30), s(y - 30), s(x + 30), s(y + 30)], fill=(255, 205, 80, 255))
    d.ellipse([s(c - 125), s(c - 125), s(c + 125), s(c + 125)], fill=(255, 224, 102, 255))
    d.ellipse([s(c - 92), s(c - 112), s(c - 30), s(c - 70)], fill=(255, 242, 180, 170))
    for ex in (c - 42, c + 42):
        ey = c - 14
        if blink > .85: d.arc([s(ex - 18), s(ey - 8), s(ex + 18), s(ey + 10)], 200, 340, fill=INK, width=s(6))
        else:
            d.ellipse([s(ex - 18), s(ey - 24), s(ex + 18), s(ey + 24)], fill=INK)
            d.ellipse([s(ex - 9), s(ey - 17), s(ex + 3), s(ey - 5)], fill=(255, 255, 255, 255))
    for ex in (c - 72, c + 72): d.ellipse([s(ex - 20), s(c + 18), s(ex + 20), s(c + 40)], fill=(255, 150, 120, 160))
    for fx, fy in ((c - 70, c + 6), (c - 58, c + 14), (c + 64, c + 8), (c + 76, c + 16)):
        d.ellipse([s(fx - 3), s(fy - 3), s(fx + 3), s(fy + 3)], fill=(230, 140, 60, 255))
    my = c + 34
    if mouth > .1:
        dep = 14 + 30 * mouth
        d.chord([s(c - 40), s(my - dep * .4), s(c + 40), s(my + dep)], 0, 180, fill=(140, 60, 60, 255))
    else: d.arc([s(c - 40), s(my - 30), s(c + 40), s(my + 20)], 20, 160, fill=INK, width=s(7))
    return im.resize((SW, SW), Image.LANCZOS)

WW = 380
def wind_img(t, mouth, blink):
    im = Image.new("RGBA", (WW * S, WW * S), (0, 0, 0, 0)); d = ImageDraw.Draw(im); s = lambda v: int(v * S); c = WW / 2
    # soft swirl body
    body = Image.new("RGBA", im.size, (0, 0, 0, 0)); bd = ImageDraw.Draw(body)
    bd.ellipse([s(c - 130), s(c - 110), s(c + 130), s(c + 120)], fill=(196, 240, 222, 235))
    for k in range(3):
        a0 = t * 40 + k * 120
        bd.arc([s(c - 160 + k * 8), s(c - 140 + k * 8), s(c + 160 - k * 8), s(c + 150 - k * 8)], a0, a0 + 120, fill=(170, 226, 206, 220), width=s(18))
    body = body.filter(ImageFilter.GaussianBlur(4)); im.alpha_composite(body); d = ImageDraw.Draw(im)
    for ex in (c - 45, c + 45):  # glasses + eyes
        ey = c - 20
        d.ellipse([s(ex - 30), s(ey - 30), s(ex + 30), s(ey + 30)], outline=(90, 90, 110, 255), width=s(5))
        if blink > .85: d.arc([s(ex - 14), s(ey - 6), s(ex + 14), s(ey + 8)], 200, 340, fill=INK, width=s(5))
        else:
            d.ellipse([s(ex - 12), s(ey - 15), s(ex + 12), s(ey + 15)], fill=INK)
            d.ellipse([s(ex - 6), s(ey - 11), s(ex + 2), s(ey - 3)], fill=(255, 255, 255, 255))
    d.line([s(c - 15), s(c - 20), s(c + 15), s(c - 20)], fill=(90, 90, 110, 255), width=s(5))
    # mustache
    for sgn in (-1, 1):
        d.chord([s(c + sgn * 8 - (70 if sgn < 0 else 0)), s(c + 18), s(c + sgn * 8 + (70 if sgn > 0 else 0)), s(c + 62)], 180, 360, fill=(255, 255, 255, 255))
    my = c + 62
    if mouth > .1: d.ellipse([s(c - 18), s(my - 4), s(c + 18), s(my + 8 + 18 * mouth)], fill=(120, 60, 70, 255))
    else: d.arc([s(c - 22), s(my - 14), s(c + 22), s(my + 10)], 20, 160, fill=INK, width=s(5))
    for ex in (c - 85, c + 85): d.ellipse([s(ex - 18), s(c + 18), s(ex + 18), s(c + 36)], fill=(255, 170, 170, 140))
    return im.resize((WW, WW), Image.LANCZOS)

# ---------------- props ----------------
PR = 70
def prop_img(name, t=0.0):
    sz = PR * 2
    im = Image.new("RGBA", (sz * S, sz * S), (0, 0, 0, 0)); d = ImageDraw.Draw(im); s = lambda v: int(v * S); c = PR
    if name == "ball":
        cols = [(255, 110, 110), (255, 175, 90), (255, 225, 100), (120, 210, 130), (100, 175, 255), (165, 130, 240)]
        for k, col in enumerate(cols): d.pieslice([s(8), s(8), s(sz - 8), s(sz - 8)], t * 90 + k * 60, t * 90 + (k + 1) * 60, fill=col + (255,))
    elif name == "apple":
        d.ellipse([s(18), s(30), s(c + 6), s(sz - 14)], fill=(232, 64, 70, 255)); d.ellipse([s(c - 6), s(30), s(sz - 18), s(sz - 14)], fill=(232, 64, 70, 255))
        d.line([s(c), s(36), s(c + 6), s(10)], fill=(110, 70, 40, 255), width=s(7)); d.ellipse([s(c + 6), s(8), s(c + 40), s(28)], fill=(90, 180, 90, 255))
    elif name == "banana":
        d.arc([s(14), s(-30), s(sz - 14), s(sz - 30)], 30, 150, fill=(250, 214, 70, 255), width=s(30))
    elif name == "soap":
        d.rounded_rectangle([s(16), s(48), s(sz - 16), s(sz - 30)], radius=s(22), fill=(255, 176, 210, 255))
        for bx, by, br in [(40, 36, 12), (70, 24, 16), (102, 34, 11)]:
            d.ellipse([s(bx - br), s(by - br), s(bx + br), s(by + br)], outline=(170, 210, 255, 255), width=s(4))
    elif name == "toothbrush":
        d.rounded_rectangle([s(18), s(c - 8), s(sz - 18), s(c + 8)], radius=s(8), fill=(90, 170, 255, 255))
        d.rectangle([s(sz - 52), s(c - 30), s(sz - 20), s(c - 8)], fill=(255, 255, 255, 255))
    elif name == "water":
        d.polygon([(s(34), s(20)), (s(sz - 34), s(20)), (s(sz - 44), s(sz - 12)), (s(44), s(sz - 12))], fill=(220, 240, 255, 220))
        d.polygon([(s(38), s(56)), (s(sz - 38), s(56)), (s(sz - 45), s(sz - 16)), (s(45), s(sz - 16))], fill=(110, 190, 250, 230))
    elif name == "star":
        pts = [(s(c + 60 * math.cos(math.pi / 2 + k * math.pi / 5) * (1 if k % 2 == 0 else .45)), s(c - 60 * math.sin(math.pi / 2 + k * math.pi / 5) * (1 if k % 2 == 0 else .45))) for k in range(10)]
        d.polygon(pts, fill=(255, 210, 70, 255))
    elif name == "heart":
        d.ellipse([s(18), s(28), s(c + 4), s(c + 22)], fill=(255, 100, 140, 255)); d.ellipse([s(c - 4), s(28), s(sz - 18), s(c + 22)], fill=(255, 100, 140, 255))
        d.polygon([(s(22), s(c + 4)), (s(sz - 22), s(c + 4)), (s(c), s(sz - 14))], fill=(255, 100, 140, 255))
    elif name == "flower":
        for k in range(6):
            a = k * math.pi / 3; x, y = c + 34 * math.cos(a), c - 10 + 34 * math.sin(a)
            d.ellipse([s(x - 22), s(y - 22), s(x + 22), s(y + 22)], fill=(255, 150, 200, 255))
        d.ellipse([s(c - 18), s(c - 28), s(c + 18), s(c + 8)], fill=(255, 220, 90, 255))
        d.line([s(c), s(c + 10), s(c), s(sz - 6)], fill=(90, 170, 90, 255), width=s(8))
    elif name == "umbrella":
        d.pieslice([s(10), s(20), s(sz - 10), s(sz + 40)], 180, 360, fill=(255, 120, 120, 255))
        d.line([s(c), s(c + 20), s(c), s(sz - 10)], fill=(90, 90, 110, 255), width=s(6)); d.arc([s(c - 20), s(sz - 30), s(c), s(sz - 6)], 0, 180, fill=(90, 90, 110, 255), width=s(6))
    elif name == "book":
        d.rounded_rectangle([s(16), s(30), s(sz - 16), s(sz - 24)], radius=s(8), fill=(120, 140, 240, 255))
        d.line([s(c), s(30), s(c), s(sz - 24)], fill=(255, 255, 255, 255), width=s(5))
    elif name.startswith("n") and name[1:].isdigit():
        d.ellipse([s(6), s(6), s(sz - 6), s(sz - 6)], fill=(255, 255, 255, 240), outline=(120, 170, 240, 255), width=s(6))
        f = ImageFont.truetype(FONT, s(90))
        txt = name[1:]; tw = d.textlength(txt, font=f)
        d.text((s(c) - tw / 2, s(c - 62)), txt, font=f, fill=(58, 92, 160, 255))
    else:
        d.ellipse([s(10), s(10), s(sz - 10), s(sz - 10)], fill=(255, 220, 120, 255))
    d.ellipse([s(c - 34), s(28), s(c - 10), s(46)], fill=(255, 255, 255, 110))
    return im.resize((sz, sz), Image.LANCZOS)

def make_rainbow():
    RS = 1700
    R = Image.new("RGBA", (RS * S, RS * S), (0, 0, 0, 0)); d = ImageDraw.Draw(R); c = RS * S // 2
    for i, col in enumerate([(255, 120, 120), (255, 180, 110), (255, 232, 120), (140, 220, 140), (120, 190, 255), (160, 150, 240), (210, 150, 230)]):
        r = (800 - i * 40) * S; d.ellipse([c - r, c - r, c + r, c + r], outline=(*col, 200), width=40 * S)
    return R.resize((RS, RS), Image.LANCZOS).filter(ImageFilter.GaussianBlur(3))
RAINBOW = make_rainbow()
random.seed(4)
DROPS = [(random.uniform(150, 930), random.uniform(0, 1), random.uniform(.8, 1.3)) for _ in range(46)]
FLAKES = [(random.uniform(60, 1020), random.uniform(0, 1), random.uniform(.6, 1.2), random.uniform(6, 13)) for _ in range(40)]
BUBBLES = [(random.uniform(250, 830), random.uniform(0, 1), random.uniform(.5, 1.0), random.uniform(14, 34)) for _ in range(18)]

def sparkle(d, sx, sy, r, a):
    d.polygon([(sx, sy - r * 2), (sx + r * .5, sy - r * .5), (sx + r * 2, sy), (sx + r * .5, sy + r * .5), (sx, sy + r * 2),
               (sx - r * .5, sy + r * .5), (sx - r * 2, sy), (sx - r * .5, sy - r * .5)], fill=(255, 250, 210, int(a)))

def blink_at(t, period, ph):
    x = (t + ph) % period
    return 1 - abs(x / 0.09 - 1) if x < 0.18 else 0.0

# ======================= TIMELINE =======================
class Timeline:
    def __init__(self, spec, lines, total, env):
        self.spec, self.lines, self.total, self.env = spec, lines, total, env
        L = spec["lines"]
        self.mood_keys = [(0.0, spec.get("start_mood", "calm"))]
        self.drip_keys = [(0.0, spec.get("drip_start", "happy"))]
        self.enter = {"drip": 0.0 if "drip" in spec["cast"] else None, "sunny": None, "wind": None}
        for who in ("sunny", "wind"):
            if who in spec["cast"]: self.enter[who] = 0.0
        self.props = {}       # name -> list of (t, state, holder)
        self.mud = [(0.0, 1.0 if spec.get("start_mud") else 0.0)]
        self.bubbles, self.rainbow = [], None
        self.loc_keys = [(0.0, spec.get("location", "sky"))]
        for l, (who, a, b) in zip(L, lines):
            if l.get("mood"): self.mood_keys.append((a, l["mood"]))
            if l.get("drip"): self.drip_keys.append((a, l["drip"]))
            for act in l.get("do", []):
                p = act.split(":")
                if p[0] == "enter": self.enter[p[1]] = a - 0.8
                elif p[0] == "show": self.props.setdefault(p[1], []).append((a, "show", "moko"))
                elif p[0] == "give": self.props.setdefault(p[1], []).append((a + (b - a) * .5, "give", p[2]))
                elif p[0] == "hide": self.props.setdefault(p[1], []).append((b, "hide", None))
                elif p[0] == "mud": self.mud.append((a + (b - a) * .4, 1.0 if p[1] == "on" else 0.0))
                elif p[0] == "bubbles": self.bubbles.append((a, b + 0.5))
                elif p[0] == "rainbow": self.rainbow = a - 0.2
                elif p[0] == "go": self.loc_keys.append((a - 0.3, p[1]))
            if "enter:sunny" not in l.get("do", []) and who == "sunny" and self.enter["sunny"] is None: self.enter["sunny"] = a - 0.8
            if "enter:wind" not in l.get("do", []) and who == "wind" and self.enter["wind"] is None: self.enter["wind"] = a - 0.8
        if self.rainbow is None and L and L[-1].get("mood") in ("happy", "proud", None):
            self.rainbow = lines[-1][1] - 0.2

    def mood(self, t):
        """blend of mood params at time t (0.6s crossfade)"""
        cur = self.mood_keys[0][1]; prev = cur; tc = 0
        for tk, m in self.mood_keys:
            if tk <= t: prev, cur, tc = cur, m, tk
        k = ease((t - tc) / 0.6) if tc > 0 else 1
        A, B = MOOD[prev], MOOD[cur]
        out = {key: (lerp(A[key], B[key], k) if key != "sky" else None) for key in A}
        out["sky_a"], out["sky_b"], out["k"] = A["sky"], B["sky"], k
        return out, cur

    def loc(self, t):
        """(previous location, current location, blend 0..1)"""
        prev = cur = self.loc_keys[0][1]; tc = None
        for tk, l in self.loc_keys[1:]:
            if tk <= t: prev, cur, tc = cur, l, tk
        k = 1.0 if tc is None else ease((t - tc) / 0.7)
        return prev, cur, k

    def drip_mood(self, t):
        cur = self.drip_keys[0][1]
        for tk, m in self.drip_keys:
            if tk <= t: cur = m
        return cur

    def talk(self, who, t):
        e = self.env.get(who); fi = int(t * FPS)
        return float(e[fi]) if e is not None and fi < len(e) else 0.0

    def speaker(self, t):
        for who, a, b in self.lines:
            if a <= t < b: return who
        return None

    def mud_v(self, t):
        v = self.mud[0][1]; last_t, last_v = 0, v
        for tk, val in self.mud:
            if tk <= t: last_v, v, last_t = v, val, tk
        return lerp(last_v, v, ease((t - last_t) / 0.8)) if last_t > 0 else v

POS = {"drip": (230, 1290), "sunny": (840, 300), "wind": (230, 360)}

def render_frame(tl, t):
    m, mname = tl.mood(t)
    lp, lc, lk = tl.loc(t)
    def sky_for(name, loc_name):
        if loc_name == "night" and name in ("calm", "happy"): return "night"
        return name
    sa, sb = sky_for(m["sky_a"], lc), sky_for(m["sky_b"], lc)
    img = Image.blend(SKY[sa], SKY[sb], m["k"]).copy()
    if lk < 1 and (lp == "night") != (lc == "night"):
        img = Image.blend(Image.blend(SKY[sky_for(m["sky_a"], lp)], SKY[sky_for(m["sky_b"], lp)], m["k"]), img, lk)
    d = ImageDraw.Draw(img)
    rb_a = ease((t - tl.rainbow) / 1.4) if tl.rainbow is not None else 0
    if rb_a > 0:
        rb = RAINBOW.copy(); rb.putalpha(rb.split()[3].point(lambda v: int(v * rb_a)))
        img.alpha_composite(rb, (540 - 850, 1660 - 850))
    gk = min(1, m["grey"])
    cc, cg = location(lc)
    ground = Image.blend(cc, cg, gk) if gk > 0.01 else cc
    if lk < 1:
        pc, pg = location(lp)
        ground = Image.blend(Image.blend(pc, pg, gk) if gk > 0.01 else pc, ground, lk)
    img.alpha_composite(ground)
    bob = math.sin(t * 2.1) * 16
    shake = math.sin(t * 38) * 8 * m["anger"] * 0.5
    cx, cy = W / 2 + shake, 760 + bob
    if m["rain"] > .03:
        for x, ph, sp in DROPS:
            y = 1060 + ((ph + t * .9 * sp) % 1) * 640
            d.line([x + shake, y, x + shake - 6, y + 40], fill=(170, 200, 235, int(210 * m["rain"])), width=7)
    if m["snow"] > .03:
        for x, ph, sp, r in FLAKES:
            y = ((ph + t * .12 * sp) % 1) * 1700; xx = x + math.sin(t * 2 + ph * 9) * 20
            sparkle(d, xx, y, r * .6, 230 * m["snow"])
    if m["rays"] > .03:
        for k in range(12):
            a = t * .3 + k * math.pi / 6
            d.line([cx, cy, cx + math.cos(a) * 700, cy + math.sin(a) * 700], fill=(255, 240, 170, int(90 * m["rays"])), width=26)
    sc = 0.86 * m["shrink"]
    body = Image.blend(MOKO, MOKO_GREY, min(1, m["grey"]))
    if m["pink"] > .02: body = Image.blend(body, MOKO_PINK, m["pink"])
    bw, bh = int(CW * sc), int(CH * sc)
    img.alpha_composite(body.resize((bw, bh), Image.LANCZOS), (int(cx - bw / 2), int(cy - bh / 2)))
    spk = tl.speaker(t)
    look = 0.0
    if spk == "drip": look = -0.8
    elif spk in ("sunny",): look = 0.8
    elif spk == "wind": look = -0.6
    f = moko_face(m["anger"], m["smile"], blink_at(t, 4.3, 1.1), tl.talk("moko", t), look, tl.mud_v(t))
    img.alpha_composite(f.resize((bw, bh), Image.LANCZOS), (int(cx - bw / 2), int(cy - bh / 2)))
    if m["fog"] > .03:
        fog = Image.new("RGBA", (W, H), (220, 224, 232, int(90 * m["fog"]))); img.alpha_composite(fog)
    # friends
    holders = {"moko": (cx + sc * 380, cy + sc * 70)}
    if tl.enter["drip"] is not None:
        dx = lerp(-200, POS["drip"][0], ease((t - tl.enter["drip"]) / 0.9)); dy = POS["drip"][1] + math.sin(t * 2.6 + 1) * 10
        dm = tl.drip_mood(t)
        if dm == "happy" and m["smile"] > .8: dy -= abs(math.sin(t * 5.5)) * 30
        tear = ((t * 0.62) % 1) if dm == "sad" else 0
        img.alpha_composite(drip_img(1.0 if dm == "happy" else .4, 1.0 if dm == "sad" else 0, tl.talk("drip", t), blink_at(t, 3.7, .3), tear), (int(dx - DW / 2), int(dy - DH / 2)))
        holders["drip"] = (dx + 120, dy - 150)
    if tl.enter["sunny"] is not None:
        p = ease((t - tl.enter["sunny"]) / 1.0)
        if p > 0:
            sx = lerp(W + 260, POS["sunny"][0], p); sy = lerp(-200, POS["sunny"][1], p) + math.sin(t * 1.8) * 12
            img.alpha_composite(sunny_img(t * .5, tl.talk("sunny", t), blink_at(t, 5.1, 2.0)), (int(sx - SW / 2), int(sy - SW / 2)))
            holders["sunny"] = (sx - 150, sy + 170)
    if tl.enter["wind"] is not None:
        p = ease((t - tl.enter["wind"]) / 1.0)
        if p > 0:
            wx = lerp(-260, POS["wind"][0], p); wy = POS["wind"][1] + math.sin(t * 1.5) * 14
            wi = wind_img(t, tl.talk("wind", t), blink_at(t, 4.7, 1.5)); wi.putalpha(wi.split()[3].point(lambda v: int(v * p)))
            img.alpha_composite(wi, (int(wx - WW / 2), int(wy - WW / 2)))
            holders["wind"] = (wx + 160, wy + 160)
    # props
    for name, evs in tl.props.items():
        state, pos, t0, frm = None, None, 0, None
        for te, st, hold in sorted(evs):
            if te <= t:
                if st == "show": state, pos = "held", hold
                elif st == "give": frm, state, pos, t0 = pos, "moving", hold, te
                elif st == "hide": state = None
        if state is None or pos not in holders: continue
        if state == "moving":
            p = ease((t - t0) / 1.0); a = holders.get(frm, holders["moko"]); b = holders[pos]
            px, py = lerp(a[0], b[0], p), lerp(a[1], b[1], p) + math.sin(p * math.pi) * 90
        else:
            px, py = holders[pos]
            py += math.sin(t * 3) * 6
        appear = min(1, (t - min(e[0] for e in evs)) / 0.35)
        pi_ = prop_img(name, t if name == "ball" else 0)
        if appear < 1: pi_.putalpha(pi_.split()[3].point(lambda v: int(v * appear)))
        img.alpha_composite(pi_, (int(px - PR), int(py - PR)))
    for a, b in tl.bubbles:
        if a <= t < b:
            for x, ph, sp, r in BUBBLES:
                y = 1100 - ((ph + (t - a) * .35 * sp) % 1) * 700
                d.ellipse([x - r, y - r, x + r, y + r], outline=(200, 230, 255, 220), width=4)
                d.ellipse([x - r * .5, y - r * .6, x - r * .1, y - r * .25], fill=(255, 255, 255, 180))
    if rb_a > 0:
        for k, (sx, sy) in enumerate([(200, 560), (880, 620), (330, 1150), (800, 1120), (540, 380), (120, 900), (960, 950)]):
            tw = .5 + .5 * math.sin(t * 5 + k * 1.7); sparkle(d, sx, sy, 10 + 10 * tw, 230 * rb_a * tw)
    out = img.convert("RGB")
    # gentle camera: slow push-in/out per line, leaning toward whoever is speaking
    li = 0
    for i, (_, a, b) in enumerate(tl.lines):
        if t >= a: li = i
    a0 = tl.lines[li][1]; b0 = tl.lines[min(li + 1, len(tl.lines) - 1)][1] if li + 1 < len(tl.lines) else tl.total
    ph = ease((t - a0) / max(0.5, b0 - a0))
    zoom = 1.0 + (0.06 * ph if li % 2 == 0 else 0.06 * (1 - ph))
    if tl.rainbow is not None and t > tl.rainbow: zoom = 1.0 + (zoom - 1.0) * (1 - ease((t - tl.rainbow) / 1.2))
    FOC = {"drip": (380, 1050), "sunny": (720, 700), "wind": (380, 700)}
    cur_f = FOC.get(tl.lines[li][0], (540, 900))
    prev_f = FOC.get(tl.lines[li - 1][0], (540, 900)) if li > 0 else cur_f
    kf = ease((t - a0) / 0.8)
    fx, fy = lerp(prev_f[0], cur_f[0], kf), lerp(prev_f[1], cur_f[1], kf)
    if zoom > 1.001:
        cw, ch = W / zoom, H / zoom
        x0 = min(max(fx - cw / 2, 0), W - cw); y0 = min(max(fy - ch / 2, 0), H - ch)
        out = out.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + cw, y0 + ch))
    fade = min(ease(t / .4), ease((tl.total - t) / .6))
    if fade < 1: out = Image.blend(Image.new("RGB", out.size, (255, 255, 255)), out, fade)
    return out

def make(ep_dir, preview_only=False):
    spec = json.load(open(os.path.join(ep_dir, "spec.json")))
    lines, total, env = build_audio(ep_dir, spec)
    json.dump({"lines": lines, "total": total}, open(os.path.join(ep_dir, "timing.json"), "w"), indent=1)
    tl = Timeline(spec, lines, total, env)
    # contact sheet of 6 moments for checking
    ts = [lines[i][1] + 0.8 for i in range(len(lines))][:8]
    thumbs = [render_frame(tl, x).resize((270, 480)) for x in ts]
    sheet = Image.new("RGB", (270 * len(thumbs), 480), "white")
    for i, th in enumerate(thumbs): sheet.paste(th, (i * 270, 0))
    sheet.save(os.path.join(ep_dir, "preview.png"))
    if preview_only: return
    vid = os.path.join(ep_dir, "video.mp4"); tmp = vid[:-4] + ".part.mp4"
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    for i in range(int(total * FPS)): p.stdin.write(render_frame(tl, i / FPS).tobytes())
    p.stdin.close()
    if p.wait() != 0: raise RuntimeError("video encode failed")
    os.replace(tmp, vid)
    fin = os.path.join(ep_dir, "final.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", vid, "-i", os.path.join(ep_dir, "audio.wav"), "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", fin + ".part.mp4"], check=True)
    os.replace(fin + ".part.mp4", fin)
    print("done", fin, total)

if __name__ == "__main__":
    make(sys.argv[1], preview_only=len(sys.argv) > 2 and sys.argv[2] == "preview")
