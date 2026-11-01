"""Join several Moko episodes into one landscape (16:9) compilation video.

python3 compile_week.py out.mp4 "Compilation title (Hindi)" ep_a.mp4 ep_b.mp4 ...

Each vertical episode sits in the centre of a Skyville background with the
channel name on one side and Moko on the other. A 4 s title card opens the
video and a 4 s goodbye card closes it. Audio is levelled to -16 LUFS.
"""
import os, sys, subprocess, tempfile, math
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moko_engine as E

W, H, FPS = 1920, 1080, 30
POP = "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf"
POPM = "/usr/share/fonts/truetype/google-fonts/Poppins-Medium.ttf"

def font(path, size):
    return ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)

def sky_bg():
    g = Image.new("RGB", (1, H))
    for y in range(H):
        g.putpixel((0, y), E.lerpc((110, 186, 246), (214, 238, 255), y / (H - 1)))
    bg = g.resize((W, H)).convert("RGBA")
    d = ImageDraw.Draw(bg)
    for col, y0, amp, sd in (((150, 214, 140), 900, 30, 1.0), ((118, 194, 118), 970, 22, 2.7)):
        d.polygon([(0, H)] + [(x, y0 + amp * math.sin(x / 260 + sd)) for x in range(0, W + 40, 40)] + [(W, H)], fill=col + (255,))
    return bg

def moko_img(scale, smile=1.0):
    b = E.MOKO.copy()
    b.alpha_composite(E.moko_face(0, smile, 0, 0))
    return b.resize((int(E.CW * scale), int(E.CH * scale)), Image.LANCZOS)

def outline_text(d, xy, text, f, fill, anchor="mm"):
    x, y = xy
    for dx in range(-5, 6, 2):
        for dy in range(-5, 6, 2):
            d.text((x + dx, y + dy), text, font=f, fill=(255, 255, 255, 255), anchor=anchor)
    d.text((x, y), text, font=f, fill=fill, anchor=anchor)

def frame_bg(path):
    bg = sky_bg()
    d = ImageDraw.Draw(bg)
    # left: channel name, right: Moko
    outline_text(d, (330, 430), "मोको", font(POP, 120), (58, 92, 160, 255))
    outline_text(d, (330, 560), "छोटा बादल", font(POP, 72), (58, 92, 160, 255))
    outline_text(d, (330, 660), "रोज़ एक नई सीख", font(POPM, 40), (70, 90, 120, 255))
    m = moko_img(0.42)
    bg.alpha_composite(m, (1590 - m.width // 2, 470 - m.height // 2))
    # soft frame where the episode will sit
    d.rounded_rectangle([W // 2 - 318, 6, W // 2 + 318, H - 6], radius=26, fill=(255, 255, 255, 120))
    bg.convert("RGB").save(path)

def card(path, big, small):
    bg = sky_bg()
    rb = E.RAINBOW.resize((1500, 1500))
    bg.alpha_composite(rb, (W // 2 - 750, 360))
    d = ImageDraw.Draw(bg)
    m = moko_img(0.6)
    bg.alpha_composite(m, (W // 2 - m.width // 2, 120))
    outline_text(d, (W // 2, 640), big, font(POP, 84), (58, 92, 160, 255))
    outline_text(d, (W // 2, 750), small, font(POPM, 44), (70, 90, 120, 255))
    bg.convert("RGB").save(path)

def run(cmd):
    subprocess.run(cmd, check=True)

def main(out, title, clips):
    tmp = tempfile.mkdtemp()
    bgp, introp, outrop = (os.path.join(tmp, n) for n in ("bg.png", "intro.png", "outro.png"))
    frame_bg(bgp)
    card(introp, title, "मोको – छोटा बादल")
    card(outrop, "फिर मिलेंगे!", "मन बदला, मौसम बदला!")
    parts = []
    enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS),
           "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2"]
    for i, (img, dur) in enumerate(((introp, 4), (outrop, 4))):
        p = os.path.join(tmp, f"card{i}.mp4")
        run(["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS), "-i", img, "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
             "-vf", f"format=yuv420p,loop=-1:1,trim=duration={dur},fade=t=in:st=0:d=0.5,fade=t=out:st={dur - 0.5}:d=0.5",
             "-t", str(dur)] + enc + [p])
        parts.append(p)
    intro, outro = parts
    body = []
    for i, c in enumerate(clips):
        p = os.path.join(tmp, f"ep{i}.mp4")
        run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", bgp, "-i", c,
             "-filter_complex", "[1:v]scale=-2:1068[v];[0:v][v]overlay=(W-w)/2:6:shortest=1,format=yuv420p[o];[1:a]loudnorm=I=-16:TP=-1.5:LRA=11[a]",
             "-map", "[o]", "-map", "[a]", "-shortest"] + enc + [p])
        body.append(p)
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w") as f:
        for p in [intro] + body + [outro]:
            f.write(f"file '{p}'\n")
    part = out[:-4] + ".part.mp4"
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart", part])
    os.replace(part, out)
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", out], capture_output=True, text=True).stdout.strip()
    print("done", out, dur)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
