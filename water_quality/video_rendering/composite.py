import json, os
from PIL import Image

meta = json.load(open('./gp_frame_index.json'))
meta.sort(key=lambda m: m[0])
meta_t = [m[0] for m in meta]
meta_path = [m[1] for m in meta]

import bisect

def panel_for(video_t):
    i = bisect.bisect_right(meta_t, video_t) - 1
    i = max(0, min(i, len(meta_path)-1))
    return meta_path[i]

FPS = 15.0
BOX = (562, 645, 1122, 1000)  # x0,y0,x1,y1

indir = './allframes'
outdir = './composited'
os.makedirs(outdir, exist_ok=True)

files = sorted(os.listdir(indir))
print("total frames:", len(files))

panel_cache = {}
def get_panel(path):
    if path not in panel_cache:
        panel_cache[path] = Image.open(path).convert('RGB')
    return panel_cache[path]

for idx, fn in enumerate(files):
    frame_idx = idx  # 0-based, matches ffmpeg -vsync 0 order
    video_t = frame_idx / FPS
    panel_path = panel_for(video_t)
    panel_img = get_panel(panel_path)
    frame = Image.open(os.path.join(indir, fn)).convert('RGB')
    frame.paste(panel_img, (BOX[0], BOX[1]))
    frame.save(os.path.join(outdir, fn))
    if idx % 300 == 0:
        print("composited", idx, "/", len(files), "video_t=", round(video_t,2), "panel=", os.path.basename(panel_path))

print("done compositing")
