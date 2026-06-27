"""
app.py — Gradio web UI for the deforestation model.

Pick a location (and year) in the study region; the model predicts deforestation
from the satellite tile and shows satellite / actual / predicted side by side.
Launches with a public share link so anyone (e.g. an interviewer) can try it.

Run (from src/, with DEFOR_DATA / DEFOR_OUT set):
    python app.py
or in Colab:
    import app; app.app.launch(share=True)
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gradio as gr

import demo as D

# load model + data once at startup
X, Y, rows, MEAN, STD, MODEL, DEVICE = D._load()
LATS = [r["lat"] for r in rows]; LONS = [r["lon"] for r in rows]
LAT_MIN, LAT_MAX = min(LATS), max(LATS)
LON_MIN, LON_MAX = min(LONS), max(LONS)


def analyze(lat, lon, year):
    idx, km = D.find_tile(rows, lat, lon, int(year))
    pred = D.predict(idx, X, MEAN, STD, MODEL, DEVICE)
    truth = np.asarray(Y[idx])
    img = np.asarray(X[idx], dtype=np.float32)
    pct = pred.mean() * 100
    actual = truth.mean() * 100

    detected = pct > D.VERDICT_PCT
    color = "#c62828" if detected else "#2e7d32"
    label = "Deforestation detected" if detected else "No significant deforestation"
    emoji = "🟥" if detected else "🟩"
    info = f"""
    <div style="border-radius:14px;padding:18px 22px;background:{color}12;
                border:1px solid {color}55;">
      <div style="font-size:22px;font-weight:700;color:{color};">{emoji} {label}</div>
      <div style="margin-top:12px;display:flex;gap:26px;flex-wrap:wrap;">
        <div><div style="font-size:28px;font-weight:700;color:{color};">{pct:.1f}%</div>
             <div style="color:#666;font-size:13px;">model prediction</div></div>
        <div><div style="font-size:28px;font-weight:700;color:#444;">{actual:.1f}%</div>
             <div style="color:#666;font-size:13px;">actual (labeled)</div></div>
      </div>
      <div style="margin-top:12px;color:#666;font-size:13px;">
        Nearest tile: ({rows[idx]['lat']}, {rows[idx]['lon']}) ·
        {rows[idx]['mask_date'][:7]} · ~{km*111:.1f} km from your point ·
        cloud {rows[idx]['cloud_frac']*100:.0f}%
      </div>
    </div>"""

    rgb = np.clip(np.stack([img[2], img[1], img[0]], -1) * 3.5, 0, 1)
    fig, ax = plt.subplots(1, 3, figsize=(11, 4.2))
    fig.patch.set_facecolor("white")
    titles = ["Satellite (RGB)", "Actual deforestation", f"Model prediction ({pct:.0f}%)"]
    imgs = [rgb, truth, pred]
    for a, im, t, c in zip(ax, imgs, titles, ["#333", "#c62828", "#c62828"]):
        a.imshow(im) if im is rgb else a.imshow(im, cmap="Reds", vmin=0, vmax=1)
        a.set_title(t, fontsize=12, fontweight="bold", color=c)
        a.axis("off")
    plt.tight_layout()
    return info, fig


def random_location():
    # pick a clear tile that actually has some deforestation -> a meaningful demo
    good = [i for i in range(len(rows))
            if rows[i]["cloud_frac"] <= 0.35 and rows[i]["pos_frac"] > 0.05]
    if not good:
        good = list(range(len(rows)))
    i = int(np.random.choice(good))
    return rows[i]["lat"], rows[i]["lon"]


CSS = """
.gradio-container {max-width: 1150px !important; margin: auto;}
#banner {background: linear-gradient(135deg,#1b5e20,#43a047); border-radius:16px;
         padding:26px 30px; color:white; margin-bottom:8px;}
#banner h1 {margin:0; font-size:30px;}
#banner p {margin:6px 0 0; font-size:15px; opacity:.92;}
.card {border:1px solid #e0e6e0; border-radius:14px; padding:16px; background:#fafdfa;}
footer {visibility:hidden;}
"""

THEME = gr.themes.Soft(primary_hue="green", secondary_hue="green",
                       font=[gr.themes.GoogleFont("Inter"), "sans-serif"])

with gr.Blocks(title="Amazon Deforestation Detector", theme=THEME, css=CSS) as app:
    gr.HTML(
        '<div id="banner"><h1>🌳 Amazon Deforestation Detector</h1>'
        '<p>Pick a location in the Amazon study region and a year. The model reads the '
        'satellite imagery for that spot, highlights deforested areas, and compares its '
        'prediction to the real labeled data.</p></div>'
    )
    with gr.Row(equal_height=False):
        with gr.Column(scale=2, min_width=300):
            with gr.Group():
                gr.Markdown("### 📍 Choose a location")
                lat = gr.Slider(LAT_MIN, LAT_MAX, value=round((LAT_MIN + LAT_MAX) / 2, 2),
                                step=0.01, label="Latitude")
                lon = gr.Slider(LON_MIN, LON_MAX, value=round((LON_MIN + LON_MAX) / 2, 2),
                                step=0.01, label="Longitude")
                year = gr.Dropdown(["2019", "2020", "2021"], value="2021", label="Year")
            with gr.Row():
                go = gr.Button("🔍 Analyze", variant="primary", scale=2)
                rnd = gr.Button("🎲 Random", scale=1)
            gr.Markdown("**Try an example:**")
            gr.Examples(
                examples=[[-4.05, -54.90, "2021"], [-3.80, -55.00, "2020"],
                          [-4.20, -54.70, "2019"]],
                inputs=[lat, lon, year], label="",
            )
            gr.Markdown(
                "<small>🟥 red = deforested area &nbsp;|&nbsp; works for the Amazon "
                "study region (2019–2021).</small>")
        with gr.Column(scale=3, min_width=420):
            verdict = gr.HTML()
            picture = gr.Plot(show_label=False)

    go.click(analyze, [lat, lon, year], [verdict, picture])
    rnd.click(random_location, None, [lat, lon])

if __name__ == "__main__":
    app.launch(share=True)
