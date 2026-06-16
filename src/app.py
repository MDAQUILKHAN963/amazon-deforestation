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

    head = "## 🟥 Deforestation detected" if pct > D.VERDICT_PCT else "## 🟩 No significant deforestation"
    info = (f"{head}\n\n"
            f"**Model prediction:** {pct:.1f}% of this area deforested  \n"
            f"**Actual (labeled):** {actual:.1f}%  \n"
            f"**Nearest tile:** ({rows[idx]['lat']}, {rows[idx]['lon']}) · "
            f"{rows[idx]['mask_date'][:7]} · ~{km*111:.1f} km from your point")

    rgb = np.clip(np.stack([img[2], img[1], img[0]], -1) * 3.5, 0, 1)
    fig, ax = plt.subplots(1, 3, figsize=(11, 4))
    ax[0].imshow(rgb);                                 ax[0].set_title("Satellite (RGB)")
    ax[1].imshow(truth, cmap="Reds", vmin=0, vmax=1);  ax[1].set_title("Actual deforestation")
    ax[2].imshow(pred,  cmap="Reds", vmin=0, vmax=1);  ax[2].set_title(f"Model prediction ({pct:.0f}%)")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    return info, fig


def random_location():
    i = int(np.random.randint(len(rows)))
    return rows[i]["lat"], rows[i]["lon"]


with gr.Blocks(title="Amazon Deforestation Detector") as app:
    gr.Markdown(
        "# 🌳 Amazon Deforestation Detector\n"
        "Pick a location in the Amazon study region and a year. The model reads the "
        "satellite imagery for that spot and highlights deforested areas, then compares "
        "its prediction to the actual labeled data."
    )
    with gr.Row():
        lat = gr.Slider(LAT_MIN, LAT_MAX, value=round((LAT_MIN + LAT_MAX) / 2, 2), label="Latitude")
        lon = gr.Slider(LON_MIN, LON_MAX, value=round((LON_MIN + LON_MAX) / 2, 2), label="Longitude")
        year = gr.Dropdown(["2019", "2020", "2021"], value="2021", label="Year")
    with gr.Row():
        go = gr.Button("🔍 Analyze", variant="primary")
        rnd = gr.Button("🎲 Random location")
    verdict = gr.Markdown()
    picture = gr.Plot()

    go.click(analyze, [lat, lon, year], [verdict, picture])
    rnd.click(random_location, None, [lat, lon])

if __name__ == "__main__":
    app.launch(share=True)
