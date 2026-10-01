#!/usr/bin/env python3
from html import escape
from pathlib import Path


OUT = Path("/home/zyt/deep_final/artifacts/model_architecture_diagrams_20260921")
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "input": "#dcebf9", "encoder": "#e9e1f7", "fusion": "#ddf1e5",
    "aggregate": "#fceacf", "head": "#dde8f4", "str": "#faddd8",
    "edge": "#334155", "text": "#172033", "muted": "#536174",
}


def box(x, y, w, h, title, lines, fill, accent=False):
    stroke = "#b6463a" if accent else COLORS["edge"]
    result = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" '
        f'fill="{COLORS[fill]}" stroke="{stroke}" stroke-width="{4 if accent else 2.5}"/>',
        f'<text x="{x+w/2}" y="{y+35}" class="title" text-anchor="middle">{escape(title)}</text>',
    ]
    start = y + 68
    spacing = min(29, (h - 80) / max(1, len(lines) - 0.25))
    for index, line in enumerate(lines):
        result.append(
            f'<text x="{x+w/2}" y="{start+index*spacing:.1f}" class="body" '
            f'text-anchor="middle">{escape(line)}</text>'
        )
    return "\n".join(result)


def arrow(x1, y1, x2, y2, label="", red=False, curve=0):
    color = "#b6463a" if red else COLORS["edge"]
    if curve:
        cx = (x1 + x2) / 2 + curve
        path = f"M {x1} {y1} Q {cx} {(y1+y2)/2} {x2} {y2}"
    else:
        path = f"M {x1} {y1} L {x2} {y2}"
    result = [
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3" '
        f'marker-end="url(#{"arrow-red" if red else "arrow"})"/>'
    ]
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 - 10
        result.extend([
            f'<rect x="{mx-72}" y="{my-17}" width="144" height="24" rx="5" fill="white" opacity="0.94"/>',
            f'<text x="{mx}" y="{my}" class="label" text-anchor="middle">{escape(label)}</text>',
        ])
    return "\n".join(result)


def shell(title, subtitle, body, notes):
    note_svg = []
    for index, note in enumerate(notes):
        note_svg.append(
            f'<text x="800" y="{925 + index*29}" class="note" text-anchor="middle">{escape(note)}</text>'
        )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{COLORS['edge']}"/></marker>
  <marker id="arrow-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#b6463a"/></marker>
  <style>
    .heading {{ font: 700 34px Arial, Helvetica, sans-serif; fill: {COLORS['text']}; }}
    .subtitle {{ font: 400 19px Arial, Helvetica, sans-serif; fill: {COLORS['muted']}; }}
    .title {{ font: 700 19px Arial, Helvetica, sans-serif; fill: {COLORS['text']}; }}
    .body {{ font: 400 15px Arial, Helvetica, sans-serif; fill: {COLORS['text']}; }}
    .label {{ font: 400 14px Arial, Helvetica, sans-serif; fill: {COLORS['muted']}; }}
    .note {{ font: 400 16px Arial, Helvetica, sans-serif; fill: {COLORS['muted']}; }}
  </style>
</defs>
<rect width="1600" height="1000" fill="white"/>
<text x="800" y="52" class="heading" text-anchor="middle">{escape(title)}</text>
<text x="800" y="84" class="subtitle" text-anchor="middle">{escape(subtitle)}</text>
{body}
{''.join(note_svg)}
</svg>'''


def frontend():
    parts = [
        box(35, 370, 205, 235, "Subject input", [
            "11 fixed activities", "Left + right wrist", "Acc XYZ + Gyro XYZ",
            "[B, 11, 2, 6, Tₐ]",
        ], "input"),
        box(285, 145, 300, 315, "Temporal path (shared)", [
            "Conv1D 6→64, k=15, s=2", "GroupNorm(8) + GELU",
            "3× separable residual blocks", "k=7; dilation 1 / 2 / 4",
            "GN + GELU + dropout 0.1", "attention + mean + std pooling",
            "Linear 192→64 + LN + GELU",
        ], "encoder"),
        box(285, 535, 300, 315, "Learned moment path (shared)", [
            "Parallel learned Conv1D", "kernels 1 / 15 / 63",
            "concat → Conv1D 48→32, k=1", "no pre-pooling normalization",
            "global mean + std", "Linear 64→32 + LN + GELU",
            "no handcrafted statistics",
        ], "encoder"),
        box(635, 370, 245, 235, "Wrist embedding", [
            "concat temporal 64", "+ moment 32", "Linear 96→64",
            "+ LN + GELU", "64D per wrist", "shared L/R weights",
        ], "fusion"),
        box(930, 345, 250, 285, "Full bilateral fusion", [
            "left 64D", "right 64D", "masked mean 64D", "|left − right| 64D",
            "wrist-valid mask 2D", "→ 258D per activity",
        ], "fusion"),
        arrow(240, 455, 285, 300, "each wrist"),
        arrow(240, 535, 285, 690),
        arrow(585, 300, 635, 445, "64D"),
        arrow(585, 690, 635, 545, "32D"),
        arrow(880, 487, 930, 487, "L/R × 64D"),
    ]
    return "\n".join(parts)


def v8():
    body = [frontend()]
    body.extend([
        box(1230, 205, 330, 365, "Activity Attention", [
            "11 × 258D activity tokens", "+ learned activity-ID embedding",
            "LayerNorm", "score: Linear 258→64", "Tanh + dropout 0.1 → Linear 1",
            "masked softmax over activities", "weighted sum → subject 258D",
        ], "aggregate"),
        box(1250, 650, 290, 185, "Original classifier", [
            "Dropout 0.2", "Linear 258→2", "PD / DD logits", "softmax probability",
        ], "head"),
        arrow(1180, 487, 1230, 390, "11 × 258D"),
        arrow(1395, 570, 1395, 650, "subject 258D"),
    ])
    return shell(
        "V8-GN Architecture",
        "Pure-deep bilateral multi-activity classifier · 71,026 trainable parameters",
        "\n".join(body),
        [
            "Shared wrist/activity encoders; masks handle unavailable wrists and activities.",
            "Raw normalized Acc/Gyro only—no H1 features, FFT/STFT branch, Transformer, or domain-alignment module.",
        ],
    )


def str01():
    body = [frontend()]
    body.extend([
        box(1225, 130, 340, 315, "Original V8 main path", [
            "11 × 258D activity tokens", "+ activity-ID embedding + LN",
            "Activity Attention 258→64→1", "masked weighted sum",
            "subject embedding 258D", "Dropout 0.2 + Linear 258→2",
            "base logits",
        ], "aggregate"),
        box(1225, 505, 340, 330, "STR residual path", [
            "from pre-aggregation activity tokens", "shared Linear 258→16 + GELU",
            "mask unavailable activities", "keep fixed activity identity/order",
            "11 × 16 → flatten 176D", "Linear 176→2 (zero-initialized)",
            "residual logits",
        ], "str", accent=True),
        box(835, 730, 320, 145, "Residual decision", [
            "final logits = base logits + residual logits", "softmax → PD / DD probability",
        ], "head"),
        arrow(1180, 455, 1225, 300),
        arrow(1180, 520, 1225, 650, red=True),
        arrow(1225, 405, 1080, 730, curve=-55),
        arrow(1225, 720, 1155, 795, red=True),
    ])
    return shell(
        "STR-01 Architecture",
        "V8-GN main path + Structured Token Residual Readout · 75,524 parameters (+4,498)",
        "\n".join(body),
        [
            "The original Activity Attention, subject embedding, and classifier remain unchanged; only the red path is added.",
            "The residual path consumes learned 258D activity representations—not H1 features—and is trained end to end.",
        ],
    )


(OUT / "v8_gn_architecture.svg").write_text(v8(), encoding="utf-8")
(OUT / "str01_architecture.svg").write_text(str01(), encoding="utf-8")
print(OUT)
