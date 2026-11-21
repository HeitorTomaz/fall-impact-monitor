"""Draws the README's "How it works" diagram as two SVG files (light and dark themes).

The home is drawn as a floor plan: its walls are the privacy boundary. Cameras sit in the
corners, everything is recorded and analysed inside, and only the alert leaves through
the door; a clip follows only on request. Run from the repository root:

    python docs/assets/how_it_works.py
"""

from pathlib import Path

HERE = Path(__file__).parent
WIDTH, HEIGHT = 1240, 660
FONT = "'Atkinson Hyperlegible', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"

THEMES = {
    "light": {
        "paper": "#EDF1EE",
        "ink": "#1F2A26",
        "muted": "#56645E",
        "hairline": "#A9B5AF",
        "cone": "#D7E0DB",
        "accent": "#C98200",
        "bubble": "#F7E6C4",
        "quiet": "#DDE4E0",
        "phone": "#FFFFFF",
    },
    "dark": {
        "paper": "#151B19",
        "ink": "#DCE5E0",
        "muted": "#97A79F",
        "hairline": "#4B5852",
        "cone": "#222C28",
        "accent": "#F2B33D",
        "bubble": "#3A2F17",
        "quiet": "#26302C",
        "phone": "#1D2522",
    },
}

DESCRIPTION = (
    "Floor plan of a home. Two cameras in its corners send video to a local server inside "
    "the home, which records it on local disk, finds and follows people and estimates "
    "their pose, then judges three things: a possible fall, a person on the floor and a "
    "possible head impact (experimental). An event below the confidence threshold is kept "
    "as a near-miss. At or above it, a text alert leaves through the door to a phone: "
    "Possible fall, camera 1, confidence 82%, with a Watch video button. A Monitoring "
    "degraded message is sent too when a camera is lost. A video clip leaves the home "
    "only when someone presses Watch video."
)


def text(
    x: float, y: float, s: str, size: int, fill: str, weight: int = 400, anchor: str = "start"
) -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{s}</text>'
    )


def station(x: float, title: str, lines: list[str], c: dict[str, str]) -> str:
    parts = [
        f'<circle cx="{x}" cy="300" r="10" fill="{c["paper"]}" stroke="{c["ink"]}" '
        'stroke-width="3"/>',
        text(x - 10, 266, title, 25, c["ink"], 700),
    ]
    parts += [text(x + 6, 342 + 24 * i, line, 19, c["muted"]) for i, line in enumerate(lines)]
    return "\n".join(parts)


def svg(c: dict[str, str]) -> str:
    camera = (
        '<g transform="translate({x} {y}) rotate({r})">'
        f'<rect x="-22" y="-12" width="44" height="24" rx="5" fill="{c["ink"]}"/>'
        f'<circle cx="14" cy="0" r="6" fill="{c["paper"]}"/></g>'
    )
    body = [
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{c["paper"]}"/>',
        # Field of view of each camera, under everything else.
        f'<path d="M 78 78 L 262 127 A 190 190 0 0 1 127 262 Z" fill="{c["cone"]}"/>',
        f'<path d="M 78 582 L 127 398 A 190 190 0 0 1 262 533 Z" fill="{c["cone"]}"/>',
        # Walls with the door gap on the right (290 to 370).
        f'<path d="M 960 290 V 40 H 40 V 620 H 960 V 370" fill="none" stroke="{c["ink"]}" '
        'stroke-width="12" stroke-linejoin="miter"/>',
        camera.format(x=72, y=72, r=45),
        camera.format(x=72, y=588, r=-45),
        text(930, 92, "Your home", 25, c["ink"], 700, "end"),
        text(930, 120, "Video is recorded and analysed here,", 19, c["muted"], 400, "end"),
        text(930, 144, "on a local server.", 19, c["muted"], 400, "end"),
        # Video from both cameras to the recorder.
        f'<path d="M 96 96 L 222 292 M 96 564 L 222 308" fill="none" '
        f'stroke="{c["hairline"]}" stroke-width="2.5"/>',
        # The processing line.
        f'<line x1="230" y1="300" x2="818" y2="300" stroke="{c["ink"]}" stroke-width="3"/>',
        station(230, "Record", ["on local disk"], c),
        station(430, "Find people", ["follow each one,", "estimate the pose"], c),
        station(
            630,
            "Judge",
            ["possible fall", "person on the floor", "possible head impact", "(experimental)"],
            c,
        ),
        # Threshold: below stays home as a near-miss, at or above leaves as an alert.
        f'<path d="M 830 286 L 844 300 L 830 314 L 816 300 Z" fill="{c["ink"]}"/>',
        text(806, 266, "Threshold", 25, c["ink"], 700),
        f'<line x1="830" y1="314" x2="830" y2="452" stroke="{c["hairline"]}" stroke-width="2.5"/>',
        f'<circle cx="830" cy="458" r="6" fill="none" stroke="{c["muted"]}" stroke-width="2.5"/>',
        text(812, 480, "Below it: kept as", 19, c["muted"], 400, "end"),
        text(812, 504, "a near-miss", 19, c["muted"], 400, "end"),
        f'<line x1="844" y1="300" x2="1034" y2="300" stroke="{c["accent"]}" stroke-width="6"/>',
        f'<path d="M 1046 300 L 1030 290 L 1030 310 Z" fill="{c["accent"]}"/>',
        text(1004, 286, "alert", 19, c["accent"], 700, "middle"),
        # The phone outside the home.
        f'<rect x="1050" y="150" width="174" height="350" rx="22" fill="{c["phone"]}" '
        f'stroke="{c["ink"]}" stroke-width="3"/>',
        f'<rect x="1064" y="190" width="146" height="172" rx="10" fill="{c["bubble"]}"/>',
        text(1076, 220, "Possible fall", 18, c["ink"], 700),
        text(1076, 248, "Camera 1", 16, c["ink"]),
        text(1076, 272, "Confidence: 82%", 16, c["ink"]),
        f'<rect x="1076" y="320" width="122" height="30" rx="6" fill="none" '
        f'stroke="{c["ink"]}" stroke-width="2"/>',
        text(1137, 340, "Watch video", 15, c["ink"], 700, "middle"),
        f'<rect x="1064" y="376" width="146" height="76" rx="10" fill="{c["quiet"]}"/>',
        text(1076, 402, "Monitoring", 16, c["ink"], 700),
        text(1076, 422, "degraded", 16, c["ink"], 700),
        text(1076, 442, "camera lost", 15, c["muted"]),
        # A clip leaves only on request: from the recording, out through the door.
        f'<path d="M 230 312 V 560 H 900 V 335 H 1062" fill="none" stroke="{c["muted"]}" '
        'stroke-width="2.5" stroke-dasharray="9 7"/>',
        f'<path d="M 1074 335 L 1060 327 L 1060 343 Z" fill="{c["muted"]}"/>',
        text(
            260,
            592,
            "A clip leaves the home only when someone presses Watch video.",
            19,
            c["muted"],
        ),
    ]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" font-family="{FONT}" role="img" '
        'aria-labelledby="title desc">\n'
        '<title id="title">How fall-impact-monitor works</title>\n'
        f'<desc id="desc">{DESCRIPTION}</desc>\n' + "\n".join(body) + "\n</svg>\n"
    )


if __name__ == "__main__":
    for name, colours in THEMES.items():
        path = HERE / f"how-it-works-{name}.svg"
        path.write_text(svg(colours), encoding="utf-8", newline="\n")
