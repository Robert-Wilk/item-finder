"""
Shared vocabulary for the item-finder demo.

Keep TARGET_CLASSES small and matched to what you'll actually demo with.
Fewer, well-tested classes beats a long list you haven't verified the
detector actually sees reliably in your demo room/lighting.

Each key in ALIASES is a lowercase phrase a user might type/say; each
value is the canonical COCO class name the detector will output.
"""

# --- Narrow this down to objects you've actually confirmed the model
# --- detects reliably in your test environment. Start small.
TARGET_CLASSES = [
    "remote",
    "cell phone",
    "cup",
    "bottle",
    "book",
    "backpack",
]

ALIASES = {
    "remote": "remote",
    "tv remote": "remote",
    "remote control": "remote",
    "clicker": "remote",
    "controller": "remote",

    "phone": "cell phone",
    "cell phone": "cell phone",
    "cellphone": "cell phone",
    "mobile": "cell phone",
    "smartphone": "cell phone",

    "cup": "cup",
    "mug": "cup",

    "bottle": "bottle",
    "water bottle": "bottle",

    "book": "book",

    "backpack": "backpack",
    "bag": "backpack",
}


def resolve_target(query: str):
    """
    Resolve a free-text query to a canonical COCO class name.

    Returns (resolved_class: str | None, matched_via: str)
    matched_via is one of: "alias", "fuzzy", "none"
    """
    q = query.strip().lower()

    if q in ALIASES:
        return ALIASES[q], "alias"

    # Fuzzy fallback using stdlib difflib - no extra dependencies,
    # cheap enough to run on the Pi with zero setup cost.
    import difflib
    candidates = list(ALIASES.keys())
    matches = difflib.get_close_matches(q, candidates, n=1, cutoff=0.6)
    if matches:
        return ALIASES[matches[0]], "fuzzy"

    return None, "none"
