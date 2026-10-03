"""Named presets + the server-side safety floor.

Design:
- Named NEGATIVE profiles the server expands (`sfw-strict` default, `mature`, craft add-ons `candid`,
  `anti-anime`). Combine with `+` (e.g. "candid+sfw-strict"); a raw negative string may also be passed and
  is appended. These craft profiles are hard-won IP — the consuming repo is free to override/extend by
  passing its own negative text; what's here is the sane built-in baseline.
- ★ The NSFW block is UNCONDITIONAL and non-overridable. It is appended to EVERY request regardless of
  profile, prompt, or calling model — because an MCP cannot assume a smart/aligned caller (a weak or
  jailbroken client hits the same wall). Legitimate NSFW is a human-at-the-Forge-UI action, never via this
  tool. A positive-prompt denylist rejects the obvious attempts with a clear error.
"""

from __future__ import annotations

import re

# --- the safety floor (never trust the caller) ---------------------------------------------------------

# Always appended to the negative prompt. Biases hard away from explicit content.
NSFW_NEGATIVE = (
    "nsfw, nude, nudity, naked, explicit, sexual, sex, porn, pornographic, genitalia, nipples, areola, "
    "cleavage focus, lingerie, underwear, suggestive, erotic, fetish, hentai, rating_explicit, rating_questionable"
)

# If any of these appear in the POSITIVE prompt, reject outright (don't even try to render).
# Matched on WORD BOUNDARIES, never as raw substrings — "cum" must not fire on "do(cum)entary",
# "sex" must not fire on "uni(sex)"/"(sex)tant". Prefix terms allow natural suffixes (nude->nudes,
# nipple->nipples, genital->genitalia); the short, false-positive-prone words are whole-word only.
_DENY_PREFIX = (
    "nsfw", "nude", "nudity", "naked", "explicit", "porn", "hentai", "sexual", "genital",
    "penis", "vagina", "blowjob", "topless", "bottomless", "undressed", "nipple", "areola",
)
_DENY_WHOLE = ("cum", "sex")  # whole-word only (substring would hit documentary/cucumber/unisex/…)
# Booru rating tags (Pony/Illustrious vocabulary). "questionable" alone is an innocent word, so only the tag form.
_DENY_PHRASE = (r"rating\W*questionable",)

_DENY_RE = re.compile(
    r"\b(?:" + "|".join(_DENY_PREFIX) + r")"          # word-start boundary, any suffix
    r"|\b(?:" + "|".join(_DENY_WHOLE) + r")\b"        # both boundaries
    r"|\b(?:" + "|".join(_DENY_PHRASE) + r")\b",      # tag-form phrases
    re.IGNORECASE,
)


def positive_is_blocked(prompt: str) -> str | None:
    """Return the offending term if the positive prompt requests NSFW, else None (word-boundary match).

    `_` is treated as a separator: booru-style tags (`rating_explicit`, `completely_nude`) would otherwise
    slip past the word boundary, since regex counts `_` as a word character."""
    m = _DENY_RE.search((prompt or "").replace("_", " "))
    return m.group(0).lower() if m else None


# --- craft / style negative profiles (overridable by the consuming repo) --------------------------------

NEGATIVE_PROFILES: dict[str, str] = {
    "sfw-strict": (
        "lowres, bad anatomy, bad hands, extra digits, fewer digits, deformed, disfigured, mutated, "
        "watermark, signature, text, jpeg artifacts, worst quality, low quality"
    ),
    # Seinen hard-SF: allows grit (combat, injury, somber intensity) while staying SFW. Does NOT relax NSFW.
    "mature": "cartoonish, cute, chibi, moe, saccharine, cheerful, bright pastel, comedic",
    # Anti-glamour / documentary register: fights the model's prettify-and-swan-neck centroid.
    "candid": "glamour, fashion photography, airbrushed, plastic skin, posed, studio lighting, swan neck, elongated neck, model pose",
    "anti-anime": "anime, manga, cel shading, 2d, illustration, cartoon, drawing",
}


def expand_negative(profile: str | None, raw_negative: str | None) -> str:
    """Resolve `profile` (e.g. "candid+sfw-strict") + any raw negative, then ALWAYS append the NSFW block."""
    parts: list[str] = []
    for name in (profile or "sfw-strict").split("+"):
        name = name.strip()
        if name and name in NEGATIVE_PROFILES:
            parts.append(NEGATIVE_PROFILES[name])
    if raw_negative and raw_negative.strip():
        parts.append(raw_negative.strip())
    parts.append(NSFW_NEGATIVE)  # ★ unconditional, last word
    # de-dup while keeping order
    seen: set[str] = set()
    out: list[str] = []
    for chunk in ", ".join(parts).split(","):
        t = chunk.strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return ", ".join(out)


# --- style presets (positive-side boilerplate the server can prepend) -----------------------------------

STYLE_PRESETS: dict[str, str] = {
    "photoreal": "photorealistic, realistic, natural lighting, detailed",
    "illustration": "detailed illustration, painterly",
}


def apply_style(prompt: str, style: str | None) -> str:
    if style and style in STYLE_PRESETS:
        return f"{prompt}, {STYLE_PRESETS[style]}"
    return prompt
