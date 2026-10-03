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

# --- the safety floor (never trust the caller) ---------------------------------------------------------

# Always appended to the negative prompt. Biases hard away from explicit content.
NSFW_NEGATIVE = (
    "nsfw, nude, nudity, naked, explicit, sexual, sex, porn, pornographic, genitalia, nipples, areola, "
    "cleavage focus, lingerie, underwear, suggestive, erotic, fetish, hentai"
)

# If any of these appear in the POSITIVE prompt, reject outright (don't even try to render).
POSITIVE_DENYLIST = (
    "nsfw", "nude", "naked", "nudity", "explicit", "porn", "hentai", "sex ", "sexual", "genital",
    "penis", "vagina", "cum", "blowjob", "topless", "bottomless", "undressed", "nipple",
)


def positive_is_blocked(prompt: str) -> str | None:
    """Return the offending term if the positive prompt requests NSFW, else None."""
    low = f" {prompt.lower()} "
    for term in POSITIVE_DENYLIST:
        if term in low:
            return term.strip()
    return None


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
