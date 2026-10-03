"""Safety-floor tests: the NSFW denylist must block real NSFW without false-positiving on innocent words."""

from forge_mcp.presets import expand_negative, positive_is_blocked


def test_no_substring_false_positives():
    # "cum" in "documentary", "cumulus", "cucumber"; "sex" in "unisex"/"Essex" — must NOT block.
    for p in [
        "RAW candid documentary photograph, portrait, a 21-year-old woman",
        "accumulate cumulus clouds over a cucumber field",
        "a unisex salon in Essex, scalextric on the shelf",
        "a knight in weathered armor, candid, natural light",
    ]:
        assert positive_is_blocked(p) is None, p


def test_real_nsfw_still_blocked():
    for p in ["a nude woman", "nipples visible", "explicit sexual content",
              "genitalia", "cum", "hardcore sex", "nsfw please", "naked", "topless"]:
        assert positive_is_blocked(p) is not None, p


def test_booru_underscore_tags_blocked():
    # Pony/Illustrious prompts join words with `_`, which regex `\b` treats as a word char — these slipped past.
    for p in ["rating_explicit, score_9", "1girl, completely_nude", "score_9, rating_questionable",
              "rating:questionable", "masterpiece, topless_female", "1girl, nsfw_style"]:
        assert positive_is_blocked(p) is not None, p


def test_booru_tags_no_false_positives():
    for p in ["score_9, score_8_up, rating_safe, source_anime, 1girl, silver_hair",
              "a questionable_decision, documentary_photo, unisex_salon, cucumber_field"]:
        assert positive_is_blocked(p) is None, p


def test_nsfw_floor_always_appended():
    # Whatever the profile or raw negative, the NSFW block is appended unconditionally.
    neg = expand_negative("candid+sfw-strict", "cluttered background")
    assert "nsfw" in neg and "nude" in neg          # the floor
    assert "swan neck" in neg                         # the candid profile
    assert "cluttered background" in neg              # the caller's raw negative
    # even with a bogus/empty profile:
    assert "nsfw" in expand_negative(None, None)
    assert "nsfw" in expand_negative("does-not-exist", "")
