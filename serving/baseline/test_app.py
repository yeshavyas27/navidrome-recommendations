"""
Targeted tests for the two contracts we care about:

  1. Parquet lookup — track_meta must be keyed by str so downstream
     str(track_id) lookups hit. This was a silent bug (int keys vs str
     lookup) that made every recommendation return empty title/artist.

  2. /play fallback — cache hit returns the track's own mp3, cache miss
     returns a random mp3 from the bucket. The demo relies on: "UI shows
     correct title/artist even if audio is substituted."

Not a full integration suite — no model, no real S3, no real parquet.
Just unit tests that pin down the two behaviors.

Run from the serving/baseline dir:

    pip install pytest
    pytest test_app.py -v
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import app as app_module


# ─── fixtures ────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def clean_state():
    """Reset module state around each test so they don't leak into each other."""
    original = dict(app_module.state)
    app_module.state.clear()
    yield app_module.state
    app_module.state.clear()
    app_module.state.update(original)


@pytest.fixture
def mock_s3():
    """A fresh mocked S3 client for each test, pre-wired into state."""
    s3 = MagicMock()
    # presigned URL is deterministic for assertion convenience
    s3.generate_presigned_url.side_effect = (
        lambda op, Params, ExpiresIn: f"https://swift.example/{Params['Key']}?presigned=1"
    )
    app_module.state["audio_s3"] = s3
    return s3


# ─── 1. parquet lookup ───────────────────────────────────────────────────
def test_lookup_track_meta_finds_str_keyed_track(clean_state):
    """With str keys in track_meta (what the fixed loader produces),
    a lookup with the same str track_id returns the expected dict."""
    clean_state["track_meta"] = {
        "1030591": {"title": "Come With Me", "artist": "Echosmith"},
    }
    meta = app_module._lookup_track_meta("1030591")
    assert meta == {"title": "Come With Me", "artist": "Echosmith"}


def test_lookup_track_meta_handles_int_track_id_as_string(clean_state):
    """Defensive: if a caller passes an int-convertible str and keys
    happen to still be ints (old/stale parquet loader), the helper's
    int fallback still finds it."""
    clean_state["track_meta"] = {
        1030591: {"title": "Come With Me", "artist": "Echosmith"},  # int key
    }
    meta = app_module._lookup_track_meta("1030591")
    assert meta == {"title": "Come With Me", "artist": "Echosmith"}


def test_lookup_track_meta_returns_empty_for_unknown_track(clean_state):
    clean_state["track_meta"] = {"1030591": {"title": "x", "artist": "y"}}
    assert app_module._lookup_track_meta("9999999") == {}


def test_lookup_track_meta_returns_empty_when_track_meta_missing(clean_state):
    # No track_meta in state at all (e.g. parquet load failed at startup).
    assert app_module._lookup_track_meta("1030591") == {}


# ─── 2. /play cache hit vs random fallback ──────────────────────────────
def test_play_cache_hit_redirects_to_own_mp3(mock_s3):
    """If audio/<id>.mp3 exists, /play redirects to a presigned URL for
    that exact object — not to a random one."""
    mock_s3.head_object.return_value = {}  # HEAD succeeds → cache hit

    resp = app_module.play(track_id="1030591")

    assert resp.status_code == 302
    # presigned URL should point at the track's own key, not a random one
    expected_key = f"{app_module.AUDIO_KEY_PREFIX}1030591.mp3"
    assert expected_key in resp.headers["location"]
    mock_s3.list_objects_v2.assert_not_called()  # no fallback listing on hit


def test_play_cache_miss_redirects_to_random_bucket_object(mock_s3):
    """If audio/<id>.mp3 is not in the bucket, /play lists available
    objects and redirects to a random one — demo fallback."""
    # HEAD raises → cache miss
    mock_s3.head_object.side_effect = Exception("NoSuchKey")
    # list_objects_v2 returns a handful of candidate keys
    mock_s3.list_objects_v2.return_value = {
        "Contents": [
            {"Key": "audio_complete/1040741.mp3"},
            {"Key": "audio_complete/1030591.mp3"},
            {"Key": "audio_complete/2618754.mp3"},
        ],
    }

    resp = app_module.play(track_id="9999999")  # id not in bucket

    assert resp.status_code == 302
    # redirect target must be one of the listed keys, not the requested id
    location = resp.headers["location"]
    valid_keys = {"audio_complete/1040741.mp3",
                  "audio_complete/1030591.mp3",
                  "audio_complete/2618754.mp3"}
    assert any(k in location for k in valid_keys), f"unexpected location: {location}"
    assert "9999999" not in location  # the requested (missing) id isn't in the URL
    mock_s3.list_objects_v2.assert_called_once()


def test_play_cache_miss_with_empty_bucket_returns_404(mock_s3):
    """If the bucket is empty and the track isn't there, /play 404s
    instead of returning a bogus redirect."""
    from fastapi import HTTPException
    mock_s3.head_object.side_effect = Exception("NoSuchKey")
    mock_s3.list_objects_v2.return_value = {"Contents": []}

    with pytest.raises(HTTPException) as exc_info:
        app_module.play(track_id="9999999")
    assert exc_info.value.status_code == 404


def test_play_returns_503_when_s3_not_configured(clean_state):
    """Guard against running without Swift/MinIO creds."""
    from fastapi import HTTPException
    # Make _audio_s3_client return None by clearing state and env
    import os
    for var in ("MINIO_URL", "S3_ENDPOINT_URL"):
        os.environ.pop(var, None)
    # Ensure no cached client
    clean_state.pop("audio_s3", None)

    with pytest.raises(HTTPException) as exc_info:
        app_module.play(track_id="1030591")
    assert exc_info.value.status_code == 503


# ─── 3. key helpers ──────────────────────────────────────────────────────
def test_audio_cache_key_format():
    assert app_module._audio_cache_key("1030591").endswith("1030591.mp3")
    assert app_module._audio_cache_key("1030591").startswith(app_module.AUDIO_KEY_PREFIX)


def test_audio_cached_true_when_head_succeeds(mock_s3):
    mock_s3.head_object.return_value = {}
    assert app_module._audio_cached(mock_s3, "1030591") is True


def test_audio_cached_false_when_head_raises(mock_s3):
    mock_s3.head_object.side_effect = Exception("NoSuchKey")
    assert app_module._audio_cached(mock_s3, "1030591") is False
