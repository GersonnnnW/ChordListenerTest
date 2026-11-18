import hashlib
import shutil

import pytest
from fastapi.testclient import TestClient

from chordlistener import ENGINE_VERSION
from chordlistener.server.app import create_app
from tools import songgen
from chordlistener.server.catalog import CatalogProvider, Track
from chordlistener.lyrics import Lyrics
from chordlistener.server.config import Settings
from chordlistener.server.lyricsdb import LyricsProvider


class FakeCatalog(CatalogProvider):
    """Stands in for the real song catalogue; serves the test recording."""

    name = "fake"

    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.downloads = 0
        self.track = Track("fake", "42", "Test Song", "Test Band", preview_url="https://example.invalid/a.m4a", clip_seconds=30.0)

    async def search(self, query, limit=8):
        return [self.track] if "test" in query.lower() else []

    async def lookup(self, track_id):
        return self.track if track_id == "42" else None

    async def download(self, track, dest_path, max_bytes):
        self.downloads += 1
        shutil.copyfile(self.audio_path, dest_path)
        with open(dest_path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()


class FakeLyrics(LyricsProvider):
    """Stands in for the lyrics database: knows one song, timed for whatever length is asked about."""

    name = "fake"

    def __init__(self):
        self.asked = []

    async def find(self, title, artist, duration=None):
        self.asked.append((title, artist, duration))
        if "song" not in title.lower():
            return None
        lines = [(1.0, "first line of words"), (6.0, "second line here"), (11.0, "and a third")]
        return Lyrics(lines, "fake", title, artist, duration if duration is not None else 200.0)


def make_client(tmp_path, wav_path, **overrides):
    overrides.setdefault("lyrics", "none")  # never reach for the real lyrics database from a test
    settings = Settings(data_dir=str(tmp_path / "data"), **overrides)
    catalog = FakeCatalog(wav_path)
    lyrics = FakeLyrics()
    app = create_app(settings, catalog, lyrics if overrides["lyrics"] == "fake" else None)
    client = TestClient(app)
    client.lyrics = lyrics
    client.catalog = catalog
    client.store = app.state.store
    return client


@pytest.fixture()
def client(tmp_path, wav_path):
    with make_client(tmp_path, wav_path) as c:
        yield c


def upload(client, wav_path, **kwargs):
    with open(wav_path, "rb") as fh:
        return client.post("/v1/transcriptions", files={"file": ("song.wav", fh, "audio/wav")}, **kwargs)


def test_health_and_website(client):
    assert client.get("/v1/health").json()["status"] == "ok"
    page = client.get("/")
    assert page.status_code == 200 and "ChordListener" in page.text
    assert client.get("/static/app.js").status_code == 200


def test_upload_returns_chords_and_every_format(client, wav_path):
    response = upload(client, wav_path, data={"title": "My Song"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed" and body["cached"] is False
    result = body["result"]
    assert result["key"]["name"] == "C major"
    assert [c["chord"] for c in result["chords"]][:4] == ["C", "G", "Am", "F"]
    assert len(result["notes"]) > 50

    tid = body["id"]
    again = client.get(f"/v1/transcriptions/{tid}", params={"transpose": 2, "notes": "false"}).json()
    assert again["result"]["key"]["name"] == "D major" and "notes" not in again["result"]

    xml = client.get(f"/v1/transcriptions/{tid}/musicxml", params={"arrangement": "lead", "download": "true"})
    assert xml.status_code == 200 and "<score-partwise" in xml.text
    assert 'filename="My Song.lead.musicxml"' in xml.headers["content-disposition"]
    midi = client.get(f"/v1/transcriptions/{tid}/midi")
    assert midi.content[:4] == b"MThd"
    chart = client.get(f"/v1/transcriptions/{tid}/chart")
    assert "| C " in chart.text
    assert client.get(f"/v1/transcriptions/{tid}/musicxml", params={"arrangement": "nonsense"}).status_code == 422


def test_identical_audio_is_served_from_cache(client, wav_path):
    first = upload(client, wav_path).json()
    second = upload(client, wav_path).json()
    assert second["cached"] is True and second["id"] != first["id"]
    assert second["result"]["chords"] == first["result"]["chords"]
    # Different settings are a different analysis.
    assert upload(client, wav_path, data={"vocabulary": "basic"}).json()["cached"] is False


def test_search_and_transcribe_by_name(client):
    found = client.get("/v1/songs/search", params={"q": "test song"}).json()
    assert found["results"][0]["title"] == "Test Song"
    track_id = found["results"][0]["track_id"]
    body = client.post("/v1/transcriptions", data={"track_id": track_id}).json()
    assert body["status"] == "completed"
    assert body["source"]["type"] == "catalog" and body["source"]["artist"] == "Test Band"
    # The second request for the same track needs neither a download nor an analysis.
    again = client.post("/v1/transcriptions", data={"track_id": track_id}).json()
    assert again["cached"] is True
    assert client.catalog.downloads == 1
    assert client.post("/v1/transcriptions", data={"track_id": "999"}).status_code == 404


def test_bad_requests(client, wav_path, tmp_path):
    assert client.post("/v1/transcriptions").status_code == 400
    assert upload(client, wav_path, data={"vocabulary": "jazz"}).status_code == 400
    assert upload(client, wav_path, data={"tempo": "fast"}).status_code == 400
    bogus = client.post("/v1/transcriptions", files={"file": ("x.mp3", b"this is not audio at all, just text", "audio/mpeg")})
    assert bogus.status_code == 422
    assert bogus.json()["error"]["code"] == "invalid_audio"
    assert client.get("/v1/transcriptions/tr_missing").status_code == 404
    assert client.get("/v1/songs/search", params={"q": "x"}).status_code == 422


def test_upload_size_limit(tmp_path, wav_path):
    with make_client(tmp_path, wav_path, max_upload_mb=0) as client:
        response = upload(client, wav_path)
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "file_too_large"


def test_api_keys_isolate_results_and_meter_usage(client, wav_path):
    _, alice = client.store.create_key("Alice Ltd")
    _, bob = client.store.create_key("Bob GmbH")
    as_alice = {"Authorization": f"Bearer {alice}"}

    body = upload(client, wav_path, headers=as_alice).json()
    tid = body["id"]
    assert client.get(f"/v1/transcriptions/{tid}", headers=as_alice).status_code == 200
    assert client.get(f"/v1/transcriptions/{tid}", headers={"X-API-Key": bob}).status_code == 404
    assert client.get(f"/v1/transcriptions/{tid}").status_code == 404  # anonymous
    assert client.get("/v1/health", headers={"Authorization": "Bearer cl_wrong"}).status_code == 200
    assert client.get(f"/v1/transcriptions/{tid}", headers={"Authorization": "Bearer cl_wrong"}).status_code == 401

    usage = client.get("/v1/usage", headers=as_alice).json()
    assert usage["requests"] == 1 and usage["audio_seconds"] > 10
    assert client.get("/v1/usage", headers={"X-API-Key": bob}).json()["requests"] == 0

    assert client.delete(f"/v1/transcriptions/{tid}", headers=as_alice).status_code == 204
    assert client.get(f"/v1/transcriptions/{tid}", headers=as_alice).status_code == 404


def test_revoked_key_and_closed_public_access(tmp_path, wav_path):
    with make_client(tmp_path, wav_path, public_access=False) as client:
        assert upload(client, wav_path).status_code == 401
        key_id, secret = client.store.create_key("Carol")
        assert upload(client, wav_path, headers={"X-API-Key": secret}).status_code == 200
        client.store.revoke_key(key_id)
        assert upload(client, wav_path, headers={"X-API-Key": secret}).status_code == 401


def test_rate_limits(tmp_path, wav_path):
    with make_client(tmp_path, wav_path, public_per_hour=2) as client:
        assert upload(client, wav_path).status_code == 200
        assert upload(client, wav_path).status_code == 200
        limited = upload(client, wav_path)
        assert limited.status_code == 429 and int(limited.headers["retry-after"]) > 0
        _, secret = client.store.create_key("Dave", per_minute=1)
        headers = {"X-API-Key": secret}
        assert upload(client, wav_path, headers=headers).status_code == 200
        assert upload(client, wav_path, headers=headers).status_code == 429


def test_monthly_quota(tmp_path, wav_path):
    with make_client(tmp_path, wav_path) as client:
        _, secret = client.store.create_key("Erin", monthly_seconds=20)
        headers = {"X-API-Key": secret}
        assert upload(client, wav_path, headers=headers).status_code == 200  # ~50 s of audio
        over = upload(client, wav_path, headers=headers)
        assert over.status_code == 429 and "quota" in over.json()["error"]["message"]


def test_async_mode_can_be_polled(client, wav_path):
    response = upload(client, wav_path, params={"wait": "false"})
    assert response.status_code in (200, 202)
    tid = response.json()["id"]
    import time

    for _ in range(100):
        body = client.get(f"/v1/transcriptions/{tid}").json()
        if body["status"] != "processing":
            break
        time.sleep(0.1)
    assert body["status"] == "completed"


# ---- Editable scores -------------------------------------------------------


def created(client, wav_path, **kwargs):
    body = upload(client, wav_path, **kwargs).json()
    return body["id"], {"X-Edit-Token": body["edit_token"]} if "edit_token" in body else {}


def test_score_can_be_corrected_saved_and_reverted(client, wav_path):
    tid, token = created(client, wav_path, data={"title": "My Song"})
    score = client.get(f"/v1/transcriptions/{tid}/score", headers=token).json()
    assert score["edited"] is False and score["can_save"] is True
    assert score["view"]["chords"][0]["chord"] == "C"
    assert client.get(f"/v1/transcriptions/{tid}/score").json()["can_save"] is False  # no token: view only

    doc = score["document"]
    doc["chords"][0][1] = "Cmaj7"
    doc["lyrics"] = {"0": "Hello there"}
    doc["title"] = "Corrected"
    saved = client.put(f"/v1/transcriptions/{tid}/score", json=doc, headers=token)
    assert saved.status_code == 200 and saved.json()["edited"] is True

    again = client.get(f"/v1/transcriptions/{tid}/score").json()
    assert again["edited"] is True and again["document"]["title"] == "Corrected"
    xml = client.get(f"/v1/transcriptions/{tid}/musicxml").text
    assert "major-seventh" in xml and "<text>Hello</text>" in xml and "Corrected" in xml
    assert "<text>Hello</text>" not in client.get(f"/v1/transcriptions/{tid}/musicxml", params={"lyrics": "false"}).text
    assert "major-seventh" not in client.get(f"/v1/transcriptions/{tid}/musicxml", params={"original": "true"}).text
    chart = client.get(f"/v1/transcriptions/{tid}/chart").text
    assert "| Cmaj7" in chart and "Hello there" in chart

    assert client.delete(f"/v1/transcriptions/{tid}/score").status_code == 403
    assert client.delete(f"/v1/transcriptions/{tid}/score", headers=token).status_code == 204
    assert client.get(f"/v1/transcriptions/{tid}/score").json()["edited"] is False


def test_only_the_creator_can_save(client, wav_path):
    tid, token = created(client, wav_path)
    doc = client.get(f"/v1/transcriptions/{tid}/score").json()["document"]
    assert client.put(f"/v1/transcriptions/{tid}/score", json=doc).status_code == 403
    assert client.put(f"/v1/transcriptions/{tid}/score", json=doc, headers={"X-Edit-Token": "et_guess"}).status_code == 403
    assert client.put(f"/v1/transcriptions/{tid}/score", json=doc, headers=token).status_code == 200

    _, alice = client.store.create_key("Alice Ltd")
    _, bob = client.store.create_key("Bob GmbH")
    body = upload(client, wav_path, headers={"X-API-Key": alice}).json()
    assert "edit_token" not in body  # the API key is the credential
    keyed = body["id"]
    assert client.put(f"/v1/transcriptions/{keyed}/score", json=doc, headers={"X-API-Key": alice}).status_code == 200
    assert client.put(f"/v1/transcriptions/{keyed}/score", json=doc, headers={"X-API-Key": bob}).status_code == 404


def test_saving_rejects_broken_scores(client, wav_path):
    tid, token = created(client, wav_path)
    doc = client.get(f"/v1/transcriptions/{tid}/score").json()["document"]
    doc["arrangements"]["lead"][0].append([0, 1, [300]])
    refused = client.put(f"/v1/transcriptions/{tid}/score", json=doc, headers=token)
    assert refused.status_code == 400 and "pitch" in refused.json()["error"]["message"].lower()
    assert client.put(f"/v1/transcriptions/{tid}/score", content=b"{not json", headers={**token, "Content-Type": "application/json"}).status_code == 400
    assert client.get(f"/v1/transcriptions/{tid}/score").json()["edited"] is False


def test_render_without_saving(client, wav_path):
    tid, _ = created(client, wav_path)
    doc = client.get(f"/v1/transcriptions/{tid}/score").json()["document"]
    doc["lyrics"] = {"1": "unsaved words"}
    out = client.post("/v1/score/render", json={"document": doc, "arrangement": "lead", "transpose": 2, "chart": True, "show": {"tempo": False}}).json()
    assert out["view"]["key"]["name"] == "D major" and out["view"]["has_lyrics"] is True
    assert "<text>unsaved</text>" in out["musicxml"] and "<metronome>" not in out["musicxml"] and "<staves>" not in out["musicxml"]
    assert "unsaved words" in out["chart"]
    assert "musicxml" not in client.post("/v1/score/render", json={"document": doc, "musicxml": False}).json()
    assert client.get(f"/v1/transcriptions/{tid}/score").json()["document"]["lyrics"] == {}  # nothing was stored
    for bad in ({"document": {}}, {"document": doc, "arrangement": "band"}, {"document": doc, "transpose": 40}, []):
        assert client.post("/v1/score/render", json=bad).status_code == 400


def test_chord_sheet_and_plain_options(client, wav_path):
    tid, _ = created(client, wav_path)
    sheet = client.get(f"/v1/transcriptions/{tid}/musicxml", params={"notes": "false", "key": "false", "tempo": "false"}).text
    assert "<notehead>slash</notehead>" in sheet and "<movement-title>" not in sheet
    plain = client.get(f"/v1/transcriptions/{tid}/musicxml", params={"chords": "false"}).text
    assert "<harmony" not in plain and "Key: C major" in plain


def test_ad_spaces_are_off_unless_enabled(tmp_path, wav_path):
    with make_client(tmp_path, wav_path) as client:
        page = client.get("/")
        assert 'class="ads-on"' not in page.text and "/static/ads.js" not in page.text
        assert "frame-src 'none'" in page.headers["content-security-policy"]
    with make_client(tmp_path / "b", wav_path, ads=True, ad_origins="https://ads.example") as client:
        page = client.get("/")
        assert 'class="ads-on"' in page.text and "/static/ads.js" in page.text
        assert "frame-src https://ads.example" in page.headers["content-security-policy"]
        assert client.get("/static/ads.js").status_code == 200


# ---- Lyrics, video uploads, capo --------------------------------------------


def test_lyrics_are_found_and_placed_for_a_full_track(tmp_path, wav_path):
    with make_client(tmp_path, wav_path, lyrics="fake") as client:
        with open(wav_path, "rb") as fh:
            body = client.post("/v1/transcriptions", files={"file": ("The Band - My Song (Official Video).wav", fh, "audio/wav")}).json()
        tid, token = body["id"], {"X-Edit-Token": body["edit_token"]}
        assert (body["source"]["title"], body["source"]["artist"]) == ("My Song", "The Band")
        title, artist, duration = client.lyrics.asked[-1]
        assert (title, artist) == ("My Song", "The Band") and abs(duration - body["source"]["audio_seconds"]) < 0.01

        found = client.get(f"/v1/transcriptions/{tid}/lyrics").json()
        assert found["found"] and found["aligned"] and found["synced"]
        assert [line["text"] for line in found["lines"]][0] == "first line of words"
        doc = client.get(f"/v1/transcriptions/{tid}/score").json()["document"]
        assert doc["lyrics"] == found["bars"] and "first" in doc["lyrics"]["0"]
        sung = " ".join(doc["lyrics"][bar] for bar in sorted(doc["lyrics"], key=int))
        assert sung == "first line of words second line here and a third"  # every word, in order
        assert "<text>first</text>" in client.get(f"/v1/transcriptions/{tid}/musicxml").text

        # Looking again under another name replaces what is stored only for the creator.
        assert client.post(f"/v1/transcriptions/{tid}/lyrics", data={"title": "Unknown tune"}, headers=token).json()["found"] is False
        assert client.get(f"/v1/transcriptions/{tid}/lyrics").json()["found"] is True
        assert client.post(f"/v1/transcriptions/{tid}/lyrics", data={"title": "Other song", "artist": "X"}).json()["title"] == "Other song"
        assert client.get(f"/v1/transcriptions/{tid}/lyrics").json()["title"] == "My Song"


def test_lyrics_for_an_excerpt_are_shown_but_not_placed(tmp_path, wav_path):
    with make_client(tmp_path, wav_path, lyrics="fake") as client:
        tid = client.post("/v1/transcriptions", data={"track_id": "42"}).json()["id"]
        found = client.get(f"/v1/transcriptions/{tid}/lyrics").json()
        assert found["found"] is True and found["aligned"] is False and "bars" not in found
        assert client.get(f"/v1/transcriptions/{tid}/score").json()["document"]["lyrics"] == {}


def test_no_lyrics_provider(client, wav_path):
    tid, _ = created(client, wav_path)
    assert client.get(f"/v1/transcriptions/{tid}/lyrics").json() == {"found": False, "provider": None}
    assert client.get("/v1/health").json()["lyrics"] is None


def test_video_upload_is_converted(client, wav_path, tmp_path):
    import shutil
    import subprocess

    video = str(tmp_path / "clip.avi")
    made = subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x48:r=5", "-i", wav_path, "-shortest", "-c:v", "mpeg4", "-c:a", "libmp3lame", video],
        capture_output=True,
    )
    if made.returncode != 0:
        pytest.skip("this ffmpeg build cannot make a test video")
    with open(video, "rb") as fh:
        response = client.post("/v1/transcriptions", files={"file": ("Holiday clip.avi", fh, "video/x-msvideo")})
    assert response.status_code == 200
    body = response.json()
    assert body["source"]["video"] is True
    assert [c["chord"] for c in body["result"]["chords"]][:4] == ["C", "G", "Am", "F"]  # same music as the audio-only test
    tid, token = body["id"], body["edit_token"]
    audio = client.get(body["links"]["audio"], params={"token": token})
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/mpeg" and len(audio.content) > 20_000
    assert client.get(body["links"]["audio"], headers={"X-Edit-Token": token}).status_code == 200
    assert client.get(body["links"]["audio"]).status_code == 403  # the link alone is not enough
    assert client.delete(f"/v1/transcriptions/{tid}").status_code == 204
    assert client.get(body["links"]["audio"], params={"token": token}).status_code == 404


def test_audio_uploads_keep_no_copy(client, wav_path):
    body = upload(client, wav_path).json()
    assert "audio" not in body["links"] and "video" not in body["source"]
    assert client.get(f"/v1/transcriptions/{body['id']}/audio", params={"token": body["edit_token"]}).status_code == 404


def test_capo_and_guitar_tip(client, wav_path):
    tid, _ = created(client, wav_path)
    score = client.get(f"/v1/transcriptions/{tid}/score").json()
    assert score["view"]["guitar"] is None  # C G Am F needs no help
    doc = score["document"]
    doc["chords"] = [[0, "Bbm"], [4, "Ab"], [8, "Db"]]
    doc["key"] = {"tonic": 1, "minor": False}
    out = client.post("/v1/score/render", json={"document": doc, "transpose": -1, "capo": 1, "chart": True}).json()
    assert out["view"]["guitar"] == {"transpose": -1, "capo": 1, "tune_down": 0, "chords": ["Am", "G", "C"]}
    assert [c["chord"] for c in out["view"]["chords"]] == ["Am", "G", "C"]
    assert "Capo 1" in out["musicxml"] and "Capo 1" in out["chart"]
    assert client.post("/v1/score/render", json={"document": doc, "capo": 40}).status_code == 400
    assert "Capo 2" in client.get(f"/v1/transcriptions/{tid}/musicxml", params={"transpose": -2, "capo": 2}).text


def test_health_reports_limits(client):
    health = client.get("/v1/health").json()
    assert health["max_upload_mb"] == 1000 and health["engine_version"] == ENGINE_VERSION


needs_piano = pytest.mark.skipif(not __import__("chordlistener.highres").highres.available("piano"), reason="piano model not installed")


@needs_piano
def test_high_accuracy_upload_and_switching_back_and_forth(client, wav_path):
    assert "piano" in client.get("/v1/health").json()["high_accuracy"]
    assert upload(client, wav_path, data={"accuracy": "extreme"}).status_code == 400
    high = upload(client, wav_path, data={"accuracy": "high", "tempo": "half"}).json()  # "high" is the earlier name for piano
    assert high["status"] == "completed" and high["result"]["accuracy"] == "piano"
    assert high["source"]["options"]["accuracy"] == "piano"

    # Back to standard for the same upload: the file is not kept, so it must be sent again ...
    tid = high["id"]
    assert client.post(f"/v1/transcriptions/{tid}/reanalyze", data={"accuracy": "standard"}).status_code == 400
    with open(__file__, "rb") as other:
        wrong = client.post(f"/v1/transcriptions/{tid}/reanalyze", data={"accuracy": "standard"}, files={"file": ("x.wav", other, "audio/wav")})
    assert wrong.status_code == 400
    with open(wav_path, "rb") as fh:
        standard = client.post(f"/v1/transcriptions/{tid}/reanalyze", data={"accuracy": "standard"}, files={"file": ("song.wav", fh, "audio/wav")}).json()
    assert standard["result"]["accuracy"] == "standard" and standard["id"] != tid
    assert standard["source"]["options"] == {**high["source"]["options"], "accuracy": "standard"}  # other settings kept
    assert standard["edit_token"]
    # ... unless that mode's result is already cached.
    again = client.post(f"/v1/transcriptions/{standard['id']}/reanalyze", data={"accuracy": "piano"}).json()
    assert again["cached"] is True and again["result"]["accuracy"] == "piano"


@needs_piano
def test_a_song_found_by_name_can_be_switched_to_high_accuracy(client):
    body = client.post("/v1/transcriptions", data={"track_id": "42"}).json()
    assert body["result"]["accuracy"] == "standard"
    high = client.post(f"/v1/transcriptions/{body['id']}/reanalyze").json()  # piano is the default
    assert high["status"] == "completed" and high["result"]["accuracy"] == "piano"
    assert high["source"]["type"] == "catalog" and high["source"]["title"] == "Test Song"
    assert client.catalog.downloads == 2  # fetched again: the clip is never kept
    assert client.post("/v1/transcriptions/nope/reanalyze").status_code == 404


@pytest.mark.skipif(not __import__("chordlistener.highres").highres.available("guitar"), reason="guitar model not installed")
def test_guitar_accuracy(client, wav_path):
    assert "guitar" in client.get("/v1/health").json()["high_accuracy"]
    body = upload(client, wav_path, data={"accuracy": "guitar"}).json()
    assert body["status"] == "completed" and body["result"]["accuracy"] == "guitar"
    piano = client.post(f"/v1/transcriptions/{body['id']}/reanalyze", data={"accuracy": "piano"})
    assert piano.status_code == 400  # an upload has to be sent again


def test_high_accuracy_can_be_switched_off(tmp_path, wav_path):
    with make_client(tmp_path, wav_path, high_accuracy=False) as c:
        assert c.get("/v1/health").json()["high_accuracy"] == []
        assert upload(c, wav_path, data={"accuracy": "high"}).status_code == 400
        assert upload(c, wav_path, data={"accuracy": "guitar"}).status_code == 400


def test_guitar_tab(client, wav_path):
    tid = upload(client, wav_path).json()["id"]
    text = client.get(f"/v1/transcriptions/{tid}/tab", params={"source": "melody", "download": "true"})
    assert text.status_code == 200 and "e|" in text.text and ".tab.txt" in text.headers["content-disposition"]
    xml = client.get(f"/v1/transcriptions/{tid}/musicxml", params={"arrangement": "guitar"})
    assert xml.status_code == 200 and "<sign>TAB</sign>" in xml.text
    doc = client.get(f"/v1/transcriptions/{tid}/score").json()["document"]
    out = client.post("/v1/score/render", json={"document": doc, "arrangement": "guitar_melody", "tab": True}).json()
    assert "e|" in out["tab"] and "<sign>TAB</sign>" in out["musicxml"]
    assert client.get(f"/v1/transcriptions/{tid}/tab", params={"source": "bass"}).status_code == 422


def test_versions_can_be_shared_chosen_and_rated(tmp_path, wav_path):
    import json as _json

    with make_client(tmp_path, wav_path) as c:
        named = {"title": "Our Song (Official Video)", "artist": "The Band"}
        tid = upload(c, wav_path, data=named).json()["id"]
        listing = c.get(f"/v1/transcriptions/{tid}/versions").json()
        assert [v["id"] for v in listing["versions"]] == ["auto"] and listing["recommended"] == "auto"
        assert listing["by_name"] is True and listing["song"] == "band--our-song"

        # A chord sheet with bar lines, from bar 1.
        made = c.post(f"/v1/transcriptions/{tid}/versions", data={"author": "Ann", "text": "| C | G | Am | F |"})
        assert made.status_code == 200
        version = made.json()["version"]
        assert version["author"] == "Ann" and version["how"] == "chords" and version["ratings"] == 0
        assert [ch[1] for ch in made.json()["document"]["chords"][:4]] == ["C", "G", "Am", "F"]

        # MusicXML: our own export comes back with the same chords.
        xml = c.get(f"/v1/transcriptions/{tid}/musicxml").text
        imported = c.post(f"/v1/transcriptions/{tid}/versions", data={"author": "Bob"}, files={"file": ("song.musicxml", xml.encode(), "application/xml")}).json()
        assert imported["version"]["how"] == "musicxml" and imported["document"]["chords"][:4] == made.json()["document"]["chords"][:4]

        # Ratings: one per person, changeable, 0 takes it back.
        after = c.post(f"/v1/transcriptions/{tid}/versions/{version['id']}/rating", data={"stars": 4}).json()
        mine = next(v for v in after["versions"] if v["id"] == version["id"])
        assert (mine["ratings"], mine["stars"], mine["my_stars"]) == (1, 4.0, 4)
        after = c.post(f"/v1/transcriptions/{tid}/versions/{version['id']}/rating", data={"stars": 5}).json()
        assert next(v for v in after["versions"] if v["id"] == version["id"])["stars"] == 5.0
        after = c.post(f"/v1/transcriptions/{tid}/versions/auto/rating", data={"stars": 2}).json()
        assert after["versions"][0]["stars"] == 2.0
        assert c.post(f"/v1/transcriptions/{tid}/versions/{version['id']}/rating", data={"stars": 9}).status_code == 422
        assert c.get(f"/v1/transcriptions/{tid}/versions/v_nope").status_code == 404

        # Another recording of the same song (other tempo, other key, a bar of silence first) finds the versions by name,
        # and each one is moved and transposed to fit it.
        other = tmp_path / "other.wav"
        song = songgen.make_song([("D", 4), ("A", 4), ("Bm", 4), ("G", 4)], bpm=120, key="D", style="piano", repeats=3, seed=3)
        songgen.write_wav(str(other), songgen.render_numpy(song, seed=3))
        with open(other, "rb") as fh:
            tid2 = c.post("/v1/transcriptions", files={"file": ("other.wav", fh, "audio/wav")}, data={"title": "Our Song", "artist": "Band"}).json()["id"]
        listing2 = c.get(f"/v1/transcriptions/{tid2}/versions").json()
        assert listing2["song"] == "band--our-song" and version["id"] in [v["id"] for v in listing2["versions"]]
        assert listing2["versions"][0]["ratings"] == 0  # the automatic version of this recording has its own ratings
        got = c.get(f"/v1/transcriptions/{tid2}/versions/{version['id']}").json()
        assert got["fitted"]["transpose"] == 2
        assert [ch["chord"] for ch in got["view"]["chords"][:4]] == ["D", "A", "Bm", "G"]
        assert c.get(f"/v1/transcriptions/{tid}/versions/{version['id']}").json()["fitted"] is None  # its own recording: as made

        # Corrections from the editor can be shared too; nonsense is refused.
        doc = got["document"]
        doc["chords"][0][1] = "Dmaj7"
        shared = c.post(f"/v1/transcriptions/{tid2}/versions", data={"author": "Cy", "document": _json.dumps(doc)}).json()
        assert shared["version"]["how"] == "edited"
        assert c.post(f"/v1/transcriptions/{tid}/versions", data={"text": "hello there"}).status_code == 400
        assert c.post(f"/v1/transcriptions/{tid}/versions", data={}).status_code == 400

        # Without an artist, versions stay with the exact recording.
        tid3 = upload(c, wav_path).json()["id"]
        assert c.get(f"/v1/transcriptions/{tid3}/versions").json()["by_name"] is False

