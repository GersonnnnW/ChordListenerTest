import hashlib
import shutil

import pytest
from fastapi.testclient import TestClient

from chordlistener.server.app import create_app
from chordlistener.server.catalog import CatalogProvider, Track
from chordlistener.server.config import Settings


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


def make_client(tmp_path, wav_path, **overrides):
    settings = Settings(data_dir=str(tmp_path / "data"), **overrides)
    catalog = FakeCatalog(wav_path)
    app = create_app(settings, catalog)
    client = TestClient(app)
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
