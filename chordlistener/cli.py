"""Command line: analyse files, run the server, manage API keys."""

from __future__ import annotations

import argparse
import json
import os
import sys


def _analyze(args: argparse.Namespace) -> int:
    from . import AnalysisOptions, analyze, export
    from .audio import AudioError
    from .document import ScoreOptions

    try:
        scale = {"auto": 1.0, "half": 0.5, "double": 2.0}[args.tempo]
        result = analyze(args.audio, AnalysisOptions(vocabulary=args.vocabulary, beats_per_bar=args.beats_per_bar, tempo_scale=scale, accuracy="high" if args.high_accuracy else "standard"))
    except (AudioError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    title = args.title or os.path.splitext(os.path.basename(args.audio))[0]
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        base = os.path.join(args.out, os.path.splitext(os.path.basename(args.audio))[0])
        with open(base + ".json", "w") as fh:
            json.dump(result.to_dict(args.transpose), fh)
        for arrangement in ("piano", "lead"):
            with open(f"{base}.{arrangement}.musicxml", "w") as fh:
                fh.write(export.musicxml(result, ScoreOptions(arrangement=arrangement, transpose=args.transpose, title=title)))
        with open(base + ".mid", "wb") as fh:
            fh.write(export.midi(result, args.transpose))
        with open(base + ".chords.txt", "w") as fh:
            fh.write(export.chord_chart(result, args.transpose, title))
        print(f"Wrote {base}.{{json,piano.musicxml,lead.musicxml,mid,chords.txt}}", file=sys.stderr)
    if args.json:
        json.dump(result.to_dict(args.transpose), sys.stdout)
        print()
    else:
        print(export.chord_chart(result, args.transpose, title), end="")
        speed = result.duration / max(result.processing_ms / 1000, 1e-3)
        print(f"\n{len(result.notes)} notes | {result.duration:.0f}s of audio in {result.processing_ms} ms ({speed:.0f}x real time)", file=sys.stderr)
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("chordlistener.server.app:app", host=args.host, port=args.port, proxy_headers=False)
    return 0


def _keys(args: argparse.Namespace) -> int:
    from .server.config import Settings
    from .server.store import Store

    store = Store(Settings().db_path)
    if args.action == "create":
        if not args.name:
            print("error: --name is required", file=sys.stderr)
            return 2
        key_id, secret = store.create_key(args.name, args.per_minute, args.monthly_minutes * 60 if args.monthly_minutes else None)
        print(f"Created {key_id} for {args.name}\n\n  {secret}\n\nStore it now: only a hash is kept, so it cannot be shown again.")
    elif args.action == "list":
        for row in store.list_keys():
            quota = f"{row['monthly_seconds'] // 60} min/month" if row["monthly_seconds"] else "no quota"
            print(f"{row['id']}  {row['prefix']}…  {'active ' if row['active'] else 'revoked'}  {row['per_minute']}/min  {quota}  {row['name']}")
    elif args.action == "revoke":
        if not args.id or not store.revoke_key(args.id):
            print("error: no active key with that --id", file=sys.stderr)
            return 1
        print(f"Revoked {args.id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chordlistener", description="Audio to chords, notes and sheet music.")
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="analyse an audio file")
    a.add_argument("audio")
    a.add_argument("-o", "--out", help="directory to write JSON, MusicXML, MIDI and a chord chart into")
    a.add_argument("--json", action="store_true", help="print the full result as JSON")
    a.add_argument("--vocabulary", choices=["basic", "standard", "full"], default="standard")
    a.add_argument("--beats-per-bar", type=int, choices=[3, 4])
    a.add_argument("--tempo", choices=["auto", "half", "double"], default="auto", help="count the beat at half or double the detected speed")
    a.add_argument("--high-accuracy", action="store_true", help="use the large piano note detector (best for piano recordings; slower)")
    a.add_argument("--transpose", type=int, default=0, help="semitones")
    a.add_argument("--title", default="")
    a.set_defaults(func=_analyze)

    s = sub.add_parser("serve", help="run the website and API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=_serve)

    k = sub.add_parser("keys", help="manage API keys")
    k.add_argument("action", choices=["create", "list", "revoke"])
    k.add_argument("--name", help="who the key is for")
    k.add_argument("--id", help="key id (for revoke)")
    k.add_argument("--per-minute", type=int, default=60, help="request rate limit")
    k.add_argument("--monthly-minutes", type=int, help="monthly audio quota in minutes (default: unlimited)")
    k.set_defaults(func=_keys)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
