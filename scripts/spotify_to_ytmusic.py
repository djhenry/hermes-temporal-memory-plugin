#!/usr/bin/env python3
"""Copy all of your Spotify playlists to YouTube Music.

This is a standalone utility (unrelated to the rest of this repo). It reads every
playlist you own on Spotify, then recreates each one on YouTube Music by
searching for the closest matching track and adding it to a new playlist.

Quick start
-----------
1. Install dependencies::

       pip install spotipy ytmusicapi

2. Create a Spotify app at https://developer.spotify.com/dashboard and add
   ``http://localhost:8888/callback`` as a Redirect URI. Then export::

       export SPOTIPY_CLIENT_ID=xxxxxxxx
       export SPOTIPY_CLIENT_SECRET=xxxxxxxx
       export SPOTIPY_REDIRECT_URI=http://localhost:8888/callback

3. Authenticate with YouTube Music (creates ``oauth.json`` in the current dir)::

       ytmusicapi oauth

4. Run a dry run first to see what would be copied::

       python scripts/spotify_to_ytmusic.py --dry-run

   Then do it for real::

       python scripts/spotify_to_ytmusic.py

Useful flags
------------
--dry-run            Show what would happen without writing to YouTube Music.
--headers FILE       Path to the YouTube Music auth file (default: oauth.json).
--privacy {PRIVATE,PUBLIC,UNLISTED}
                     Privacy status for created playlists (default: PRIVATE).
--include-liked      Also copy your "Liked Songs" as a playlist.
--only NAME          Only copy playlists whose name contains NAME (repeatable).
--match-threshold N  Skip matches scored below N (0-100, default: 0 = keep best).
"""

from __future__ import annotations

import argparse
import difflib
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
except ImportError:  # pragma: no cover - import guard
    sys.exit("Missing dependency 'spotipy'. Run: pip install spotipy ytmusicapi")

try:
    from ytmusicapi import YTMusic
except ImportError:  # pragma: no cover - import guard
    sys.exit("Missing dependency 'ytmusicapi'. Run: pip install spotipy ytmusicapi")


# Scopes needed to read the user's playlists and library.
SPOTIFY_SCOPE = "playlist-read-private playlist-read-collaborative user-library-read"


@dataclass
class Track:
    """A single track pulled from Spotify."""

    title: str
    artists: list[str]
    album: str
    duration_ms: int

    @property
    def artist_str(self) -> str:
        return ", ".join(self.artists)

    @property
    def query(self) -> str:
        return f"{self.title} {self.artist_str}".strip()


@dataclass
class Playlist:
    name: str
    description: str
    tracks: list[Track] = field(default_factory=list)


@dataclass
class MigrationStats:
    playlists_created: int = 0
    tracks_matched: int = 0
    tracks_missing: int = 0
    missing: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Spotify
# --------------------------------------------------------------------------- #
def spotify_client() -> spotipy.Spotify:
    """Build an authenticated Spotify client from environment variables."""
    for var in ("SPOTIPY_CLIENT_ID", "SPOTIPY_CLIENT_SECRET", "SPOTIPY_REDIRECT_URI"):
        if not os.environ.get(var):
            sys.exit(
                f"Environment variable {var} is not set. See the module "
                "docstring for setup instructions."
            )
    auth = SpotifyOAuth(scope=SPOTIFY_SCOPE, open_browser=True)
    return spotipy.Spotify(auth_manager=auth, retries=5, requests_timeout=30)


def _track_from_item(item: dict) -> Optional[Track]:
    track = item.get("track") if "track" in item else item
    if not track or track.get("is_local") or not track.get("name"):
        return None
    return Track(
        title=track["name"],
        artists=[a["name"] for a in track.get("artists", []) if a.get("name")],
        album=(track.get("album") or {}).get("name", ""),
        duration_ms=track.get("duration_ms", 0),
    )


def fetch_playlist_tracks(sp: spotipy.Spotify, playlist_id: str) -> list[Track]:
    tracks: list[Track] = []
    results = sp.playlist_items(playlist_id, additional_types=("track",), limit=100)
    while results:
        for item in results["items"]:
            track = _track_from_item(item)
            if track:
                tracks.append(track)
        results = sp.next(results) if results.get("next") else None
    return tracks


def fetch_liked_songs(sp: spotipy.Spotify) -> list[Track]:
    tracks: list[Track] = []
    results = sp.current_user_saved_tracks(limit=50)
    while results:
        for item in results["items"]:
            track = _track_from_item(item)
            if track:
                tracks.append(track)
        results = sp.next(results) if results.get("next") else None
    return tracks


def fetch_all_playlists(
    sp: spotipy.Spotify,
    name_filters: list[str],
    include_liked: bool,
) -> list[Playlist]:
    me = sp.current_user()
    user_id = me["id"]
    print(f"Signed in to Spotify as {me.get('display_name') or user_id}")

    playlists: list[Playlist] = []
    results = sp.current_user_playlists(limit=50)
    while results:
        for pl in results["items"]:
            # Only migrate playlists the user actually owns.
            if pl["owner"]["id"] != user_id:
                continue
            if name_filters and not any(
                f.lower() in pl["name"].lower() for f in name_filters
            ):
                continue
            print(f"  Reading playlist: {pl['name']} ({pl['tracks']['total']} tracks)")
            playlists.append(
                Playlist(
                    name=pl["name"],
                    description=pl.get("description", "") or "",
                    tracks=fetch_playlist_tracks(sp, pl["id"]),
                )
            )
        results = sp.next(results) if results.get("next") else None

    if include_liked and not name_filters:
        print("  Reading Liked Songs")
        playlists.append(
            Playlist(
                name="Liked Songs (from Spotify)",
                description="Imported from Spotify Liked Songs",
                tracks=fetch_liked_songs(sp),
            )
        )
    return playlists


# --------------------------------------------------------------------------- #
# YouTube Music
# --------------------------------------------------------------------------- #
def ytmusic_client(headers_file: str) -> YTMusic:
    if not os.path.exists(headers_file):
        sys.exit(
            f"YouTube Music auth file '{headers_file}' not found. Create one with "
            "`ytmusicapi oauth` (or `ytmusicapi browser`) and pass it with --headers."
        )
    return YTMusic(headers_file)


def _score_candidate(track: Track, candidate: dict) -> float:
    """Score a YouTube Music search result against a Spotify track (0-100)."""
    title_ratio = difflib.SequenceMatcher(
        None, track.title.lower(), (candidate.get("title") or "").lower()
    ).ratio()

    cand_artists = " ".join(
        a.get("name", "") for a in candidate.get("artists", [])
    ).lower()
    artist_ratio = difflib.SequenceMatcher(
        None, track.artist_str.lower(), cand_artists
    ).ratio()

    score = (title_ratio * 0.6 + artist_ratio * 0.4) * 100

    # Bonus for a close duration match (within 3 seconds).
    cand_seconds = candidate.get("duration_seconds")
    if cand_seconds and track.duration_ms:
        if abs(cand_seconds - track.duration_ms / 1000) <= 3:
            score = min(100.0, score + 5)
    return score


def find_best_match(yt: YTMusic, track: Track, threshold: float) -> Optional[str]:
    """Return the videoId of the best matching song, or None."""
    try:
        results = yt.search(track.query, filter="songs", limit=5)
    except Exception as exc:  # noqa: BLE001 - network/library errors are non-fatal
        print(f"      ! search failed for '{track.query}': {exc}")
        return None
    if not results:
        return None

    best, best_score = None, -1.0
    for candidate in results:
        score = _score_candidate(track, candidate)
        if score > best_score:
            best, best_score = candidate, score

    if best is None or best_score < threshold:
        return None
    return best.get("videoId")


def existing_playlist_names(yt: YTMusic) -> set[str]:
    try:
        return {pl["title"] for pl in yt.get_library_playlists(limit=None)}
    except Exception:  # noqa: BLE001
        return set()


# --------------------------------------------------------------------------- #
# Migration
# --------------------------------------------------------------------------- #
def migrate(
    sp: spotipy.Spotify,
    yt: Optional[YTMusic],
    playlists: Iterable[Playlist],
    *,
    privacy: str,
    threshold: float,
    dry_run: bool,
    already_present: set[str],
) -> MigrationStats:
    stats = MigrationStats()

    for playlist in playlists:
        print(f"\n=== {playlist.name} ({len(playlist.tracks)} tracks) ===")

        if not dry_run and playlist.name in already_present:
            print("  Skipping — a playlist with this name already exists on YT Music.")
            continue

        video_ids: list[str] = []
        for track in playlist.tracks:
            if dry_run or yt is None:
                # In dry-run we still surface what we'd search for.
                print(f"  would add: {track.title} — {track.artist_str}")
                continue

            video_id = find_best_match(yt, track, threshold)
            if video_id:
                video_ids.append(video_id)
                stats.tracks_matched += 1
            else:
                stats.tracks_missing += 1
                label = f"{playlist.name}: {track.title} — {track.artist_str}"
                stats.missing.append(label)
                print(f"  no match: {track.title} — {track.artist_str}")
            time.sleep(0.05)  # be polite to the API

        if dry_run or yt is None:
            continue

        if not video_ids:
            print("  Nothing matched — not creating an empty playlist.")
            continue

        try:
            yt.create_playlist(
                title=playlist.name,
                description=playlist.description or "Imported from Spotify",
                privacy_status=privacy,
                video_ids=video_ids,
            )
            stats.playlists_created += 1
            print(f"  Created with {len(video_ids)} tracks.")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! Failed to create playlist '{playlist.name}': {exc}")

    return stats


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy all your Spotify playlists to YouTube Music.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read from Spotify and print what would be copied, without writing.",
    )
    parser.add_argument(
        "--headers",
        default="oauth.json",
        help="YouTube Music auth file (default: oauth.json).",
    )
    parser.add_argument(
        "--privacy",
        default="PRIVATE",
        choices=["PRIVATE", "PUBLIC", "UNLISTED"],
        help="Privacy status for created playlists (default: PRIVATE).",
    )
    parser.add_argument(
        "--include-liked",
        action="store_true",
        help="Also copy your Spotify 'Liked Songs' as a playlist.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="NAME",
        help="Only copy playlists whose name contains NAME (repeatable).",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.0,
        metavar="N",
        help="Skip matches scored below N (0-100). Default 0 keeps the best match.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    sp = spotify_client()
    playlists = fetch_all_playlists(sp, args.only, args.include_liked)
    if not playlists:
        print("No matching playlists found on Spotify.")
        return 0

    total_tracks = sum(len(p.tracks) for p in playlists)
    print(f"\nFound {len(playlists)} playlist(s), {total_tracks} track(s) total.")

    yt: Optional[YTMusic] = None
    already_present: set[str] = set()
    if not args.dry_run:
        yt = ytmusic_client(args.headers)
        already_present = existing_playlist_names(yt)

    stats = migrate(
        sp,
        yt,
        playlists,
        privacy=args.privacy,
        threshold=args.match_threshold,
        dry_run=args.dry_run,
        already_present=already_present,
    )

    print("\n" + "=" * 48)
    if args.dry_run:
        print("Dry run complete. No changes were made to YouTube Music.")
    else:
        print(f"Playlists created : {stats.playlists_created}")
        print(f"Tracks matched    : {stats.tracks_matched}")
        print(f"Tracks not found  : {stats.tracks_missing}")
        if stats.missing:
            print("\nCould not find matches for:")
            for item in stats.missing:
                print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
