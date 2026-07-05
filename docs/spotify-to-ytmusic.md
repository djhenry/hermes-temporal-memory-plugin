# Spotify → YouTube Music playlist migration

`scripts/spotify_to_ytmusic.py` copies every playlist you own on Spotify over to
YouTube Music. It reads each Spotify playlist, searches YouTube Music for the
closest matching track (title + artist, with a small duration bonus), and
creates a matching playlist on your YouTube Music account.

> This utility is standalone and independent of the rest of this repository.

## 1. Install dependencies

```bash
pip install -r scripts/requirements-spotify-migration.txt
```

## 2. Authorize Spotify

1. Create an app at <https://developer.spotify.com/dashboard>.
2. Add `http://localhost:8888/callback` under the app's Redirect URIs.
3. Export the credentials:

   ```bash
   export SPOTIPY_CLIENT_ID=xxxxxxxxxxxx
   export SPOTIPY_CLIENT_SECRET=xxxxxxxxxxxx
   export SPOTIPY_REDIRECT_URI=http://localhost:8888/callback
   ```

The first run opens a browser to grant read access to your playlists
(read-only — the script never modifies your Spotify account).

## 3. Authorize YouTube Music

```bash
ytmusicapi oauth
```

This writes an `oauth.json` file in the current directory. Point the script at a
different location with `--headers path/to/oauth.json` if needed.

## 4. Run it

Preview first (no changes are made):

```bash
python scripts/spotify_to_ytmusic.py --dry-run
```

Then run for real:

```bash
python scripts/spotify_to_ytmusic.py
```

## Options

| Flag | Description |
| --- | --- |
| `--dry-run` | Print what would be copied without writing to YouTube Music. |
| `--headers FILE` | YouTube Music auth file (default `oauth.json`). |
| `--privacy {PRIVATE,PUBLIC,UNLISTED}` | Privacy of created playlists (default `PRIVATE`). |
| `--include-liked` | Also copy your Spotify "Liked Songs" as a playlist. |
| `--only NAME` | Only copy playlists whose name contains `NAME` (repeatable). |
| `--match-threshold N` | Skip matches scored below `N` (0–100). Default keeps the best match. |

## Notes

- Only playlists **you own** are copied (collaborative/followed playlists from
  others are skipped).
- Playlists whose name already exists on YouTube Music are skipped, so re-running
  is safe and won't create duplicates.
- Tracks with no confident match are reported in a summary at the end so you can
  add them manually.
- Matching is best-effort: YouTube Music's catalog differs from Spotify's, so a
  few tracks may not transfer or may match a slightly different version.
