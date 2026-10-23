"""Stage existing, checksum-verified CI packages without rebuilding or publishing."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path


def gh(*arguments: str) -> str:
    return subprocess.check_output(["gh", *arguments], text=True)


def api(path: str):
    return json.loads(gh("api", path))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    repository = os.environ["GITHUB_REPOSITORY"]
    run_id = os.environ["SOURCE_RUN_ID"]
    source = os.environ["SOURCE_SHA"]
    tag = os.environ["RELEASE_TAG"]
    if not re.fullmatch(r"[1-9][0-9]*", run_id) or not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("An exact run ID and 40-character source commit are required")
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag):
        raise ValueError("Release tag must be vMAJOR.MINOR.PATCH")
    prefix = f"repos/{repository}"
    run = api(f"{prefix}/actions/runs/{run_id}")
    if (run["status"], run["conclusion"], run["head_sha"]) != ("completed", "success", source):
        raise RuntimeError("The source build has not completed successfully at the requested commit")
    if run["path"] != ".github/workflows/build-executables.yml" or run["head_repository"]["full_name"] != repository:
        raise RuntimeError("Artifacts must come from this repository's portable build workflow")
    version = base64.b64decode(api(f"{prefix}/contents/VERSION?ref={source}")["content"]).decode().strip()
    if tag != f"v{version}":
        raise RuntimeError("Release tag differs from the built source VERSION")
    # GitHub's by-tag REST endpoint excludes unpublished releases without a tag.
    # gh resolves those drafts through the release list; subsequent reads use ID.
    release_id = json.loads(gh("release", "view", tag, "--repo", repository, "--json", "id"))["id"]

    def draft():
        release = api(f"{prefix}/releases/{release_id}")
        if not release["draft"] or release["target_commitish"] != source or release["tag_name"] != tag:
            raise RuntimeError("The destination must remain a draft targeting the exact built commit")
        return release

    draft()
    expected = {f"AI-RVC-{platform}-{variant}" for platform in ("Windows", "Linux") for variant in ("CPU", "GPU")}
    response = api(f"{prefix}/actions/runs/{run_id}/artifacts?per_page=100")
    artifacts = response["artifacts"]
    if response["total_count"] != 4 or {item["name"] for item in artifacts} != expected:
        raise RuntimeError("Exactly the four expected portable artifacts are required")
    if any(item["expired"] for item in artifacts):
        raise RuntimeError("A source artifact has expired")

    with tempfile.TemporaryDirectory(prefix="ai-rvc-stage-") as workspace:
        root = Path(workspace)
        gh("run", "download", run_id, "--repo", repository, "--dir", str(root))
        verified = {}
        uploads = []
        for name in sorted(expected):
            directory = root / name
            checksum = directory / f"{name}-Portable.SHA256SUMS"
            entries = {}
            for line in checksum.read_text(encoding="utf-8").splitlines():
                match = re.fullmatch(r"([0-9a-f]{64})\s+\*?([^/\\]+)", line)
                if not match:
                    raise RuntimeError(f"Invalid checksum line in {checksum.name}")
                digest, filename = match.groups()
                if not re.fullmatch(re.escape(name + "-Portable") + r"(?:\.7z\.[0-9]{3}|\.tar\.gz(?:\.part[a-z]{2})?)", filename):
                    raise RuntimeError(f"Unexpected archive name: {filename}")
                path = directory / filename
                if filename in entries or sha256(path) != digest:
                    raise RuntimeError(f"Duplicate or mismatched archive: {filename}")
                entries[filename] = {"sha256": digest, "bytes": path.stat().st_size}
            if not entries or {path.name for path in directory.iterdir()} != set(entries) | {checksum.name}:
                raise RuntimeError(f"Unexpected or unlisted contents in {name}")
            entries[checksum.name] = {"sha256": sha256(checksum), "bytes": checksum.stat().st_size}
            verified[name] = entries
            uploads.extend(directory / filename for filename in sorted(entries))
            print(f"Verified {name}: {len(entries) - 1} archive volumes", flush=True)

        provenance = root / "DRAFT-ASSET-PROVENANCE.json"
        provenance.write_text(json.dumps({
            "version": version, "source_sha": source, "build_run_id": int(run_id),
            "build_url": run["html_url"], "artifacts": verified,
            "scope": "Original CI archives and checksums; runtime acceptance is recorded separately. Not published.",
        }, indent=2) + "\n", encoding="utf-8")
        uploads.append(provenance)
        for path in uploads:
            existing = {asset["name"]: asset for asset in draft()["assets"]}
            if path.name in existing:
                if existing[path.name].get("digest") != "sha256:" + sha256(path):
                    raise RuntimeError(f"Refusing to replace a differing or unverifiable asset: {path.name}")
                print(f"Already staged and identical: {path.name}", flush=True)
                continue
            gh("release", "upload", tag, str(path), "--repo", repository)
            print(f"Staged: {path.name}", flush=True)
        draft()
        print("All four original packages are attached to the draft; no release was published.")


if __name__ == "__main__":
    main()
