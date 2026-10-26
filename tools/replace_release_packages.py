"""Explicitly rebuild an existing version from a verified portable Actions run."""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.parse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.release_pipeline import api, PACKAGE_NAME, verify_assets
from tools.stage_release_draft import verified_package_files


def verified_build(repository: str, run_id: str) -> dict:
    if not re.fullmatch(r"[1-9][0-9]*", run_id):
        raise ValueError("A completed build run ID is required")
    prefix = f"repos/{repository}"
    run = api(f"{prefix}/actions/runs/{run_id}")
    if (run["status"], run["conclusion"], run["path"], run["head_branch"], run["head_repository"]["full_name"]) != (
        "completed", "success", ".github/workflows/build-executables.yml", "master", repository
    ):
        raise RuntimeError("Packages must come from a successful master portable build in this repository")
    expected = {f"AI-RVC-{platform}-{variant}" for platform in ("Windows", "Linux") for variant in ("CPU", "GPU")}
    artifacts = api(f"{prefix}/actions/runs/{run_id}/artifacts?per_page=100")
    if artifacts["total_count"] != 4 or {item["name"] for item in artifacts["artifacts"]} != expected or any(item["expired"] for item in artifacts["artifacts"]):
        raise RuntimeError("All four original build artifacts must be available")
    jobs = api(f"{prefix}/actions/runs/{run_id}/jobs?per_page=100")
    required = {name.replace("AI-RVC-", "Build ") for name in expected}
    passed = {item["name"] for item in jobs["jobs"] if (item["status"], item["conclusion"]) == ("completed", "success")}
    if jobs["total_count"] > 100 or not required <= passed:
        raise RuntimeError("Every platform/variant build must pass before replacing the release")
    return run


def source_text(prefix: str, source: str, path: str) -> str:
    return base64.b64decode(api(f"{prefix}/contents/{path}?ref={source}")["content"]).decode("utf-8-sig").strip()


def upload_pending(prefix: str, current: dict, path: Path, name: str, entry: dict) -> dict:
    # Upload all new bytes under temporary names, leaving current downloads intact.
    existing = [item for item in api(f"{prefix}/releases/{current['id']}")["assets"] if item["name"] == name]
    if existing:
        verify_assets(existing, {name: entry})
        return existing[0]
    endpoint = f"https://uploads.github.com/{prefix}/releases/{current['id']}/assets?name={urllib.parse.quote(name)}"
    try:
        result = subprocess.check_output(
            ["gh", "api", endpoint, "--method", "POST", "-H", "Content-Type: application/octet-stream", "--input", str(path)],
            text=True, encoding="utf-8", timeout=1800,
        )
        item = json.loads(result)
    except (subprocess.SubprocessError, json.JSONDecodeError):
        # A lost response does not mean the upload failed; verify remote state.
        existing = [item for item in api(f"{prefix}/releases/{current['id']}")["assets"] if item["name"] == name]
        verify_assets(existing, {name: entry})
        return existing[0]
    verify_assets([item], {name: entry})
    return item


def replace(repository: str, run: dict, root: Path, previous_sha: str, receipt: Path) -> dict:
    prefix = f"repos/{repository}"
    source = run["head_sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", previous_sha) or not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("Exact previous and rebuilt source commits are required")
    version = source_text(prefix, source, "VERSION")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("Invalid source VERSION")
    tag = "v" + version
    current = api(f"{prefix}/releases/tags/{tag}")
    tag_path = f"{prefix}/git/refs/tags/{tag}"
    previous_tag = api(f"{prefix}/git/ref/tags/{tag}")["object"]
    if (current["draft"] or current.get("immutable") or current["tag_name"] != tag
            or previous_tag["type"] != "commit" or previous_tag["sha"] != previous_sha):
        raise RuntimeError("The existing public release/tag differs from the expected replaceable version")
    if current["target_commitish"] != previous_sha:
        raise RuntimeError("The existing release source changed; no assets were replaced")
    notes = source_text(prefix, source, f"docs/releases/{tag}.md")
    if "## English" not in notes or not re.search(r"[\u3400-\u9fff]", notes):
        raise RuntimeError("Release notes must cover both supported languages")
    verified, paths = verified_package_files(root)
    expected = {name: entry for entries in verified.values() for name, entry in entries.items()}
    suffix = f".rebuild-{run['id']}"
    old = [item for item in current["assets"] if not item["name"].endswith(suffix)]
    if not old or any(not PACKAGE_NAME.fullmatch(item["name"]) or item.get("state") != "uploaded" or not item.get("digest") for item in old):
        raise RuntimeError("The existing release contains unrelated or unverified assets")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    record = {"previous_release": current, "previous_tag": previous_tag, "source": source,
              "run_id": run["id"], "expected": expected, "status": "uploading"}
    receipt.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    pending = []
    for path in paths:
        pending.append(upload_pending(prefix, current, path, path.name + suffix, expected[path.name]))
        print(f"Verified pending asset: {path.name}", flush=True)
    fresh = api(f"{prefix}/releases/{current['id']}")
    old_ids = {item["id"] for item in old}
    def identity(item):
        return tuple(item.get(key) for key in ("id", "name", "digest", "size", "state"))

    if ({identity(item) for item in fresh["assets"] if item["id"] in old_ids} != {identity(item) for item in old}
            or {item["id"] for item in fresh["assets"]} != old_ids | {item["id"] for item in pending}
            or fresh["target_commitish"] != current["target_commitish"] or fresh["body"] != current["body"]
            or api(f"{prefix}/git/ref/tags/{tag}")["object"] != previous_tag):
        raise RuntimeError("Release assets or tag changed while uploading; downloads were not replaced")
    release_path = f"{prefix}/releases/{current['id']}"
    try:
        for item in old:
            api(f"{prefix}/releases/assets/{item['id']}", "PATCH", {"name": item["name"] + suffix + ".previous"})
        for item in pending:
            name = item["name"][:-len(suffix)]
            api(f"{prefix}/releases/assets/{item['id']}", "PATCH", {"name": name})
        fresh = api(release_path)
        verify_assets([item for item in fresh["assets"] if item["id"] not in old_ids], expected)
        api(tag_path, "PATCH", {"sha": source, "force": True})
        build_note = f"\n\n便携包重新打包 / Portable rebuild: [{source[:12]}]({run['html_url']}).\n"
        api(release_path, "PATCH", {"target_commitish": source, "body": notes + build_note})
        if api(f"{prefix}/git/ref/tags/{tag}")["object"]["sha"] != source:
            raise RuntimeError("The existing version tag did not move to the verified build")
    except Exception:
        # All original bytes are still present. Restore names and metadata before failing.
        # Read actual names: a PATCH may have succeeded before its response was lost.
        actual = {item["id"]: item for item in api(release_path)["assets"]}
        for item in pending:
            if actual[item["id"]]["name"] != item["name"]:
                api(f"{prefix}/releases/assets/{item['id']}", "PATCH", {"name": item["name"]})
        for item in old:
            if actual[item["id"]]["name"] != item["name"]:
                api(f"{prefix}/releases/assets/{item['id']}", "PATCH", {"name": item["name"]})
        if api(f"{prefix}/git/ref/tags/{tag}")["object"]["sha"] == source:
            api(tag_path, "PATCH", {"sha": previous_sha, "force": True})
        api(release_path, "PATCH", {"target_commitish": current["target_commitish"], "body": current["body"]})
        raise
    # Delete only superseded IDs after every replacement and the existing tag pass.
    obsolete = {item["id"]: item for item in api(release_path)["assets"] if item["id"] in old_ids}
    for item in old:
        if identity(obsolete[item["id"]]) != identity({**item, "name": item["name"] + suffix + ".previous"}):
            raise RuntimeError("A superseded asset changed; refusing to delete it")
        api(f"{prefix}/releases/assets/{item['id']}", "DELETE")
    verify_assets(api(release_path)["assets"], expected)
    record["status"] = "replaced"
    receipt.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--previous-sha", required=True)
    parser.add_argument("--packages", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    repository = os.environ["GITHUB_REPOSITORY"]
    run = verified_build(repository, args.run_id)
    result = replace(repository, run, args.packages, args.previous_sha, args.receipt)
    print(f"Replaced existing release packages with verified commit {result['source']}", flush=True)


if __name__ == "__main__":
    main()
