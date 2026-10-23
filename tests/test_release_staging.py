import base64
import copy
import hashlib
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import stage_release_draft as staging


class ReleaseStagingTests(unittest.TestCase):
    def setUp(self):
        self.source = "a" * 40
        self.names = [f"AI-RVC-{platform}-{variant}" for platform in ("Windows", "Linux") for variant in ("CPU", "GPU")]
        self.run = dict(status="completed", conclusion="success", head_sha=self.source,
                        path=".github/workflows/build-executables.yml", head_repository={"full_name": "owner/repo"},
                        html_url="https://github.com/owner/repo/actions/runs/123")
        self.release = dict(id=42, draft=True, target_commitish=self.source, tag_name="v1.6.0", assets=[])
        self.artifacts = dict(total_count=4, artifacts=[dict(name=name, expired=False) for name in self.names])
        self.version = "1.6.0"
        self.corrupt = False
        self.uploaded = []

    def api(self, path):
        if "/artifacts?" in path:
            return copy.deepcopy(self.artifacts)
        if "/actions/runs/" in path:
            return copy.deepcopy(self.run)
        if "/contents/VERSION?" in path:
            return {"content": base64.b64encode(self.version.encode()).decode()}
        if "/releases?" in path:
            return [copy.deepcopy(self.release)]
        if path.endswith("/releases/42"):
            return copy.deepcopy(self.release)
        raise AssertionError(path)

    def gh(self, *args):
        if args[:2] == ("run", "download"):
            root = Path(args[-1])
            for name in self.names:
                directory = root / name
                directory.mkdir()
                filename = name + ("-Portable.7z.001" if "Windows" in name else "-Portable.tar.gz.partaa")
                data = name.encode()
                (directory / filename).write_bytes(data)
                digest = "0" * 64 if self.corrupt else hashlib.sha256(data).hexdigest()
                (directory / f"{name}-Portable.SHA256SUMS").write_text(f"{digest}  ./{filename}\n")
        elif args[:2] == ("release", "upload"):
            path = Path(args[3])
            self.uploaded.append(path.name)
            self.release["assets"].append(dict(name=path.name, digest="sha256:" + staging.sha256(path)))
        else:
            raise AssertionError(args)
        return ""

    def execute(self):
        with patch.dict(os.environ, dict(GITHUB_REPOSITORY="owner/repo", SOURCE_RUN_ID="123", SOURCE_SHA=self.source, RELEASE_TAG="v1.6.0")), \
             patch.object(staging, "api", side_effect=self.api), patch.object(staging, "gh", side_effect=self.gh):
            staging.main()

    def reject(self):
        with self.assertRaises(RuntimeError):
            self.execute()
        self.assertEqual(self.uploaded, [])

    def test_success_stages_originals_and_is_idempotent(self):
        self.execute()
        self.assertEqual(len(self.uploaded), 9)
        self.execute()
        self.assertEqual(len(self.uploaded), 9)
        self.assertTrue(self.release["draft"])

    def test_incomplete_build_is_rejected(self):
        self.run["status"] = "in_progress"
        self.reject()

    def test_failed_build_is_rejected(self):
        self.run["conclusion"] = "failure"
        self.reject()

    def test_wrong_build_commit_is_rejected(self):
        self.run["head_sha"] = "b" * 40
        self.reject()

    def test_wrong_workflow_is_rejected(self):
        self.run["path"] = ".github/workflows/other.yml"
        self.reject()

    def test_fork_artifacts_are_rejected(self):
        self.run["head_repository"]["full_name"] = "other/repo"
        self.reject()

    def test_published_release_is_rejected(self):
        self.release["draft"] = False
        self.reject()

    def test_wrong_draft_target_is_rejected(self):
        self.release["target_commitish"] = "master"
        self.reject()

    def test_mismatched_version_is_rejected(self):
        self.version = "1.5.1"
        self.reject()

    def test_missing_package_is_rejected(self):
        self.artifacts["artifacts"].pop()
        self.reject()

    def test_expired_artifact_is_rejected(self):
        self.artifacts["artifacts"][0]["expired"] = True
        self.reject()

    def test_corrupt_volume_is_rejected_before_upload(self):
        self.corrupt = True
        self.reject()

    def test_different_existing_asset_is_never_overwritten(self):
        self.release["assets"] = [dict(name="AI-RVC-Linux-CPU-Portable.SHA256SUMS", digest="sha256:" + "0" * 64)]
        self.reject()


if __name__ == "__main__":
    unittest.main()
