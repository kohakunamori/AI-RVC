import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import replace_release_packages as replacement


class ReplacePortablePackagesTests(unittest.TestCase):
    def setUp(self):
        self.previous = "a" * 40
        self.source = "b" * 40
        self.run = {"id": 123, "head_sha": self.source, "html_url": "https://github.com/owner/repo/actions/runs/123"}
        self.release = {"id": 10, "draft": False, "immutable": False, "tag_name": "v1.6.0",
                        "target_commitish": self.previous, "body": "Original release", "assets": []}
        self.tag = {"type": "commit", "sha": self.previous}
        self.writes = []
        self.fail_tag_update = False
        self.corrupt_pending = False

    def api(self, path, method="GET", payload=None):
        if method != "GET":
            self.writes.append((path, method, payload))
        if "/git/ref/tags/" in path:
            return {"object": copy.deepcopy(self.tag)}
        if "/git/refs/tags/" in path:
            if self.fail_tag_update:
                self.fail_tag_update = False
                raise RuntimeError("Injected tag update failure")
            self.tag["sha"] = payload["sha"]
            return {"object": copy.deepcopy(self.tag)}
        if "/releases/assets/" in path:
            asset_id = int(path.rsplit("/", 1)[1])
            item = next(item for item in self.release["assets"] if item["id"] == asset_id)
            if method == "DELETE":
                self.release["assets"].remove(item)
                return None
            item.update(payload)
            return copy.deepcopy(item)
        if method == "PATCH":
            self.release.update(payload)
        # Downloads continue while uploads are staged; counters are not asset identity.
        for item in self.release["assets"]:
            item["download_count"] = item.get("download_count", 0) + 1
        return copy.deepcopy(self.release)

    def upload(self, prefix, current, path, name, entry):
        item = dict(id=100 + len(self.release["assets"]), name=name, state="uploaded",
                    size=entry["bytes"], digest="sha256:" + entry["sha256"])
        if self.corrupt_pending:
            item["digest"] = "sha256:" + "0" * 64
        self.release["assets"].append(item)
        return copy.deepcopy(item)

    def execute(self, root):
        with patch.object(replacement, "api", side_effect=self.api), \
             patch.object(replacement, "upload_pending", side_effect=self.upload), \
             patch.object(replacement, "source_text", side_effect=lambda prefix, source, path: "1.6.0" if path == "VERSION" else "修复\n\n## English\nFix"):
            return replacement.replace("owner/repo", self.run, root, self.previous, root.parent / "receipt.json")

    def packages(self, root):
        for platform in ("Windows", "Linux"):
            for variant in ("CPU", "GPU"):
                name = f"AI-RVC-{platform}-{variant}"
                directory = root / name
                directory.mkdir(parents=True)
                filename = name + ("-Portable.7z.001" if platform == "Windows" else "-Portable.tar.gz")
                content = (name + " rebuilt").encode()
                (directory / filename).write_bytes(content)
                digest = hashlib.sha256(content).hexdigest()
                checksum = name + "-Portable.SHA256SUMS"
                (directory / checksum).write_text(f"{digest}  {filename}\n", encoding="utf-8")
                for old_name in (filename, checksum):
                    self.release["assets"].append(dict(id=len(self.release["assets"]) + 1, name=old_name,
                        state="uploaded", size=3, digest="sha256:" + hashlib.sha256(b"old").hexdigest()))

    def test_replaces_only_after_all_volumes_pass_and_keeps_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "packages"
            self.packages(root)
            old_ids = {item["id"] for item in self.release["assets"]}
            record = self.execute(root)
            self.assertEqual(record["status"], "replaced")
            self.assertEqual(self.release["tag_name"], "v1.6.0")
            self.assertEqual(self.tag["sha"], self.source)
            self.assertEqual(self.release["target_commitish"], self.source)
            self.assertFalse(old_ids & {item["id"] for item in self.release["assets"]})
            replacement.verify_assets(self.release["assets"], record["expected"])
            self.assertEqual(json.loads((root.parent / "receipt.json").read_text())["previous_tag"]["sha"], self.previous)
            first_delete = next(i for i, operation in enumerate(self.writes) if operation[1] == "DELETE")
            tag_update = next(i for i, operation in enumerate(self.writes) if "/git/refs/" in operation[0])
            self.assertGreater(first_delete, tag_update)

    def test_tag_failure_restores_original_downloads_without_deleting_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "packages"
            self.packages(root)
            original = copy.deepcopy(self.release)
            self.fail_tag_update = True
            with self.assertRaisesRegex(RuntimeError, "Injected tag"):
                self.execute(root)
            self.assertEqual([(a["id"], a["name"], a["digest"]) for a in self.release["assets"][:8]],
                             [(a["id"], a["name"], a["digest"]) for a in original["assets"]])
            self.assertEqual(self.release["body"], original["body"])
            self.assertEqual(self.tag["sha"], self.previous)
            self.assertFalse(any(operation[1] == "DELETE" for operation in self.writes))

    def test_corrupt_remote_volume_prevents_tag_change_and_restores_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "packages"
            self.packages(root)
            original = copy.deepcopy(self.release["assets"])
            self.corrupt_pending = True
            with self.assertRaisesRegex(RuntimeError, "verification failed"):
                self.execute(root)
            self.assertEqual([(a["id"], a["name"], a["digest"]) for a in self.release["assets"][:8]],
                             [(a["id"], a["name"], a["digest"]) for a in original])
            self.assertEqual(self.tag["sha"], self.previous)
            self.assertFalse(any(operation[1] == "DELETE" for operation in self.writes))

    def test_changed_tag_and_unrelated_assets_are_rejected_before_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "packages"
            self.packages(root)
            self.tag["sha"] = "c" * 40
            with self.assertRaisesRegex(RuntimeError, "expected replaceable"):
                self.execute(root)
            self.tag["sha"] = self.previous
            self.release["assets"].append(dict(id=90, name="user-backup.zip", digest="sha256:" + "0" * 64))
            with self.assertRaisesRegex(RuntimeError, "unrelated"):
                self.execute(root)
            self.assertEqual(self.writes, [])

    def test_partial_platform_build_is_rejected(self):
        run = dict(self.run, status="completed", conclusion="success", head_branch="master",
                   path=".github/workflows/build-executables.yml", head_repository={"full_name": "owner/repo"})
        names = [f"AI-RVC-{platform}-{variant}" for platform in ("Windows", "Linux") for variant in ("CPU", "GPU")]
        artifacts = {"total_count": 4, "artifacts": [dict(name=name, expired=False) for name in names]}
        jobs = {"total_count": 4, "jobs": [dict(name=name.replace("AI-RVC-", "Build "),
                status="completed", conclusion="skipped" if name.endswith("Linux-GPU") else "success") for name in names]}
        with patch.object(replacement, "api", side_effect=[run, artifacts, jobs]):
            with self.assertRaisesRegex(RuntimeError, "Every platform"):
                replacement.verified_build("owner/repo", "123")


if __name__ == "__main__":
    unittest.main()
