from pathlib import Path

from utils.release import build_release_manifest, sign_manifest_hmac, verify_manifest_hmac


def test_release_manifest_build_and_sign(tmp_path: Path) -> None:
    a = tmp_path / "a.whl"
    b = tmp_path / "b.tar.gz"
    a.write_bytes(b"alpha")
    b.write_bytes(b"beta")

    manifest = build_release_manifest([a, b], version="v1.2.3")
    assert manifest["version"] == "v1.2.3"
    assert "build" in manifest
    assert len(manifest["artifacts"]) == 2
    assert all("sha256" in x for x in manifest["artifacts"])

    signed = sign_manifest_hmac(manifest, "secret")
    assert "signature" in signed
    assert signed["signature"]["type"] == "hmac-sha256"


def test_release_manifest_signing_is_idempotent() -> None:
    manifest = {
        "version": "v9.9.9",
        "generated_at": "2026-02-14T00:00:00+00:00",
        "algorithm": "sha256",
        "artifacts": [],
    }
    signed_once = sign_manifest_hmac(manifest, "secret")
    signed_twice = sign_manifest_hmac(signed_once, "secret")

    assert signed_once["signature"]["value"] == signed_twice["signature"]["value"]
    assert signed_twice["signature"]["type"] == "hmac-sha256"


def test_release_manifest_hmac_verify_roundtrip() -> None:
    manifest = {
        "version": "v1.0.0",
        "generated_at": "2026-02-14T00:00:00+00:00",
        "algorithm": "sha256",
        "artifacts": [],
    }
    signed = sign_manifest_hmac(manifest, "secret")
    assert verify_manifest_hmac(signed, "secret") is True
    assert verify_manifest_hmac(signed, "wrong-secret") is False
