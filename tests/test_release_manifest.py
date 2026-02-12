from pathlib import Path

from utils.release import build_release_manifest, sign_manifest_hmac


def test_release_manifest_build_and_sign(tmp_path: Path):
    a = tmp_path / "a.whl"
    b = tmp_path / "b.tar.gz"
    a.write_bytes(b"alpha")
    b.write_bytes(b"beta")

    manifest = build_release_manifest([a, b], version="v1.2.3")
    assert manifest["version"] == "v1.2.3"
    assert len(manifest["artifacts"]) == 2
    assert all("sha256" in x for x in manifest["artifacts"])

    signed = sign_manifest_hmac(manifest, "secret")
    assert "signature" in signed
    assert signed["signature"]["type"] == "hmac-sha256"
