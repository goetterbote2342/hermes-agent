"""Unit tests for the Matrix cross-signing signature helper.

Covers the two directions the helper publishes — our own device signed by our
self-signing key, and the other user's master key signed by our user-signing key —
without libolm or mautrix: the signed object is built as plain canonical JSON, so
everything here runs with fakes only.
"""

import asyncio
import types
from typing import Any

from plugins.platforms.matrix.cross_signing import (
    CrossSigningSigner,
    _canonical_json,
)


class _FakeSigningKey:
    """Stand-in for an olm.PkSigning (only sign() and public_key are used)."""

    def __init__(self, public_key="USK"):
        self.public_key = public_key
        self.signed = []

    def sign(self, data):
        self.signed.append(data)
        return f"SIG<{self.public_key}:{data}>"


class _FakePrivateKeys:
    def __init__(self, user_signing_key=None):
        self.user_signing_key = user_signing_key or _FakeSigningKey()


class _FakeCryptoStore:
    def __init__(self):
        self.signatures = []

    async def put_signature(self, target, signer, signature):
        self.signatures.append((target, signer, signature))


class _FakeOlm:
    def __init__(self, keys=None, master_key="ALICEMSK"):
        self._cross_signing_private_keys: Any = keys
        self.crypto_store = _FakeCryptoStore()
        self.own_identity: Any = types.SimpleNamespace(
            user_id="@bot:example.org", device_id="BOTDEV"
        )
        self._master_key: Any = master_key
        self.signed_own: list = []
        self.fail_sign_own: Any = None

    async def get_cross_signing_public_keys(self, user_id):
        if self._master_key is None:
            return None
        return types.SimpleNamespace(master_key=self._master_key)

    async def sign_own_device(self, device):
        if self.fail_sign_own is not None:
            raise self.fail_sign_own
        self.signed_own.append(device)


class _FakeClient:
    def __init__(self):
        self.mxid = "@bot:example.org"
        self.device_id = "BOTDEV"
        self.uploads = []

    async def upload_one_signature(self, user_id, key_id, keys):
        self.uploads.append((user_id, key_id, keys))


def _signer(olm):
    return CrossSigningSigner(adapter=None, client=_FakeClient(), olm=olm)


class TestSignOwnDevice:
    def test_without_private_keys_nothing_is_signed(self):
        olm = _FakeOlm(keys=None)
        signer = _signer(olm)
        assert signer.has_identity() is False
        assert asyncio.run(signer.sign_own_device()) is False
        assert olm.signed_own == []

    def test_without_device_identity_nothing_is_signed(self):
        olm = _FakeOlm(keys=_FakePrivateKeys())
        olm.own_identity = None
        assert asyncio.run(_signer(olm).sign_own_device()) is False
        assert olm.signed_own == []

    def test_signs_our_own_device(self):
        olm = _FakeOlm(keys=_FakePrivateKeys())
        signer = _signer(olm)
        assert signer.has_identity() is True
        assert asyncio.run(signer.sign_own_device()) is True
        assert olm.signed_own == [olm.own_identity]

    def test_failure_is_not_fatal(self):
        olm = _FakeOlm(keys=_FakePrivateKeys())
        olm.fail_sign_own = RuntimeError("homeserver down")
        assert asyncio.run(_signer(olm).sign_own_device()) is False


class TestTrustUser:
    def test_requires_our_private_keys(self):
        olm = _FakeOlm(keys=None)
        signer = _signer(olm)
        assert asyncio.run(signer.trust_user("@alice:example.org", ["ed25519:ALICEMSK"])) is False
        assert signer._client.uploads == []

    def test_never_signs_our_own_identity(self):
        olm = _FakeOlm(keys=_FakePrivateKeys())
        signer = _signer(olm)
        assert asyncio.run(signer.trust_user("@bot:example.org", ["ed25519:ALICEMSK"])) is False
        assert signer._client.uploads == []

    def test_requires_a_published_master_key(self):
        olm = _FakeOlm(keys=_FakePrivateKeys(), master_key=None)
        signer = _signer(olm)
        assert asyncio.run(signer.trust_user("@alice:example.org", ["ed25519:ALICEMSK"])) is False
        assert signer._client.uploads == []

    def test_session_verification_does_not_sign_the_identity(self):
        """A MAC set that only covers the device key must not verify the identity."""
        olm = _FakeOlm(keys=_FakePrivateKeys())
        signer = _signer(olm)
        ok = asyncio.run(
            signer.trust_user("@alice:example.org", ["ed25519:ALICEDEV"])
        )
        assert ok is False
        assert signer._client.uploads == []

    def test_signs_the_verified_master_key(self):
        usk = _FakeSigningKey("USK")
        olm = _FakeOlm(keys=_FakePrivateKeys(usk), master_key="ALICEMSK")
        signer = _signer(olm)

        assert asyncio.run(signer.trust_user("@alice:example.org", ["ed25519:ALICEMSK"])) is True

        assert len(signer._client.uploads) == 1
        user_id, key_id, obj = signer._client.uploads[0]
        assert (user_id, key_id) == ("@alice:example.org", "ed25519:ALICEMSK")
        payload = obj.serialize()
        assert payload["user_id"] == "@alice:example.org"
        assert payload["usage"] == ["master"]
        assert payload["keys"] == {"ed25519:ALICEMSK": "ALICEMSK"}

        # The signature is our user-signing key over the canonical form of the
        # object without signatures/unsigned.
        expected_payload = _canonical_json(
            {
                "user_id": "@alice:example.org",
                "usage": ["master"],
                "keys": {"ed25519:ALICEMSK": "ALICEMSK"},
            }
        )
        assert usk.signed == [expected_payload]
        assert payload["signatures"] == {
            "@bot:example.org": {"ed25519:USK": f"SIG<USK:{expected_payload}>"}
        }

        # And it is stored locally so trust questions answer without the network.
        assert olm.crypto_store.signatures == [
            (
                ("@alice:example.org", "ALICEMSK"),
                ("@bot:example.org", "USK"),
                f"SIG<USK:{expected_payload}>",
            )
        ]

    def test_upload_failure_is_not_fatal(self):
        class _BrokenClient(_FakeClient):
            async def upload_one_signature(self, user_id, key_id, keys):
                raise RuntimeError("M_FORBIDDEN")

        olm = _FakeOlm(keys=_FakePrivateKeys())
        signer = CrossSigningSigner(adapter=None, client=_BrokenClient(), olm=olm)
        assert asyncio.run(signer.trust_user("@alice:example.org", ["ed25519:ALICEMSK"])) is False
        assert olm.crypto_store.signatures == []


class TestCanonicalJson:
    def test_matches_matrix_canonical_form(self):
        assert _canonical_json({"b": 1, "a": [2, {"d": 4, "c": 3}]}) == (
            '{"a":[2,{"c":3,"d":4}],"b":1}'
        )
