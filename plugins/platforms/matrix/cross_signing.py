"""Cross-signing signatures for the Matrix adapter.

SAS verification (``verification.py``) proves that two clients hold the keys
they claim to hold.  What makes that proof *stick* is a signature: Matrix has
no "verified" bit, only published signatures chaining a device key to a user's
master key, and a master key to another user's identity.

Two directions matter here:

``self-signing``
    Our own device key signed by our self-signing key, so our device chains to
    our own identity (device <- self-signing key <- master key).  Without it
    clients show the bot's session as an unverified device even after a
    successful emoji comparison.  ``OlmMachine.sign_own_device()`` does the
    signing; this module calls it when the private keys are available and keeps
    every failure non-fatal.

``user-signing``
    The other user's *master key* signed by our user-signing key — the way one
    identity declares another identity verified.  Element decides on the green
    shield from exactly this signature, so without it the verdict only ever
    exists in the user's client and never on our side.

Both need the private half of our cross-signing identity.  mautrix keeps it in
memory after ``verify_with_recovery_key()``/``generate_recovery_key()`` (see
``_verify_or_bootstrap_cross_signing`` in ``adapter.py``); it is reached through
a guarded ``getattr`` shim, the same way the rest of this platform reaches
mautrix internals, so a mautrix upgrade degrades to "no signature" instead of an
import-time breakage.

Deliberately mautrix-free: the signed object is built as plain JSON exactly like
the spec's canonical form, so this module stays unit-testable on hosts without
python-olm.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, NamedTuple

logger = logging.getLogger("hermes.matrix.cross_signing")

# Cross-signing key usages, spelled the way the spec serializes them.
_USAGE_MASTER = "master"


class _CrossSigner(NamedTuple):
    """(user_id, key) pair as mautrix's crypto store expects it."""

    user_id: str
    key: str


@dataclass
class _SignedMasterKey:
    """A user's master key plus the signatures we are uploading for it.

    Duck-typed to what ``Client.upload_one_signature()`` accepts: mautrix calls
    ``.serialize()`` on whatever it is handed, so an object of ours works
    without importing mautrix here.
    """

    user_id: str
    usage: list[str]
    keys: dict[str, str]
    signatures: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def key_id(self) -> str:
        return next(iter(self.keys))

    def serialize(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "usage": list(self.usage),
            "keys": dict(self.keys),
            "signatures": {user: dict(sigs) for user, sigs in self.signatures.items()},
        }


def _canonical_json(data: Any) -> str:
    """Matrix canonical JSON — what every signature in Matrix is computed over."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class CrossSigningSigner:
    """Publishes the cross-signing signatures a SAS exchange earns."""

    def __init__(self, adapter: Any, client: Any, olm: Any) -> None:
        self._adapter = adapter
        self._client = client
        self._olm = olm

    # ------------------------------------------------------------------
    # Identity helpers
    # ------------------------------------------------------------------

    @property
    def _user_id(self) -> str:
        return str(getattr(self._client, "mxid", "") or "")

    def _device_id(self) -> str:
        return str(getattr(self._client, "device_id", "") or "")

    def _private_keys(self) -> Any:
        """Our cross-signing private keys, or None when we have no identity.

        mautrix exposes the public halves but only keeps the private halves on
        the machine after a recovery key was imported (or a new identity was
        generated), so the store alone is not enough to sign anything.
        """
        return getattr(self._olm, "_cross_signing_private_keys", None)

    def has_identity(self) -> bool:
        return self._private_keys() is not None

    # ------------------------------------------------------------------
    # self-signing: our device -> our identity
    # ------------------------------------------------------------------

    async def sign_own_device(self) -> bool:
        """Sign our own device with the self-signing key (idempotent).

        Returns True when a signature was published.  Never raises: a missing
        identity or a homeserver hiccup must not keep the adapter from
        connecting, it only leaves the device unverified as before.
        """
        identity = getattr(self._olm, "own_identity", None)
        if identity is None:
            logger.debug("Matrix: no own device identity to cross-sign yet")
            return False
        if not self.has_identity():
            logger.info(
                "Matrix: own device is not cross-signed — no cross-signing private keys "
                "(set MATRIX_RECOVERY_KEY so the bot can sign its own devices)"
            )
            return False
        try:
            await self._olm.sign_own_device(identity)
        except Exception as exc:
            logger.warning("Matrix: signing our own device failed: %s", exc)
            return False
        logger.info(
            "Matrix: our device %s is signed by our self-signing key", self._device_id()
        )
        return True

    # ------------------------------------------------------------------
    # user-signing: the other user's master key -> our identity
    # ------------------------------------------------------------------

    async def trust_user(self, user_id: str, verified_key_ids: Iterable[str] = ()) -> bool:
        """Sign *user_id*'s master key with our user-signing key.

        ``verified_key_ids`` are the key ids the SAS exchange actually verified
        (``ed25519:<pubkey>``); the master key is only signed when it is one of
        them, so verifying a single session never silently upgrades into
        verifying that person's whole identity.
        """
        user_id = str(user_id or "")
        if not user_id or user_id == self._user_id:
            return False
        if not self.has_identity():
            logger.info(
                "Matrix: not cross-signing %s — no cross-signing private keys on our side",
                user_id,
            )
            return False
        try:
            public = await self._olm.get_cross_signing_public_keys(user_id)
        except Exception as exc:
            logger.warning("Matrix: cross-signing key lookup for %s failed: %s", user_id, exc)
            return False
        master_key = str(getattr(public, "master_key", "") or "")
        if not master_key:
            logger.info("Matrix: %s has no published master key to cross-sign", user_id)
            return False

        key_id = f"ed25519:{master_key}"
        verified = {str(k) for k in verified_key_ids or () if str(k)}
        if key_id not in verified:
            logger.info(
                "Matrix: not signing %s's master key — the SAS exchange verified %s, "
                "which does not include the identity key",
                user_id,
                ", ".join(sorted(verified)) or "no keys",
            )
            return False

        user_signing_key = self._private_keys().user_signing_key
        target = _SignedMasterKey(
            user_id=user_id,
            usage=[_USAGE_MASTER],
            keys={key_id: master_key},
        )
        try:
            # Sign exactly the canonical form of {user_id, usage, keys} — the
            # homeserver verifies the signature over the object minus
            # signatures/unsigned, and that is what this payload is.
            signature = str(
                user_signing_key.sign(
                    _canonical_json(
                        {
                            "user_id": target.user_id,
                            "usage": list(target.usage),
                            "keys": dict(target.keys),
                        }
                    )
                )
            )
            target.signatures = {
                self._user_id: {f"ed25519:{user_signing_key.public_key}": signature}
            }
            await self._client.upload_one_signature(user_id, key_id, target)
        except Exception as exc:
            logger.warning("Matrix: uploading our signature for %s failed: %s", user_id, exc)
            return False

        # Keep the local view consistent with what the homeserver now knows, so
        # trust questions answer from the crypto store instead of the network.
        try:
            await self._olm.crypto_store.put_signature(
                _CrossSigner(user_id, master_key),
                _CrossSigner(self._user_id, str(user_signing_key.public_key)),
                signature,
            )
        except Exception as exc:
            logger.debug("Matrix: could not store our signature for %s locally: %s", user_id, exc)

        logger.info("Matrix: cross-signed %s's identity (%s)", user_id, key_id)
        return True
