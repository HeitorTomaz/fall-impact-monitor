"""Masking of credentials embedded in URLs."""

import re

_CREDENTIALS = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)[^/\s@]+@")


def mask_url(text: str) -> str:
    """Replace ``user[:password]@`` in every URL inside ``text`` with ``***@``."""
    return _CREDENTIALS.sub(r"\g<scheme>***@", text)
