"""Masking of credentials embedded in URLs."""

import re

# Userinfo runs from the scheme to the LAST "@" that is followed by a host
# (no "@", "/", "?", "#" or whitespace) and then a delimiter or the end of the
# text, so passwords containing "@", ":" or "/" are fully masked.
_CREDENTIALS = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)\S*@(?=[^@\s/?#]*(?:[/?#\s]|$))")


def mask_url(text: str) -> str:
    """Replace ``user[:password]@`` in every URL inside ``text`` with ``***@``."""
    return _CREDENTIALS.sub(r"\g<scheme>***@", text)
