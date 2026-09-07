"""Privacy enforcement.

The project's central claim to the hospital is that RGB is processed in memory
and never persisted. That claim is worth nothing as a policy, so it is enforced
three ways:

1. Structurally -- `ahfd.types` has no image field, so RGB cannot flow onward.
2. By test -- `tests/test_privacy.py` walks the AST of every module and fails
   the build if an image-writing call appears outside `ahfd.debug`.
3. At runtime -- this module. Raw capture requires three independent switches
   set by three different people/places to agree. Any one of them missing means
   no raw capture.

The triple gate exists because a single flag gets left on. A config value, an
environment variable and an explicit CLI acknowledgement will not all be
enabled by accident.
"""

from __future__ import annotations

import os

ENV_VAR = "AHFD_ALLOW_RAW"

BANNER = "RAW RECORDING -- RGB IS BEING WRITTEN TO DISK"


class PrivacyViolation(RuntimeError):
    """Raised when something tries to persist imagery without full consent."""


def raw_capture_allowed(*, config_flag: bool, cli_flag: bool) -> bool:
    """True only if all three independent switches agree.

    `config_flag`   privacy.allow_raw_capture in the YAML config
    `cli_flag`      the explicit --i-understand-raw-capture command line flag
    environment     AHFD_ALLOW_RAW=1
    """
    env_flag = os.environ.get(ENV_VAR) == "1"
    return bool(config_flag) and bool(cli_flag) and env_flag


def require_raw_capture(*, config_flag: bool, cli_flag: bool) -> None:
    """Raise unless raw capture is fully authorised. Names what is missing."""
    if raw_capture_allowed(config_flag=config_flag, cli_flag=cli_flag):
        return

    missing = []
    if not config_flag:
        missing.append("config privacy.allow_raw_capture: true")
    if not cli_flag:
        missing.append("--i-understand-raw-capture")
    if os.environ.get(ENV_VAR) != "1":
        missing.append(ENV_VAR + "=1")

    raise PrivacyViolation(
        "Raw frame capture is not authorised. Missing: " + "; ".join(missing)
    )
