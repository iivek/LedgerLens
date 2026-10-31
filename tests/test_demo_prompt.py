"""scripts/demo.sh embeds PROMPT as a shell string; it must stay byte-identical to the
schema contract, same as the Android/React Native copies."""
import re
from pathlib import Path

from ledgerlens.schema import PROMPT

DEMO = Path(__file__).resolve().parents[1] / "scripts" / "demo.sh"


def test_demo_sh_prompt_matches_schema_prompt():
    text = DEMO.read_text()
    m = re.search(r"^PROMPT='(.*?)'$", text, re.S | re.M)
    assert m, "PROMPT='...' single-quoted assignment not found in scripts/demo.sh"
    assert m.group(1) == PROMPT


def test_demo_sh_is_executable():
    assert DEMO.stat().st_mode & 0o111, "scripts/demo.sh must be chmod +x"
