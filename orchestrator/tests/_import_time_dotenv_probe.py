"""The test that test_suite_loads_no_dotenv.py runs in a nested pytest session; not collected by the suite (no ``test_`` prefix)."""
import json
import os
from pathlib import Path

import dotenv
from shared.secret_env import is_secret_env_name

dotenv.load_dotenv(Path.cwd() / '.env')
SECRET_NAMES_AFTER_IMPORT = sorted(name for name in os.environ if is_secret_env_name(name))


def test_records_the_secret_names_present_once_it_was_imported(tmp_path: Path) -> None:
    (tmp_path / 'secret-names.json').write_text(json.dumps(SECRET_NAMES_AFTER_IMPORT))
