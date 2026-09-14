"""Standalone package and cwd-independent data entrypoint regression checks."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_package_paths_do_not_import_torch_or_soraka():
    script = (
        f'import sys; sys.path.insert(0, {str(ROOT)!r}); '
        'from lulu.paths import PROJECT_ROOT, soraka_root; '
        f'assert str(PROJECT_ROOT) == {str(ROOT)!r}; '
        'assert "torch" not in sys.modules; '
        'assert not any(n.startswith("s2t_local_offline") for n in sys.modules)'
    )
    subprocess.run([sys.executable, '-S', '-c', script], check=True)


def test_prepare_cli_uses_caller_paths_and_standalone_output_default(tmp_path):
    import pyarrow.parquet as pq
    (tmp_path / 'questions.json').write_text(json.dumps([
        {'question': '1+1?', 'answer': '2'}, {'question': '2+2?', 'answer': '4'},
    ]))
    out = tmp_path / 'separate_outputs'
    environment = dict(os.environ, LULU_OUTPUT_ROOT=str(out))
    result = subprocess.run([
        sys.executable, str(ROOT / 'scripts/prepare_lulu_data.py'), '--dataset',
        'questions.json', '--dev-size', '1',
    ], cwd=tmp_path, env=environment, check=True, text=True, capture_output=True)
    manifest = json.loads(result.stdout)
    assert manifest['counts']['train_questions'] == 1
    assert manifest['counts']['dev_questions'] == 1
    assert Path(manifest['outputs']['train']['path']) == out / 'data/lulu_dapo/train.jsonl'
    assert pq.read_table(manifest['outputs']['dev_eval']['path']).num_rows == 1
    assert not (tmp_path / 'Soraka_rlrl').exists()
