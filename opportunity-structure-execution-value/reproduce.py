"""Recompute the archived evidence in a temporary copy without editing source artifacts."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', action='store_true', help='Only seven standard-library certificates')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    plan = json.loads((root / 'reproduction_plan.json').read_text(encoding='utf-8'))
    manifest = root / 'file_manifest.json'
    if manifest.exists():
        for row in json.loads(manifest.read_text(encoding='utf-8')):
            path = root / row['path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise RuntimeError('File changed since packaging: ' + row['path'])
    names = plan['core_scripts']
    if not args.core:
        names = names + plan['numeric_scripts'] + plan['figure_scripts']
    output = (root / 'reproduction_output').resolve()
    if not output.is_relative_to(root.resolve()):
        raise ValueError('Output directory escapes repository')
    output.mkdir(exist_ok=True)
    environment = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    versions = {'python': sys.version.split()[0]}
    if not args.core:
        import numpy, scipy, matplotlib, reportlab
        versions.update(numpy=numpy.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__, reportlab=reportlab.Version)
    rows = []
    report = {'versions': versions, 'core_only': args.core, 'archived_outputs_overwritten': False, 'checks': rows}
    with tempfile.TemporaryDirectory(prefix='temporary_', dir=output) as temp:
        local = Path(temp) / 'repository'
        shutil.copytree(root, local, ignore=shutil.ignore_patterns('reproduction_output', '__pycache__', '.git', '.venv'))
        for name in names:
            process = subprocess.run([sys.executable, str(local / 'experiments' / (name + '.py'))],
                                     cwd=local, env=environment, capture_output=True, text=True, encoding='utf-8', timeout=900)
            (output / (name + '.log')).write_text(process.stdout + '\n' + process.stderr, encoding='utf-8')
            passed = process.returncode == 0
            saved = root / 'experiments' / (name + '.json')
            generated = local / 'experiments' / (name + '.json')
            exact = passed and generated.is_file() and json.loads(saved.read_text(encoding='utf-8')) == json.loads(generated.read_text(encoding='utf-8'))
            rows.append({'script': name, 'exit_code': process.returncode, 'parsed_json_exact_match': exact})
            (output / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            print(name + ': ' + ('PASS' if exact else 'FAIL'), flush=True)
            if not exact:
                raise RuntimeError('Reproduction failed or differs from archive: ' + name + '. See reproduction_output logs.')
        if not args.core:
            for filename in ['revision_structure_boundary.svg', 'revision_legal_world_comparison.svg',
                             'revision_cash_and_payment_ledger.svg', 'fourth_revision_generalization.svg']:
                assert (local / 'paper' / 'figures' / filename).stat().st_size > 1000
            report['figure_outputs_present'] = True
    report['status'] = 'passed'
    (output / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Passed', len(rows), 'archived evidence comparisons. Source artifacts unchanged.')

if __name__ == '__main__':
    main()
