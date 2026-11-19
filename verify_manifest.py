"""Verify tracked scientific companion files using only the Python standard library."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent

def main():
    manifest = json.loads((ROOT / 'REPOSITORY_MANIFEST.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path = (ROOT / item['path']).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError('Missing or invalid path: ' + item['path'])
        data = path.read_bytes()
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('File differs from manifest: ' + item['path'])
    print(f"Verified {len(manifest['files'])} files against the repository SHA256 manifest.")

if __name__ == '__main__':
    main()
