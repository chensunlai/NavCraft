#!/usr/bin/env python3
"""Refresh file hashes only; never change version, status, counts, or review results."""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def refresh(root, write=False):
    root = root.resolve()
    path = root/'raw/manifest.json'
    old = path.read_bytes()
    manifest = json.loads(old)
    base = (path.parent/manifest.get('checksum_path_base','..')).resolve()
    if base != root:
        raise ValueError('expected checksum_path_base to resolve to scene root')
    excluded = {'raw/manifest.json','raw/validation.json','raw/standard_archive_validation.json'}
    hashes = {}
    for file in sorted(root.rglob('*')):
        name = file.relative_to(root).as_posix()
        if name in excluded or not file.is_file():
            continue
        if not file.resolve().is_relative_to(root):
            raise ValueError('external symlink is not a self-contained archive: ' + name)
        hashes[name] = hashlib.sha256(file.read_bytes()).hexdigest()
    previous = manifest.get('checksums',{})
    changes = {'added': sorted(set(hashes)-set(previous))}
    changes['removed'] = sorted(set(previous)-set(hashes))
    changes['changed'] = sorted(k for k in hashes.keys() & previous.keys() if hashes[k] != previous[k])
    if write:
        manifest['checksum_path_base'] = '..'
        manifest['checksums'] = hashes
        # Only replace when bytes change. Scene version backups belong to the
        # caller; this helper changes no geometry or acceptance decisions.
        encoded = (json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
        if encoded != old:
            with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.manifest-',suffix='.tmp',delete=False) as f:
                temporary = Path(f.name)
                f.write(encoded)
            try:
                if path.read_bytes() != old:
                    raise RuntimeError('manifest changed during update; retry after reviewing concurrent writes')
                os.replace(temporary,path)
            finally:
                temporary.unlink(missing_ok=True)
    return dict(written=write,changes=changes,hashed_files=len(hashes),review_status_unchanged=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('scene_directory',type=Path)
    p.add_argument('--write',action='store_true',help='Actually update manifest; default only reports hash changes')
    args = p.parse_args()
    print(json.dumps(refresh(args.scene_directory,args.write),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
