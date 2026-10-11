#!/usr/bin/env python3
"""List or download GRO.data assets using current metadata and checksum verification.

Listing is the default. No PyTorch, torchvision or GPU dependencies are required.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

HOST = 'https://data.goettingen-research-online.de'


def metadata(dataset):
    query = urllib.parse.urlencode({'persistentId': f'doi:10.25625/{dataset}'})
    with urllib.request.urlopen(f'{HOST}/api/datasets/:persistentId/?{query}', timeout=60) as response:
        version = json.load(response)['data']['latestVersion']
    return version, {item['dataFile']['filename']: item['dataFile'] for item in version['files']}


def matches(path, info):
    if not path.is_file() or path.stat().st_size != info['filesize']:
        return False
    checksum = info['checksum']
    digest = hashlib.new(checksum['type'].lower().replace('-', ''))
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest().lower() == checksum['value'].lower()


def download(info, root):
    name = info['filename']
    if Path(name).name != name:
        raise ValueError(f'Unsafe filename: {name}')
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    if matches(target, info):
        print(f'Already verified: {target}')
        return
    partial = target.with_name(target.name + '.part')
    url = f"{HOST}/api/access/datafile/{info['id']}"
    print(f'Downloading {name} ({info["filesize"] / 1e6:.1f} MB)', flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as handle:
        for chunk in iter(lambda: response.read(1024 * 1024), b''):
            handle.write(chunk)
    if not matches(partial, info):
        raise ValueError(f'Size/checksum mismatch: {partial}; no final file was installed')
    partial.replace(target)
    print(f'Verified: {target}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['VPMPID', 'QUTUWU'], required=True)
    parser.add_argument('--files', nargs='+', help='Exact filenames to download; omitted means list only')
    parser.add_argument('--root', type=Path, default=Path('data/downloads'))
    args = parser.parse_args()
    version, files = metadata(args.dataset)
    if not args.files:
        for name, info in sorted(files.items()):
            print(f'{name}\t{info["filesize"] / 1e6:.1f} MB\t{info["id"]}')
        return
    unknown = set(args.files) - files.keys()
    if unknown:
        parser.error(f'Unknown filenames: {sorted(unknown)}; list the dataset first')
    selected = [files[name] for name in dict.fromkeys(args.files)]
    for info in selected:
        download(info, args.root)
    # One receipt per selection; different downloads do not overwrite each other's provenance.
    selection_id = hashlib.sha256('\n'.join(sorted(args.files)).encode()).hexdigest()[:12]
    receipt = {
        'dataset': f'doi:10.25625/{args.dataset}',
        'version': [version.get('versionNumber'), version.get('versionMinorNumber')],
        'files': selected,
    }
    (args.root / f'download_receipt_{args.dataset}_{selection_id}.json').write_text(
        json.dumps(receipt, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
