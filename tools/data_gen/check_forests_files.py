#!/usr/bin/env python
"""Quick check: which files in data/train/forests can be read by laspy (report corrupted/incomplete LAZ)."""
import os
import os.path as osp
import sys

def main():
    base = sys.argv[1] if len(sys.argv) > 1 else osp.join(osp.dirname(__file__), '..', '..', 'data', 'train', 'forests')
    if not osp.isdir(base):
        print('Directory not found:', base)
        return
    try:
        import laspy
    except ImportError:
        print('laspy not installed')
        return
    try:
        import lazrs
        exc_type = lazrs.LazrsError
    except ImportError:
        exc_type = OSError

    ok, bad = [], []
    for name in sorted(os.listdir(base)):
        if not (name.endswith('.las') or name.endswith('.laz')):
            continue
        path = osp.join(base, name)
        try:
            laspy.read(path)
            ok.append(name)
        except exc_type as e:
            bad.append((name, str(e)))
        except Exception as e:
            bad.append((name, str(e)))

    print('Readable:', len(ok), '->', ok[:5], '...' if len(ok) > 5 else '')
    if bad:
        print('Unreadable (corrupted/incomplete):', len(bad))
        for name, err in bad:
            print('  -', name, ':', err)
    else:
        print('All checked files are readable.')

if __name__ == '__main__':
    main()
