"""
Command line entry point. Currently just copies the worked examples somewhere
editable, since the installed copies live inside site-packages.
"""
import argparse
import pathlib
import shutil
import sys

from .lib.openpopcon_util import example_dir, list_examples


def _copy_examples(dest: pathlib.Path, force: bool, which=None) -> int:
    names = which or list_examples()
    unknown = [n for n in names if n not in list_examples()]
    if unknown:
        print(f"No example named {', '.join(unknown)}. "
              f"Available: {', '.join(list_examples())}.", file=sys.stderr)
        return 1

    existing = [n for n in names if dest.joinpath(n).exists()]
    if existing and not force:
        print(f"{dest}{pathlib.os.sep}{{{','.join(existing)}}} already exists. "
              f"Use --force to overwrite.", file=sys.stderr)
        return 1

    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = dest.joinpath(name)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(example_dir(name), target)
        print(f"  {target}")
    print(f"\nCopied {len(names)} example(s). To run one:\n"
          f"  cd {dest.joinpath(names[0])}\n"
          f"  python -c \"import openpopcon as op; "
          f"pc = op.POPCON(settingsfile='POPCON_input_example.yaml', "
          f"plotsettingsfile='plotsettings.yml'); pc.run_POPCON(); pc.plot()\"")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog='openpopcon', description='0-D tokamak scoping tool.')
    sub = parser.add_subparsers(dest='command', required=True)

    ex = sub.add_parser('examples', help='copy the worked examples somewhere editable')
    ex.add_argument('dest', nargs='?', default='./openpopcon_examples',
                    help='where to copy them (default: ./openpopcon_examples)')
    ex.add_argument('--force', action='store_true', help='overwrite existing copies')
    ex.add_argument('--only', nargs='+', metavar='NAME',
                    help=f'copy only these (of: {", ".join(list_examples())})')

    args = parser.parse_args(argv)
    if args.command == 'examples':
        return _copy_examples(pathlib.Path(args.dest), args.force, args.only)
    return 1


if __name__ == '__main__':
    sys.exit(main())
