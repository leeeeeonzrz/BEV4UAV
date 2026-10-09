#!/usr/bin/env python3
"""Validate user-supplied initialization files before BEV4UAV training."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='Path to a BEV4UAV config')
    parser.add_argument('--data-root', help='User dataset root')
    parser.add_argument('--lidar-checkpoint', help='Stage-one LiDAR checkpoint')
    parser.add_argument('--image-checkpoint', help='Swin Tiny initialization')
    parser.add_argument('--work-dir', help='Training output directory')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--resume', metavar='CHECKPOINT',
                        help='Resume from an explicit existing checkpoint')
    parser.add_argument('--launcher', default='none',
                        choices=['none', 'pytorch', 'slurm', 'mpi'])
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    parser.add_argument('--cfg-options', nargs='+', default=[], metavar='KEY=VALUE',
                        help='MMEngine config overrides, applied before validation')
    parser.add_argument('--check-only', action='store_true',
                        help='Parse config and check paths without starting training')
    return parser.parse_args(argv)


def require_checkpoint(value, description):
    """Check file metadata only; do not deserialize a checkpoint."""
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise ValueError('{} is required for the frozen fusion stage.'.format(
            description))
    path = Path(value)
    if not path.is_file():
        raise ValueError('{} does not exist: {}'.format(description, path))
    if path.stat().st_size == 0:
        raise ValueError('{} is empty: {}'.format(description, path))


def validate_initialization(cfg, resume=None):
    model = cfg.get('model', {})
    if resume is not None:
        require_checkpoint(resume, 'Resume checkpoint')
    if model.get('img_backbone'):
        # The native training entry replaces load_from with --resume.
        if resume is None:
            require_checkpoint(cfg.get('load_from'), 'Stage-one LiDAR checkpoint')
        init_cfg = model['img_backbone'].get('init_cfg') or {}
        if not isinstance(init_cfg, dict):
            raise ValueError('The image backbone needs a Pretrained init_cfg.')
        require_checkpoint(init_cfg.get('checkpoint'),
                           'Swin Tiny initialization checkpoint')
    elif resume is None and cfg.get('load_from'):
        require_checkpoint(cfg['load_from'], 'Initialization checkpoint')


def main(argv=None):
    args = parse_args(argv)
    # Keep --help available before the training environment is installed.
    try:
        from mmengine.config import Config, DictAction
    except ImportError as exc:
        raise SystemExit('Install the dependencies described in README.md first.') from exc

    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise SystemExit('Config does not exist: {}'.format(config_path))

    # Environment substitution updates every inherited dataloader/evaluator.
    if args.data_root is not None:
        os.environ['BEV4UAV_DATA_ROOT'] = str(Path(args.data_root).expanduser())
    if args.lidar_checkpoint is not None:
        os.environ['BEV4UAV_LIDAR_CHECKPOINT'] = str(
            Path(args.lidar_checkpoint).expanduser())
    if args.image_checkpoint is not None:
        os.environ['BEV4UAV_IMAGE_CHECKPOINT'] = str(
            Path(args.image_checkpoint).expanduser())

    cfg = Config.fromfile(str(config_path), import_custom_modules=False)
    if args.cfg_options:
        override_parser = argparse.ArgumentParser(add_help=False)
        override_parser.add_argument('--cfg-options', nargs='+', action=DictAction)
        overrides = override_parser.parse_args(
            ['--cfg-options'] + args.cfg_options).cfg_options
        cfg.merge_from_dict(overrides)
    try:
        validate_initialization(cfg, args.resume)
    except ValueError as exc:
        raise SystemExit('{} No training was started.'.format(exc)) from exc

    if args.check_only:
        print('Configuration and required checkpoint paths passed validation.')
        print('Checkpoint contents and dataset files were not read.')
        return 0

    command = [sys.executable, str(Path(__file__).with_name('train.py')),
               str(config_path), '--launcher', args.launcher,
               '--local-rank', str(args.local_rank)]
    if args.work_dir:
        command.extend(['--work-dir', args.work_dir])
    if args.amp:
        command.append('--amp')
    if args.resume:
        command.extend(['--resume', args.resume])
    if args.cfg_options:
        command.extend(['--cfg-options'] + args.cfg_options)
    return subprocess.call(command)


if __name__ == '__main__':
    sys.exit(main())
