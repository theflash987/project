"""Check plain-config training and optional CLI overrides without starting DDP."""

import sys
from unittest.mock import patch

from basicsr.utils import options


config_path = sys.argv[1]
source = options.yaml_load(config_path)
for overrides in ([], ['train:total_iter=30', 'name=cli_regression']):
    with patch.object(sys, 'argv', ['train.py', '-opt', config_path] +
                      (['--force_yml'] + overrides if overrides else [])), \
            patch.object(options, 'init_dist'), \
            patch.object(options, 'get_dist_info', return_value=(0, 4)), \
            patch.object(options, 'set_random_seed'):
        parsed, args = options.parse_options('.')
    assert args.force_yml == overrides
    assert parsed['train']['total_iter'] == (30 if overrides else 20000)
    assert parsed['name'] == ('cli_regression' if overrides else source['name'])
    assert parsed['network_g'] == source['network_g']
print('ADDBACK_CLI_PLAIN_AND_OVERRIDE_OK', flush=True)
