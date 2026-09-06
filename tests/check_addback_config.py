"""Require 6308179's configuration and unchanged non-fusion production code."""

import os
import sys
from copy import deepcopy
from pathlib import Path

from basicsr.utils.options import yaml_load


current = yaml_load(sys.argv[1])
parent_code = Path(os.environ['BASELINE_CODE'])
baseline = yaml_load(str(parent_code / 'options/isambard/train_addback_20k.yml'))
comparison = deepcopy(current)
comparison['name'] = baseline['name']
comparison['path'] = baseline['path']
assert comparison == baseline, 'Non-ablation configuration changed.'
assert current['train']['total_iter'] == 20000
assert current['train']['lambda_base_rec'] == 0.0
assert current['train']['lambda_base_ssim'] == 0.0
assert current['manual_seed'] == 0
changed = {
    'basicsr/archs/addback_components.py',
    'basicsr/archs/modal_content_pole_wavelet_dwt3_arch.py',
    'basicsr/models/modal_content_pole_wavelet_dwt3_video_model.py',
}
code = Path(__file__).resolve().parents[1]
for source in (parent_code / 'basicsr').rglob('*.py'):
    relative = source.relative_to(parent_code)
    if str(relative) not in changed:
        assert source.read_bytes() == (code / relative).read_bytes(), relative
print('MATCHED_PARENT_6308179_CONFIG_AND_UNCHANGED_CORE_CODE', flush=True)
