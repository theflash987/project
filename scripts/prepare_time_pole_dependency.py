"""Resolve the completed predecessor's final numbered model/state pair."""

import sys
from pathlib import Path

import yaml


def prepare_config(base_config, predecessor, name, destination):
    predecessor = Path(predecessor)
    state = max((predecessor / 'training_states').glob('*.state'),
                key=lambda path: int(path.stem))
    iteration = int(state.stem)
    model = predecessor / 'models' / f'net_g_{iteration}.pth'
    assert model.is_file(), model
    with open(base_config, encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    assert iteration < config['train']['total_iter']
    config['name'] = name
    config['resume'].update(
        model=str(model), state=str(state), expected_iter=iteration)
    with open(destination, 'w', encoding='utf-8') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    print(f'DEPENDENCY_RESUME_SOURCE iter={iteration} model={model} state={state}',
          flush=True)


if __name__ == '__main__':
    prepare_config(*sys.argv[1:])
