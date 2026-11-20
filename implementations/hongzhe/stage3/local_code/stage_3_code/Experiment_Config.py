"""Stage 3 experiment registry: every formal configuration is declared here.

Each variant changes one main factor relative to its dataset baseline, so a
difference from the baseline can be attributed to that factor. The registry and
the two selection rules below are locked into protocol.json before any test
evaluation; see Experiment_Runner.lock_protocol.
"""
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json

# One split per dataset, shared by every configuration and training seed.
SPLIT_SEED = 170
FORMAL_SEEDS = (2026, 2027, 2028)
# CIFAR first: it is the longest job, so it should start as early as possible.
DATASET_ORDER = ('cifar', 'orl', 'mnist')
REFERENCE_ACCURACY = {'mnist': 0.95, 'orl': 0.95, 'cifar': 0.70}

CHECKPOINT_RULE = (
    'Within a run, keep the epoch with the highest validation accuracy; break '
    'ties by the lowest validation cross-entropy, then by the earliest epoch. '
    'The test set is never read during training.'
)
BEST_CONFIG_RULE = (
    'Within a dataset, pick the configuration with the highest mean selected-'
    'checkpoint validation accuracy over seeds; break ties by the lowest mean '
    'validation cross-entropy, then by the fewest parameters. The final model '
    'is the seed of that configuration ranked first by the checkpoint rule. '
    'Test metrics play no part in either choice.'
)


@dataclass(frozen=True)
class CNNConfig:
    dataset: str
    name: str
    changed_factor: str
    description: str
    channels: tuple
    hidden_dims: tuple
    convs_per_stage: int = 1
    kernel_size: int = 3
    padding: int = 1
    downsample: str = 'maxpool'  # maxpool | avgpool | stride
    batch_norm: bool = True
    dropout: float = 0.0
    loss: str = 'cross_entropy'  # cross_entropy | mse_softmax
    optimizer: str = 'adam'  # adam | sgd
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.0
    nesterov: bool = False
    scheduler: str = 'cosine'  # cosine | constant
    max_epoch: int = 15
    batch_size: int = 128
    eval_batch_size: int = 1024
    aug_shift: int = 0
    aug_hflip: bool = False
    aug_padding_mode: str = 'constant'  # constant | replicate
    weight_init: str = 'kaiming_normal_fan_out_relu'

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        known = {f.name for f in fields(cls)}
        values = {key: value for key, value in values.items() if key in known}
        values['channels'] = tuple(values['channels'])
        values['hidden_dims'] = tuple(values['hidden_dims'])
        return cls(**values)

    def fingerprint(self):
        text = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()


def _panel(baseline, depth_name, depth_description, depth_changes, hidden_dims):
    """The same factors on every dataset, so the ablations read across datasets."""
    variants = [
        baseline,
        replace(baseline, name=depth_name, changed_factor='depth',
                description=depth_description, **depth_changes),
        replace(baseline, name='kernel5', changed_factor='kernel_size',
                description='5x5 kernels; padding 2 keeps the same spatial size',
                kernel_size=5, padding=2),
        replace(baseline, name='avgpool', changed_factor='pooling',
                description='Average pooling instead of max pooling',
                downsample='avgpool'),
        replace(baseline, name='stride2', changed_factor='stride',
                description='Stride-2 convolution replaces each pooling layer',
                downsample='stride'),
        replace(baseline, name='hidden_small', changed_factor='hidden_dim',
                description=f'Hidden layer width {hidden_dims[0]} instead of '
                            f'{baseline.hidden_dims[0]}',
                hidden_dims=hidden_dims),
        replace(baseline, name='mse_loss', changed_factor='loss',
                description='Squared error between softmax output and one-hot '
                            'target instead of cross-entropy',
                loss='mse_softmax'),
        replace(baseline, name='no_dropout', changed_factor='dropout',
                description='Dropout removed', dropout=0.0),
    ]
    return {config.name: config for config in variants}


MNIST_BASELINE = CNNConfig(
    dataset='mnist', name='baseline', changed_factor='none',
    description='Two 3x3 conv stages (32, 64), max pooling, one 128-unit hidden layer',
    channels=(32, 64), hidden_dims=(128,), dropout=0.25,
    optimizer='adam', learning_rate=1e-3, max_epoch=15, batch_size=128,
    aug_shift=2,
)

ORL_BASELINE = CNNConfig(
    dataset='orl', name='baseline', changed_factor='none',
    description='Four 3x3 conv stages (16-128), max pooling, one 256-unit hidden layer',
    channels=(16, 32, 64, 128), hidden_dims=(256,), dropout=0.5,
    optimizer='adam', learning_rate=1e-3, weight_decay=1e-4,
    max_epoch=120, batch_size=32, aug_shift=4, aug_hflip=True,
    aug_padding_mode='replicate',
)

CIFAR_BASELINE = CNNConfig(
    dataset='cifar', name='baseline', changed_factor='none',
    description='Three stages of two 3x3 convs (32, 64, 128), max pooling, '
                'one 256-unit hidden layer',
    channels=(32, 64, 128), convs_per_stage=2, hidden_dims=(256,), dropout=0.3,
    optimizer='sgd', learning_rate=0.05, momentum=0.9, nesterov=True,
    weight_decay=5e-4, max_epoch=40, batch_size=128, aug_shift=4, aug_hflip=True,
)

REGISTRY = {
    'mnist': _panel(MNIST_BASELINE, 'depth_3stage',
                    'A third conv stage (32, 64, 128)',
                    {'channels': (32, 64, 128)}, (32,)),
    'orl': _panel(ORL_BASELINE, 'depth_3stage',
                  'Three conv stages (16, 32, 64) instead of four',
                  {'channels': (16, 32, 64)}, (64,)),
    'cifar': _panel(CIFAR_BASELINE, 'depth_1conv',
                    'One conv per stage instead of two (3 conv layers, not 6)',
                    {'convs_per_stage': 1}, (64,)),
}


def get_config(dataset, name):
    try:
        return REGISTRY[dataset][name]
    except KeyError:
        known = ', '.join(REGISTRY.get(dataset, {})) or 'none'
        raise ValueError(f'Unknown config {dataset}/{name}; known: {known}') from None


def apply_overrides(config, assignments):
    """Apply KEY=VALUE strings; values are JSON, so lists and booleans work."""
    changes = {}
    valid = {f.name for f in fields(config)}
    for assignment in assignments:
        key, separator, raw = assignment.partition('=')
        if not separator or key not in valid:
            raise ValueError(f'Bad override {assignment!r}; expected FIELD=VALUE')
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        changes[key] = tuple(value) if isinstance(value, list) else value
    return replace(config, **changes)
