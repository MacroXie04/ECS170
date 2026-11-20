"""Predefined comparisons; test performance is never used for selection."""

def configurations(dataset):
    base = dict(channels=[16, 32], kernel=3, pool='max', grid=[4, 4], hidden=128,
                dropout=0.2, batch_norm=False, loss='cross_entropy',
                lr=0.001, weight_decay=0.0001, batch_size=128,
                epochs=12, patience=5, seed=42, augment=False)
    if dataset == 'MNIST':
        base.update(grid=[7, 7])
    elif dataset == 'ORL':
        base.update(grid=[7, 5], epochs=80, patience=15, batch_size=40, dropout=0.3)
    elif dataset == 'CIFAR':
        base.update(batch_norm=True, epochs=24, patience=6, augment=True)
    else:
        raise ValueError('Unknown dataset')
    return {'baseline': dict(base),
            'depth3': dict(base, channels=[16, 32, 64]),
            'average_pool': dict(base, pool='average'),
            'hidden32': dict(base, hidden=32),
            'squared_error': dict(base, loss='squared_error')}
