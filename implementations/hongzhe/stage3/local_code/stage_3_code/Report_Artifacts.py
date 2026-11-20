"""Tables, learning curves, and architecture descriptions built from saved runs.

Everything here reads files that runs already wrote, so summaries can be rebuilt
at any time without retraining. Choosing the best configuration reads validation
columns only.
"""
import csv
from pathlib import Path
import statistics

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch

from local_code.stage_3_code.Experiment_Config import (BEST_CONFIG_RULE, CHECKPOINT_RULE,
                                                       DATASET_ORDER, FORMAL_SEEDS,
                                                       REFERENCE_ACCURACY, REGISTRY)
from local_code.stage_3_code.Runtime import atomic_write, read_json, write_json, write_text

METRICS = ('accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
           'precision_weighted', 'recall_weighted', 'f1_weighted')
DATASET_TITLES = {'mnist': 'MNIST', 'orl': 'ORL', 'cifar': 'CIFAR-10'}

# Fixed slot per configuration, in registry order, so a colour always means the
# same variant in every figure. The eight-slot order passes adjacent-pair
# colour-vision checks; see the dataviz reference palette.
SERIES_COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                 '#e87ba4', '#008300', '#6250d6', '#e34948')
SURFACE, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
FINISHED_STATES = ('trained', 'evaluated')


def describe_model(method):
    """Per-layer output shapes and parameter counts from one dry forward pass."""
    rows, hooks = [], []

    def recorder(name):
        def record(module, _inputs, output):
            rows.append({'layer': name, 'type': type(module).__name__,
                         'settings': module.extra_repr(),
                         'output_shape': tuple(output.shape[1:]),
                         'parameters': sum(p.numel() for p in module.parameters(recurse=False))})
        return record

    for name, module in method.named_modules():
        if not list(module.children()):
            hooks.append(module.register_forward_hook(recorder(name)))
    was_training = method.training
    method.eval()
    with torch.no_grad():
        method(torch.zeros(1, *method.input_shape, device=method.device))
    method.train(was_training)
    for hook in hooks:
        hook.remove()
    return rows


def model_summary_text(method):
    rows = describe_model(method)
    config = method.config
    lines = [f'{config.dataset}/{config.name}: {config.description}',
             f'Input (channels, height, width): {method.input_shape}',
             f'Classes: {method.num_classes}',
             f'Trainable parameters: {method.parameter_count():,}', '',
             f'{"Layer":<16}{"Type":<14}{"Output shape":<20}{"Params":>10}  Settings']
    for row in rows:
        shape = 'x'.join(str(size) for size in row['output_shape'])
        lines.append(f'{row["layer"]:<16}{row["type"]:<14}{shape:<20}'
                     f'{row["parameters"]:>10,}  {row["settings"]}')
    return '\n'.join(lines) + '\n'


def _architecture_blocks(method):
    """Fold Conv+BatchNorm+ReLU and Linear+ReLU+Dropout into one drawn block."""
    channels, height, width = method.input_shape
    blocks = [('Input', f'{channels}x{height}x{width}')]
    for row in describe_model(method):
        shape = 'x'.join(str(size) for size in row['output_shape'])
        kind = row['type']
        if kind == 'Conv2d':
            kernel = method.config.kernel_size
            stride = ' s2' if 'stride=(2, 2)' in row['settings'] else ''
            blocks.append((f'Conv {kernel}x{kernel}{stride}\n+BN+ReLU'
                           if method.config.batch_norm else f'Conv {kernel}x{kernel}{stride}\n+ReLU',
                           shape))
        elif kind in ('MaxPool2d', 'AvgPool2d'):
            blocks.append((f'{kind[:3]}Pool 2x2', shape))
        elif kind == 'Flatten':
            blocks.append(('Flatten', shape))
        elif kind == 'Linear':
            blocks.append(('Linear', shape))
        elif kind == 'Dropout':
            label, previous = blocks[-1]
            blocks[-1] = (f'{label}\n+ReLU+Dropout', previous)
    blocks[-1] = ('Linear\n(logits)', blocks[-1][1])
    return blocks


def save_architecture_figure(method, path, per_row=7):
    blocks = _architecture_blocks(method)
    rows = -(-len(blocks) // per_row)
    box_width, box_height, gap_x, gap_y = 1.42, 0.82, 0.3, 0.72
    figure, axes = plt.subplots(figsize=(per_row * (box_width + gap_x) + 0.2,
                                         rows * (box_height + gap_y) + 0.45),
                                facecolor=SURFACE)
    axes.set_xlim(-0.1, per_row * (box_width + gap_x))
    axes.set_ylim(-rows * (box_height + gap_y) + gap_y - 0.3, box_height + 0.45)
    axes.axis('off')
    config = method.config
    axes.text(0, box_height + 0.2, f'{DATASET_TITLES[config.dataset]} {config.name}: '
              f'{method.parameter_count():,} parameters', fontsize=11, color=INK,
              fontweight='bold')
    positions = []
    for index, (label, shape) in enumerate(blocks):
        row, column = divmod(index, per_row)
        x = column * (box_width + gap_x)
        y = -row * (box_height + gap_y)
        positions.append((x, y))
        axes.add_patch(FancyBboxPatch((x, y), box_width, box_height, facecolor='#eef3fb',
                                      edgecolor='#2a78d6', linewidth=1,
                                      boxstyle='round,pad=0,rounding_size=0.06'))
        axes.text(x + box_width / 2, y + box_height / 2, label, ha='center', va='center',
                  fontsize=8.5, color=INK)
        axes.text(x + box_width / 2, y - 0.13, shape, ha='center', va='center',
                  fontsize=8, color=MUTED)
    for (x1, y1), (x2, y2) in zip(positions, positions[1:]):
        if y1 == y2:
            start, end = (x1 + box_width, y1 + box_height / 2), (x2, y2 + box_height / 2)
        else:
            # Down into the row gap, across it, then a short arrow into the next box.
            middle_x1, middle_x2 = x1 + box_width / 2, x2 + box_width / 2
            gap_y_position = y2 + box_height + 0.2
            axes.plot([middle_x1, middle_x1, middle_x2], [y1 - 0.24, gap_y_position, gap_y_position],
                      color=MUTED, linewidth=1, solid_capstyle='round')
            start, end = (middle_x2, gap_y_position), (middle_x2, y2 + box_height)
        axes.annotate('', xy=end, xytext=start,
                      arrowprops={'arrowstyle': '->', 'color': MUTED, 'lw': 1,
                                  'shrinkA': 0, 'shrinkB': 0})
    _save_figure(figure, path)


def _save_figure(figure, path):
    atomic_write(path, lambda target: figure.savefig(target, format='png', dpi=200,
                                                     bbox_inches='tight', pad_inches=0.12,
                                                     facecolor=figure.get_facecolor()))
    plt.close(figure)


def _distinct(sessions, key):
    return '; '.join(dict.fromkeys(str(session.get(key)) for session in sessions))


def collect_runs(root):
    """One record per formal run, including unfinished ones so they can be reported.

    Directories with seeds outside FORMAL_SEEDS are not part of the experiment.
    """
    records = []
    for config_path in sorted(Path(root).glob('runs/*/*/seed_*/config.json')):
        run_dir = config_path.parent
        meta = read_json(config_path)
        if meta['seed'] not in FORMAL_SEEDS:
            continue
        status_path = run_dir / 'status.json'
        status = read_json(status_path) if status_path.exists() else {}
        environment_path = run_dir / 'environment.json'
        sessions = read_json(environment_path)['sessions'] if environment_path.exists() else []
        config = meta['config']
        state = status.get('state', 'unknown')
        epochs = status.get('epoch')
        record = {
            'dataset': config['dataset'], 'config': config['name'],
            'changed_factor': config['changed_factor'], 'seed': meta['seed'],
            'state': state,
            'finished': state in FINISHED_STATES and epochs == config['max_epoch'],
            'parameters': meta['parameters'],
            'device': _distinct(sessions, 'device'),
            'device_name': _distinct(sessions, 'device_name'),
            'torch_threads': _distinct(sessions, 'torch_threads'),
            'concurrent_jobs': _distinct(sessions, 'concurrent_jobs'),
            'training_sessions': len(sessions), 'max_epoch': config['max_epoch'],
            'epochs_completed': epochs, 'selected_epoch': status.get('selected_epoch'),
            'val_accuracy': status.get('val_accuracy'),
            'val_cross_entropy': status.get('val_cross_entropy'),
            'train_seconds': status.get('train_seconds'),
            'run_dir': str(run_dir),
        }
        metrics_path = run_dir / 'test_metrics.json'
        if metrics_path.exists():
            metrics = read_json(metrics_path)
            record.update({f'test_{name}': metrics[name] for name in METRICS})
            record['test_cross_entropy'] = metrics['test_cross_entropy']
            record['test_samples'] = metrics['n_samples']
            record['evaluation_device'] = metrics.get('device')
        records.append(record)
    return records


def _mean_std(values):
    """Mean and sample standard deviation; a single value has no spread."""
    values = [value for value in values if value is not None]
    if not values:
        return None, None
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else None


def aggregate_configs(records):
    """Mean and sample standard deviation over each config's finished seeds."""
    groups = {}
    for record in records:
        groups.setdefault((record['dataset'], record['config']), []).append(record)
    columns = (['val_accuracy', 'val_cross_entropy', 'train_seconds']
               + [f'test_{metric}' for metric in METRICS])
    rows = []
    for (dataset, name), group in groups.items():
        finished = [item for item in group if item['finished']]
        row = {'dataset': dataset, 'config': name,
               'changed_factor': group[0]['changed_factor'],
               'description': REGISTRY[dataset][name].description if name in REGISTRY[dataset] else '',
               'parameters': group[0]['parameters'], 'seeds_planned': len(FORMAL_SEEDS),
               'seeds_finished': len(finished),
               'seeds_evaluated': sum(1 for item in finished if 'test_accuracy' in item)}
        for column in columns:
            mean, spread = _mean_std([item.get(column) for item in finished])
            row[f'{column}_mean'] = mean
            row[f'{column}_std'] = spread
        rows.append(row)
    return rows


def select_best_config(config_rows):
    """BEST_CONFIG_RULE: validation columns and parameter count only.

    Only configs with every seed finished compete, so means stay comparable.
    Rounding keeps float noise from deciding an exact tie in accuracy.
    """
    candidates = [row for row in config_rows if row['seeds_finished'] == row['seeds_planned']]
    if not candidates:
        return None
    return min(candidates, key=lambda row: (-round(row['val_accuracy_mean'], 10),
                                            round(row['val_cross_entropy_mean'], 10),
                                            row['parameters']))


def select_final_seed(records):
    """CHECKPOINT_RULE applied across the chosen configuration's seeds."""
    return min(records, key=lambda item: (-item['val_accuracy'], item['val_cross_entropy'],
                                          item['seed']))


def _write_csv(path, rows):
    fields = []
    for row in rows:
        fields += [key for key in row if key not in fields]

    def write(target):
        with open(target, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    atomic_write(path, write)


def _mean_curves(run_dirs, column):
    """Mean of one history column over finished seeds, which share an epoch count."""
    histories = [read_json(Path(run_dir) / 'history.json') for run_dir in run_dirs]
    lengths = {len(history) for history in histories}
    if len(lengths) != 1:
        raise ValueError(f'Finished seeds have different lengths {lengths}: {run_dirs}')
    epochs = list(range(1, lengths.pop() + 1))
    return epochs, [statistics.fmean(history[index][column] for history in histories)
                    for index in range(len(epochs))]


def _style_axes(axes, title, y_label):
    axes.set_facecolor(SURFACE)
    axes.set_title(title, loc='left', fontsize=10, color=INK)
    axes.set_xlabel('Epoch', color=MUTED, fontsize=9)
    axes.set_ylabel(y_label, color=MUTED, fontsize=9)
    axes.set_yscale('log')
    axes.grid(True, axis='y', which='major', color=GRID, linewidth=0.6)
    axes.set_axisbelow(True)
    axes.tick_params(which='both', colors=MUTED, labelsize=8, length=0)
    for side in ('top', 'right', 'left'):
        axes.spines[side].set_visible(False)
    axes.spines['bottom'].set_color(GRID)


def save_learning_curves(dataset, records, path):
    """Training and validation loss by epoch, one line per configuration.

    Only finished runs are drawn. MSE training loss lives on a different scale,
    so it gets its own panel rather than a second axis; validation cross-entropy
    is shared by all configurations. Returns False if there is nothing to draw.
    """
    colors = {name: SERIES_COLORS[index] for index, name in enumerate(REGISTRY[dataset])}
    run_dirs = {name: [item['run_dir'] for item in records
                       if item['dataset'] == dataset and item['config'] == name
                       and item['finished']]
                for name in REGISTRY[dataset]}
    names = [name for name, paths in run_dirs.items() if paths]
    if not names:
        return False
    cross_entropy = [name for name in names if REGISTRY[dataset][name].loss == 'cross_entropy']
    squared_error = [name for name in names if REGISTRY[dataset][name].loss != 'cross_entropy']
    panels = [('Training loss (cross-entropy objective)', 'Cross-entropy, log scale',
               cross_entropy, 'train_loss')]
    if squared_error:
        panels.append(('Training loss (MSE objective)', 'Squared error, log scale',
                       squared_error, 'train_loss'))
    panels.append(('Validation loss (all configurations)', 'Cross-entropy, log scale',
                   names, 'val_loss'))
    widths = [1.0, 0.62, 1.0] if squared_error else [1.0, 1.0]
    figure, axes_list = plt.subplots(1, len(panels), figsize=(11.5, 4.1), facecolor=SURFACE,
                                     gridspec_kw={'width_ratios': widths})
    handles = {}
    for axes, (title, label, members, column) in zip(axes_list, panels):
        _style_axes(axes, title, label)
        for name in members:
            epochs, values = _mean_curves(run_dirs[name], column)
            line, = axes.plot(epochs, values, color=colors[name], label=name,
                              linewidth=2.4 if name == 'baseline' else 1.4,
                              zorder=3 if name == 'baseline' else 2)
            handles.setdefault(name, line)
    counts = {len(run_dirs[name]) for name in names}
    if counts == {1}:
        subtitle, labels = 'one seed', names
    elif len(counts) == 1:
        subtitle, labels = f'mean of {min(counts)} seeds', names
    else:
        subtitle = 'mean of finished seeds'
        labels = [f'{name} (n={len(run_dirs[name])})' for name in names]
    figure.suptitle(f'{DATASET_TITLES[dataset]} learning curves ({subtitle})',
                    x=0.01, ha='left', fontsize=12, color=INK, fontweight='bold')
    figure.legend([handles[name] for name in names], labels, loc='lower center',
                  ncol=min(len(names), 8), frameon=False, fontsize=9, labelcolor=INK)
    figure.tight_layout(rect=(0, 0.09, 1, 0.95))
    _save_figure(figure, path)
    return True


def _format_percent(mean, spread):
    if mean is None:
        return 'n/a'
    return f'{100 * mean:.2f}' if spread is None else f'{100 * mean:.2f} ± {100 * spread:.2f}'


def _format_seconds(mean):
    return 'n/a' if mean is None else f'{mean:.0f}'


def _comparison_table(rows, best_name):
    header = ('| Config | Changed | Params | Seeds | Val acc % | Test acc % | '
              'Precision (macro) % | Recall (macro) % | F1 (macro) % | F1 (weighted) % | '
              'Train s |\n'
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
    lines = []
    for row in rows:
        marker = ' **(selected)**' if row['config'] == best_name else ''
        lines.append(
            f"| {row['config']}{marker} | {row['changed_factor']} | {row['parameters']:,} | "
            f"{row['seeds_finished']}/{row['seeds_planned']} | "
            f"{_format_percent(row['val_accuracy_mean'], row['val_accuracy_std'])} | "
            f"{_format_percent(row['test_accuracy_mean'], row['test_accuracy_std'])} | "
            f"{_format_percent(row['test_precision_macro_mean'], row['test_precision_macro_std'])} | "
            f"{_format_percent(row['test_recall_macro_mean'], row['test_recall_macro_std'])} | "
            f"{_format_percent(row['test_f1_macro_mean'], row['test_f1_macro_std'])} | "
            f"{_format_percent(row['test_f1_weighted_mean'], row['test_f1_weighted_std'])} | "
            f"{_format_seconds(row['train_seconds_mean'])} |")
    return header + '\n'.join(lines) + '\n'


def _summary_markdown(records, config_rows, best, curves):
    devices = sorted({f"{item['device']} ({item['device_name']})" for item in records})
    states = {}
    for item in records:
        states[item['state']] = states.get(item['state'], 0) + 1
    total_hours = sum(item['train_seconds'] or 0 for item in records) / 3600
    unfinished = [f"{item['dataset']}/{item['config']}/seed_{item['seed']} ({item['state']})"
                  for item in records if not item['finished']]
    parts = [
        '# Stage 3 experiment summary\n',
        'Generated from the saved runs by `run_stage3.py summarize`. Values are '
        'mean ± sample standard deviation over finished seeds.\n',
        f'- Formal runs: {len(records)} ({states})',
        f'- Unfinished runs: {", ".join(unfinished) if unfinished else "none"}',
        f'- Devices: {", ".join(devices)}',
        f'- Summed training time: {total_hours:.2f} hours',
        f'- Checkpoint rule: {CHECKPOINT_RULE}',
        f'- Configuration rule: {BEST_CONFIG_RULE}\n',
    ]
    for dataset in DATASET_ORDER:
        rows = [row for row in config_rows if row['dataset'] == dataset]
        if not rows:
            continue
        order = list(REGISTRY[dataset])
        rows.sort(key=lambda row: order.index(row['config']) if row['config'] in order else 99)
        choice = best.get(dataset)
        parts.append(f'## {DATASET_TITLES[dataset]}\n')
        parts.append(_comparison_table(rows, choice['config'] if choice else None))
        if dataset in curves:
            parts.append(f'![Learning curves](figures/learning_curves_{dataset}.png)\n')
        if not choice:
            parts.append('No configuration has every seed finished, so none is selected.\n')
            continue
        if not choice['panel_complete']:
            parts.append('**Provisional:** some configurations have unfinished seeds and '
                         'were left out of this choice.\n')
        if choice['final_test_accuracy'] is not None:
            target = REFERENCE_ACCURACY[dataset]
            reached = 'reached' if choice['final_test_accuracy'] > target else 'NOT reached'
            parts.append(
                f"Selected by validation: `{choice['config']}`. Final model: seed "
                f"{choice['final_seed']}, test accuracy "
                f"{100 * choice['final_test_accuracy']:.2f}%. Reference accuracy "
                f"above {100 * target:.0f}%: {reached}.\n")
    return '\n'.join(parts)


def summarize(root, method_factory=None):
    """Rebuild every table and figure from the saved runs."""
    root = Path(root)
    output = root / 'summary'
    records = collect_runs(root)
    if not records:
        raise FileNotFoundError(f'No formal runs under {root}')
    config_rows = aggregate_configs(records)
    best = {}
    for dataset in DATASET_ORDER:
        rows = [row for row in config_rows if row['dataset'] == dataset]
        chosen = select_best_config(rows)
        if chosen is None:
            continue
        seeds = [item for item in records if item['dataset'] == dataset
                 and item['config'] == chosen['config'] and item['finished']]
        final = select_final_seed(seeds)
        best[dataset] = {
            'config': chosen['config'], 'val_accuracy_mean': chosen['val_accuracy_mean'],
            'val_cross_entropy_mean': chosen['val_cross_entropy_mean'],
            'test_accuracy_mean': chosen['test_accuracy_mean'],
            'final_seed': final['seed'], 'final_run_dir': final['run_dir'],
            'final_test_accuracy': final.get('test_accuracy'),
            'reference_accuracy': REFERENCE_ACCURACY[dataset],
            'panel_complete': (len(rows) == len(REGISTRY[dataset]) and all(
                row['seeds_finished'] == row['seeds_planned'] for row in rows)),
            'selection_rule': BEST_CONFIG_RULE,
        }
    _write_csv(output / 'runs.csv', records)
    write_json(output / 'runs.json', records)
    _write_csv(output / 'config_comparison.csv', config_rows)
    write_json(output / 'config_comparison.json', config_rows)
    write_json(output / 'best_configs.json', best)
    curves = {dataset for dataset in DATASET_ORDER
              if save_learning_curves(dataset, records, output / 'figures'
                                      / f'learning_curves_{dataset}.png')}
    if method_factory:
        for dataset, configs in REGISTRY.items():
            for name, config in configs.items():
                method = method_factory(config)
                write_text(output / 'architecture' / f'{dataset}_{name}.txt',
                           model_summary_text(method))
                if name == 'baseline':
                    save_architecture_figure(method, output / 'figures'
                                             / f'architecture_{dataset}_baseline.png')
    write_text(output / 'SUMMARY.md', _summary_markdown(records, config_rows, best, curves))
    return output
