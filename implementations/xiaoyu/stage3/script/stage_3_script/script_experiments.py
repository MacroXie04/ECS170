import csv
from script_cnn import PROJECT_ROOT, parse_args, run_experiment


if __name__=='__main__':
    all_rows=[]
    for name in ['orl', 'mnist', 'cifar']:
        channels=['32', '64', '128'] if name=='cifar' else ['16', '32', '64']
        second=('avg_pool', ['--pooling', 'avg']) if name=='cifar' else ('kernel5', ['--kernel-size', '5'])
        rows=[]
        for run_name, options in [('baseline', []), ('deeper', ['--channels', *channels]), second]:
            args=parse_args(['--dataset', name, '--run-name', run_name, *options])
            scores=run_experiment(args)
            rows.append({'dataset': name, 'run_name': run_name, **scores})
        with (PROJECT_ROOT/'result/stage_3_result'/name/'ablation_summary.csv').open('w', newline='') as f:
            writer=csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        all_rows.extend(rows)
    with (PROJECT_ROOT/'result/stage_3_result/experiments.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
