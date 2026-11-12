import sys
from script_cnn import parse_args, run_experiment


if __name__=='__main__':
    run_experiment(parse_args(['--dataset', 'mnist', *sys.argv[1:]]))
