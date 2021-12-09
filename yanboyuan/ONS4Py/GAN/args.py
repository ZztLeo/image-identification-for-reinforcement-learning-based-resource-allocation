import argparse

parser = argparse.ArgumentParser(
    description='GRWA Training')

parser.add_argument('--mode', type=str, default='alg',
                    help='RWA执行的模式，alg表示使用ksp+FirstFit，learning表示CNN学习模式, fcl表示FC学习模式，lstml表示LSTM学习模式')


args = parser.parse_args()