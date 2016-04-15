#!/usr/bin/env bash
set -ex

INPUT1="-i  /home/amir/idiap/remote/user/mobio-male/baselines/gmm/male/ztnorm/scores-dev /home/amir/idiap/remote/user/mobio-male/baselines/gabor-graph/male/ztnorm/scores-dev  /home/amir/idiap/remote/user/mobio-male/voice/gmm/male/ztnorm/scores-dev  /home/amir/idiap/remote/user/mobio-male/voice/isv/male/ztnorm/scores-dev"
INPUT2="-I  /home/amir/idiap/remote/user/mobio-male/baselines/gmm/male/ztnorm/scores-eval /home/amir/idiap/remote/user/mobio-male/baselines/gabor-graph/male/ztnorm/scores-eval  /home/amir/idiap/remote/user/mobio-male/voice/gmm/male/ztnorm/scores-eval  /home/amir/idiap/remote/user/mobio-male/voice/isv/male/ztnorm/scores-eval"

for HIDDEN_NODES in 5 10 25 50 100 200; do
	./bin/fuse.py -vvv $INPUT1 $INPUT2 -o "/home/amir/idiap/remote/user/mobio-male/face-voice/F-gmm-gabor-graph-S-gmm-isv_MLP_${HIDDEN_NODES}/male/ztnorm/scores-dev" -O "/home/amir/idiap/remote/user/mobio-male/face-voice/F-gmm-gabor-graph-S-gmm-isv_MLP_${HIDDEN_NODES}/male/ztnorm/scores-eval" -a "bob.fusion.base.algorithm.MLP(preprocessors=[(sklearn.preprocessing.RobustScaler(), False)], n_systems=4, hidden_layers=[${HIDDEN_NODES}], seed=0)" --force --imports='sklearn.preprocessing','bob.fusion.base'
done
