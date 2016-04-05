#!/usr/bin/env python

import os
import numpy
import matplotlib.pyplot as plt
from bob.measure.load import load_score, get_all_scores,\
    get_negatives_positives_all
from bob.fusion.base.algorithm import LLR, Weighted_Sum, MLP
from bob.fusion.base.normalizer import ZNorm


def main():
  '''
/home/amir/idiap/remote/user/mobio-male/baselines/gmm/male/nonorm/scores-dev
/home/amir/idiap/remote/user/mobio-male/baselines/gmm/male/nonorm/scores-eval
/home/amir/idiap/remote/user/mobio-male/baselines/gmm/male/ztnorm/scores-dev
/home/amir/idiap/remote/user/mobio-male/baselines/gmm/male/ztnorm/scores-eval

/home/amir/idiap/remote/user/mobio-male/baselines/gabor-graph/male/nonorm/scores-dev
/home/amir/idiap/remote/user/mobio-male/baselines/gabor-graph/male/nonorm/scores-eval
/home/amir/idiap/remote/user/mobio-male/baselines/gabor-graph/male/ztnorm/scores-dev
/home/amir/idiap/remote/user/mobio-male/baselines/gabor-graph/male/ztnorm/scores-eval


/home/amir/idiap/remote/user/mobio-male/voice/gmm/male/nonorm/scores-dev
/home/amir/idiap/remote/user/mobio-male/voice/gmm/male/nonorm/scores-eval
/home/amir/idiap/remote/user/mobio-male/voice/gmm/male/ztnorm/scores-dev
/home/amir/idiap/remote/user/mobio-male/voice/gmm/male/ztnorm/scores-eval

/home/amir/idiap/remote/user/mobio-male/voice/isv/male/nonorm/scores-dev
/home/amir/idiap/remote/user/mobio-male/voice/isv/male/nonorm/scores-eval
/home/amir/idiap/remote/user/mobio-male/voice/isv/male/ztnorm/scores-dev
/home/amir/idiap/remote/user/mobio-male/voice/isv/male/ztnorm/scores-eval
   # = ['/home/amir/idiap/remote/scores-dev_gmm',
   #               '/home/amir/idiap/remote/scores-dev_isv']
  # s_eval_paths = ['/home/amir/idiap/remote/scores-eval_gmm',
  #                 '/home/amir/idiap/remote/scores-eval_isv']



  '''
  score_path_mask = '/home/amir/idiap/remote/user/mobio-male/{}/male/{}/scores-{}'
  fusion_lists = [
    (('baselines/gmm', 'voice/isv'), 'face-voice/F-gmm-S-isv{}'),
    (('baselines/gmm', 'baselines/gabor-graph'), 'baselines/gmm-gabor-graph{}'),
    (('voice/gmm', 'voice/isv'), 'voice/gmm-isv{}'),
    # (('baselines/gmm', 'baselines/gabor-graph', 'voice/gmm', 'voice/isv'), 'face-voice/F-gmm-gabor-graph-S-gmm-isv{}'),
  ]
  for norm in ['ztnorm']:
    for scores_paths_param, save_path_param in fusion_lists:
      save_path_dev = score_path_mask.format(save_path_param, norm, 'dev')
      save_path_eval = score_path_mask.format(save_path_param, norm, 'eval')
      # for path in [save_path_dev, save_path_eval]:
      #   for tag in [
      #      '_Weighted_Sum',
      #      '_LLR',
      #      '_MLP'
      #      ]:
      #     try:
      #       os.makedirs(os.path.split(path.format(tag))[0])
      #     except Exception:
      #       pass
      s_dev_paths = [
        score_path_mask.format(a, norm, 'dev') for a in scores_paths_param]
      s_eval_paths = [
        score_path_mask.format(a, norm, 'eval') for a in scores_paths_param]

      score_lines_list = [load_score(path) for path in s_dev_paths]
      scores = get_all_scores(score_lines_list)
      trainer_scores = get_negatives_positives_all(score_lines_list)
      score_lines_list_eval = [load_score(path) for path in s_eval_paths]
      scores_eval = get_all_scores(score_lines_list_eval)
      plt.figure()
      for i, (fuse, tag) in enumerate([
              (Weighted_Sum(normalizer=ZNorm(), scores=numpy.array(scores), trainer_scores=trainer_scores), '_Weighted_Sum'),
              (LLR(normalizer=ZNorm(), scores=numpy.array(scores), trainer_scores=trainer_scores), '_LLR'),
              (MLP(normalizer=ZNorm(), scores=numpy.array(scores), trainer_scores=trainer_scores, verbose=True,
                   mlp_shape=(2, 3, 1), batch_size=1, seed=0), '_MLP'),
            ]):
        plt.subplot(2,2,i+1)
        fuse.train()
        # import ipdb; ipdb.set_trace()
        fused_scores_dev = fuse()
        score_lines = numpy.array(score_lines_list[0])
        score_lines['score'] = fused_scores_dev
        # import ipdb
        # ipdb.set_trace()
        # numpy.savetxt(save_path_dev.format(tag), score_lines, fmt='%s %s %s %.6f')

        fuse.scores = numpy.array(scores_eval)
        fused_scores_eval = fuse()
        fuse.scores = numpy.array(scores_eval)

        score_lines = numpy.array(score_lines_list_eval[0])
        score_lines['score'] = fused_scores_eval
        # numpy.savetxt(save_path_eval.format(tag), score_lines, fmt='%s %s %s %.6f')

        # plot the decision boundary
        from bob.measure import eer_threshold, min_hter_threshold
        from bob.measure.load import get_negatives_positives

        score_labels = score_lines['claimed_id'] == score_lines['real_id']
        threshold = eer_threshold(*get_negatives_positives(score_lines))
        thres_system1 = min_hter_threshold(
          *get_negatives_positives(score_lines_list[0]))
        thres_system2 = min_hter_threshold(
          *get_negatives_positives(score_lines_list[1]))
        fuse.plot_boundary_decision(
          score_labels, threshold,
          scores_paths_param[0], scores_paths_param[1],
          thres_system1, thres_system2,
          True,
          seed=0
          )
        plt.title(tag[1:])
        # print(thres_system1, thres_system2)
        # thres_system1, thres_system2 = (3.51670052, 3.19892205)
        # plt.axvline(thres_system1, color='red')
        # plt.axhline(thres_system2, color='red')
        # plt.show()
      plt.savefig('scatter_{}_{}.pdf'.format(norm, '_'.join([a.replace('/','-') for a in scores_paths_param])))
      plt.close()
        # return

        # fuse = LLR(scores=scores, trainer_scores=trainer_scores)
        # fuse.train()
        # print('LLR', fuse()[:10])
        # fuse = Weighted_Sum(scores=scores)
        # print('Weighted_Sum', fuse()[:10])
        # fuse = MEAN(scores=scores)
        # print('MEAN', fuse()[:10])
        # fuse = MLP(scores=scores, trainer_scores=trainer_scores,
        #            verbose=True, max_iter=0, batch_size=1)
        # fuse.train()
        # print('MLP', fuse()[:10])

if __name__ == '__main__':
  main()
