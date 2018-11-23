#!/usr/bin/env python
import numpy
from numpy import array
from tempfile import NamedTemporaryFile
import bob.fusion.base
import bob.learn.linear
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import bob.core

logger = bob.core.log.setup("bob.fusion.base")


NEG = array([[-1.23594765, -2.59984279],
             [-2.02126202, -0.7591068],
             [-1.13244201, -3.97727788],
             [-2.04991158, -3.15135721],
             [-3.10321885, -2.5894015],
             [-2.85595643, -1.54572649],
             [-2.23896227, -2.87832498],
             [-2.55613677, -2.66632567],
             [-1.50592093, -3.20515826],
             [-2.6869323, -3.85409574]])
POS = array([[0.44701018, 3.6536186],
             [3.8644362, 2.25783498],
             [5.26975462, 1.54563433],
             [3.04575852, 2.81281615],
             [4.53277921, 4.46935877],
             [3.15494743, 3.37816252],
             [2.11221425, 1.01920353],
             [2.65208785, 3.15634897],
             [4.23029068, 4.20237985],
             [2.61267318, 2.69769725]])
X = numpy.vstack((NEG, POS))
TEST = array([[-1.04855297, -1.42001794],
              [-1.70627019, 1.9507754],
              [-0.50965218, -0.4380743],
              [-1.25279536, 0.77749036],
              [-1.61389785, -0.21274028],
              [-0.89546656, 0.3869025],
              [-0.51080514, -1.18063218],
              [-0.02818223, 0.42833187],
              [0.06651722, 0.3024719],
              [-0.63432209, -0.36274117]])


def run_steps(algorithm):
    algorithm.train_preprocessors(X)
    neg = algorithm.preprocess(NEG)
    pos = algorithm.preprocess(POS)
    algorithm.train(neg, pos)
    fused = algorithm.fuse(TEST)
    with NamedTemporaryFile(suffix='.pkl') as f:
        algorithm.save(f.name)
        loaded_algorithm = algorithm.load(f.name)

    try:
        assert str(algorithm) == str(loaded_algorithm)
    except Exception:
        logger.warn("String comparison of algorithms do not match which is OK."
                    "\n{}\n{}".format(str(algorithm), str(loaded_algorithm)))
    if algorithm.preprocessors:
        assert len(algorithm.preprocessors) == len(loaded_algorithm.preprocessors)
    assert fused.ndim == 1

    return neg, pos, fused, loaded_algorithm


def test_algorithm_llr_sklearn():
    algorithm = bob.fusion.base.algorithm.Algorithm(
        preprocessors=[
            StandardScaler(**{'copy': True, 'with_mean': True, 'with_std': True})],
        classifier=LogisticRegression(**{'C': 1.0,
                                         'class_weight': None,
                                         'dual': False,
                                         'fit_intercept': True,
                                         'intercept_scaling': 1,
                                         'max_iter': 100,
                                         'multi_class': 'ovr',
                                         'n_jobs': 1,
                                         'penalty': 'l2',
                                         'random_state': None,
                                         'solver': 'liblinear',
                                         'tol': 0.0001,
                                         'verbose': 0,
                                         'warm_start': False}))
    neg, pos, fused, loaded_algorithm = run_steps(algorithm)

    assert numpy.allclose(algorithm.preprocessors[0].mean_, array([0.52676307, 0.09832188]))
    assert numpy.allclose(algorithm.preprocessors[0].scale_, array([2.857145, 2.98815147]))

    assert numpy.allclose(neg, array([[-0.61694829, -0.90295445],
                                      [-0.89180811, -0.28694284],
                                      [-0.58072134, -1.36392007],
                                      [-0.90183545, -1.08752154],
                                      [-1.27049272, -0.89946022],
                                      [-1.18395093, -0.5501891],
                                      [-0.96800314, -0.99614993],
                                      [-1.07901413, -0.92520328],
                                      [-0.71143886, -1.10552634],
                                      [-1.12479253, -1.32269654]]))
    assert numpy.allclose(pos, array([[-0.02791349, 1.18979803],
                                      [1.16818472, 0.72269198],
                                      [1.6600458, 0.48435043],
                                      [0.88164775, 0.90841923],
                                      [1.4021046, 1.4627896],
                                      [0.91986384, 1.09761526],
                                      [0.5549075, 0.3081777],
                                      [0.74386312, 1.02338423],
                                      [1.29623369, 1.37344375],
                                      [0.73006799, 0.86989411]]))

    assert numpy.allclose(algorithm.classifier.intercept_, array([0.04577333]))
    assert numpy.allclose(algorithm.classifier.classes_, array([False, True], dtype=bool))
    assert numpy.allclose(algorithm.classifier.coef_, array([[1.33489128, 1.38092354]]))

    assert numpy.allclose(fused, array([-3.31486708, 0.4619598, -1.23950404, -0.55291754, -2.40238289,
                                        -0.61529441, -2.26645877, 0.59964668, 0.55225715, -1.30189552]))

    assert numpy.allclose(algorithm.preprocessors[0].mean_, loaded_algorithm.preprocessors[0].mean_)
    assert numpy.allclose(algorithm.preprocessors[0].scale_, loaded_algorithm.preprocessors[0].scale_)
    assert numpy.allclose(algorithm.classifier.intercept_, loaded_algorithm.classifier.intercept_)
    assert numpy.allclose(algorithm.classifier.classes_, loaded_algorithm.classifier.classes_)
    assert numpy.allclose(algorithm.classifier.coef_, loaded_algorithm.classifier.coef_)


def test_algorithm_llr_bob():
    algorithm = bob.fusion.base.algorithm.LLR(
        preprocessors=[StandardScaler(**{'copy': True,
                                         'with_mean': True,
                                         'with_std': True})],
        trainer=None)
    neg, pos, fused, loaded_algorithm = run_steps(algorithm)

    algorithm = bob.fusion.base.algorithm.LLR(
        preprocessors=[StandardScaler(**{'copy': True,
                                         'with_mean': True,
                                         'with_std': True})],
        trainer=bob.learn.linear.CGLogRegTrainer(
            prior=0.5, convergence_threshold=1e-5,
            max_iterations=10000, reg=1., mean_std_norm=False))
    neg, pos, fused, loaded_algorithm = run_steps(algorithm)
    assert algorithm.machine.is_similar_to(loaded_algorithm.machine)

    assert numpy.allclose(algorithm.machine.biases, array([0.04577333]), atol=0.05)
    assert numpy.allclose(algorithm.machine.weights, array([[1.33489128, 1.38092354]]), atol=0.05)

    assert numpy.allclose(fused, array([-3.31486708, 0.4619598, -1.23950404, -0.55291754, -2.40238289,
                                        -0.61529441, -2.26645877, 0.59964668, 0.55225715, -1.30189552]), atol=0.05)

    assert numpy.allclose(algorithm.machine.biases, loaded_algorithm.machine.biases)
    assert numpy.allclose(algorithm.machine.weights, loaded_algorithm.machine.weights)


def test_weighted_sum_1():
    algorithm = bob.fusion.base.algorithm.Weighted_Sum()
    neg, pos, fused, loaded_algorithm = run_steps(algorithm)
    assert str(algorithm) == "<class 'bob.fusion.base.algorithm.Weighted_Sum'>()", str(algorithm)
    assert numpy.allclose(fused, numpy.mean(TEST, axis=1))
    assert algorithm.weights == loaded_algorithm.weights


def test_weighted_sum_2():
    weights = [0.3, 0.7]
    algorithm = bob.fusion.base.algorithm.Weighted_Sum(weights=weights)
    neg, pos, fused, loaded_algorithm = run_steps(algorithm)
    assert str(
        algorithm) == "<class 'bob.fusion.base.algorithm.Weighted_Sum'>(weights=[0.3, 0.7])"
    assert numpy.allclose(fused, numpy.sum(TEST * weights, axis=1))
    assert algorithm.weights == loaded_algorithm.weights


def test_mlp():
    algorithm = bob.fusion.base.algorithm.MLP(
        n_systems=2, hidden_layers=[3], seed=0, batch_size=10, epoch=1,
        max_iter=1000, no_improvements=0, valley_condition=0.9,
        preprocessors=[StandardScaler(**{'copy': True,
                                         'with_mean': True,
                                         'with_std': True})])
    assert numpy.allclose(algorithm.machine.weights[0],
                          array([[0.0097627, 0.01856892, 0.04303787],
                                 [0.06885315, 0.02055267, 0.07158912]]))
    assert numpy.allclose(algorithm.machine.weights[1],
                          array([[0.02471274],
                                 [0.02917882],
                                 [-0.02312366]]))
    assert numpy.allclose(algorithm.machine.biases[0],
                          array([0.00897664, 0.06945035, -0.01526904]))
    assert numpy.allclose(algorithm.machine.biases[1],
                          array([-0.01248256]))
    _, _, fused, loaded_algorithm = run_steps(algorithm)
    assert algorithm.machine.is_similar_to(loaded_algorithm.machine)
    assert numpy.allclose(fused,
                          [-1, 1, -1, -1, -1, -1, -1, 1, 1, -1], atol=0.001)


def test_gmm():
    algorithm = bob.fusion.base.algorithm.GMM(
       number_of_gaussians=None,
       kmeans_training_iterations=25,
       gmm_training_iterations=25,
       training_threshold=5e-4,
       variance_threshold=5e-4,
       update_weights=True,
       update_means=True,
       update_variances=True,
       responsibility_threshold=0,
       init_seed=5489,)
    _, _, fused, loaded_algorithm = run_steps(algorithm)
    assert algorithm.machine.is_similar_to(loaded_algorithm.machine)
    assert numpy.allclose(algorithm.machine.means,
                          array([[5.26975462, 1.54563433],
                                 [2.55902882, 2.71385423],
                                 [4.38210031, 4.33636831]]))
    assert numpy.allclose(algorithm.machine.variances,
                          array([[5.00000000e-04, 5.00000000e-04],
                                 [9.99478941e-01, 6.62039927e-01],
                                 [2.28745081e-02, 1.78191869e-02]]))
    assert numpy.allclose(algorithm.machine.weights,
                          array([0.09999619, 0.7013807, 0.19862312]))
    assert numpy.allclose(fused,
                          array([-21.40306699, -11.52700531, -14.20002458,
                                 -12.08667715, -17.16591322, -12.04539528,
                                 -18.15526862, -9.27977024, -9.48558409,
                                 -14.23620154]))
