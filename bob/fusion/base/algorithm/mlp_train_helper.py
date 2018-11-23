#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Tue 16 Aug 14:39:22 2011

"""Trains an MLP using RProp
"""

import sys
import bob.measure
import bob.learn.mlp
import bob.learn.activation
import numpy
import numpy.linalg as la

import logging

logger = logging.getLogger(__name__)


class Analyzer(object):
    """Can analyze results in the end of a run. It can also save itself"""

    def gentargets(self, data, target):
        t = numpy.vstack(data.shape[0] * (target,))
        return t, numpy.empty_like(t)

    def __init__(self, train, devel, target):
        super(Analyzer, self).__init__()

        self.train = train
        self.devel = devel
        self.target = target

        real_train = self.gentargets(train[0], target[0])
        attack_train = self.gentargets(train[1], target[1])
        real_devel = self.gentargets(devel[0], target[0])
        attack_devel = self.gentargets(devel[1], target[1])

        self.train_target = (real_train[0], attack_train[0])
        self.train_output = (real_train[1], attack_train[1])
        self.devel_target = (real_devel[0], attack_devel[0])
        self.devel_output = (real_devel[1], attack_devel[1])

        self.data = {}  # where to store variables that will be saved
        self.data['epoch'] = []
        self.data['real-train-rmse'] = []
        self.data['attack-train-rmse'] = []
        self.data['real-devel-rmse'] = []
        self.data['attack-devel-rmse'] = []
        self.data['train-far'] = []
        self.data['train-frr'] = []
        self.data['devel-far'] = []
        self.data['devel-frr'] = []

    def __call__(self, machine, iteration):
        """Computes current outputs and evaluate performance"""

        def evalperf(outputs, targets):
            return la.norm(bob.measure.rmse(outputs, targets))

        for k in range(len(self.train)):
            machine(self.train[k], self.train_output[k])
            machine(self.devel[k], self.devel_output[k])

        self.data['real-train-rmse'].append(evalperf(self.train_output[0],
                                                     self.train_target[0]))
        self.data['attack-train-rmse'].append(evalperf(self.train_output[1],
                                                       self.train_target[1]))
        self.data['real-devel-rmse'].append(evalperf(self.devel_output[0],
                                                     self.devel_target[0]))
        self.data['attack-devel-rmse'].append(evalperf(self.devel_output[1],
                                                       self.devel_target[1]))

        thres = bob.measure.eer_threshold(self.train_output[1][:, 0],
                                          self.train_output[0][:, 0])
        train_far, train_frr = bob.measure.farfrr(
            self.train_output[1][:, 0], self.train_output[0][:, 0], thres)
        devel_far, devel_frr = bob.measure.farfrr(
            self.devel_output[1][:, 0], self.devel_output[0][:, 0], thres)

        self.data['train-far'].append(train_far)
        self.data['train-frr'].append(train_frr)
        self.data['devel-far'].append(devel_far)
        self.data['devel-frr'].append(devel_frr)

        self.data['epoch'].append(iteration)

    def str_header(self):
        """Returns the string header of what I can print"""
        return "iteration: RMSE:real/RMSE:attack (EER:%) ( train | devel )"

    def __str__(self):
        """Returns a string representation of myself"""

        retval = "%d: %.4e/%.4e (%.2f%%) | %.4e/%.4e (%.2f%%)" % \
            (self.data['epoch'][-1],
                 self.data['real-train-rmse'][-1],
                 self.data['attack-train-rmse'][-1],
                 50 *
                 (self.data['train-far'][-1] + self.data['train-frr'][-1]),
                 self.data['real-devel-rmse'][-1],
                 self.data['attack-devel-rmse'][-1],
                 50 *
                 (self.data['devel-far'][-1] + self.data['devel-frr'][-1]),
             )
        return retval

    def save(self, f):
        """Saves my contents on the bob.io.base.HDF5File you give me."""

        for k, v in self.data.items():
            f.set(k, numpy.array(v))

    def load(self, f):
        """Loads my contents from the bob.io.base.HDF5File you give me."""

        for k in f.paths():
            self.data[k.strip('/')] = f.read(k)


class MLPTrainer(object):
    """Creates a randomly initialized MLP and train it using the input data.

        This method will create an MLP with the shape (`mlp_shape`) that is
        provided. Then it will initialize the MLP with random weights and
        biases and train it for as long as the development shows improvement
        and will stop as soon as it does not anymore or we reach the maximum
        number of iterations.

        Performance is evaluated both on the trainining and development set
        during the training, every 'epoch' training steps. Each training step
        is composed of `batch_size` elements drawn randomly from all classes
        available in train set.

        Keyword Parameters:

        train
            An iterable (tuple or list) containing two arraysets: the first
            contains the real accesses (target = +1) and the second contains
            the attacks (target = -1).

        devel
            An iterable (tuple or list) containing two arraysets: the first
            contains the real accesses (target = +1) and the second contains
            the attacks (target = -1).

        batch_size
            An integer defining the number of samples per training iteration.
            Good values are greater than 100.

        mlp_shape
            Shape of the MLP machine.

        epoch
            The number of training steps to wait until we measure the error.

        max_iter
            If given (and different than zero), should tell us the maximum
            number of training steps to train the network for. If set to 0
            just train until the development sets reaches the valley (in RMSE
            terms).

        no_improvements
            If given (and different than zero), should tell us the maximum
            number of iterations we should continue trying for in case we have
            no more improvements on the development set average RMSE term.
            This value, if set, should not be too small as this may cause a
            too-early stop. Values in the order of 10% of the max_iter should
            be fine.

        """

    def __init__(self,
                 train,
                 devel,
                 mlp_shape,
                 batch_size=1,
                 epoch=1,
                 max_iter=1000,
                 no_improvements=0,
                 valley_condition=1,
                 machine=None,
                 trainer=None,
                 *args, **kwargs
                 ):
        super(MLPTrainer, self).__init__()
        self.train = train
        self.devel = devel
        self.mlp_shape = mlp_shape
        self.batch_size = batch_size
        self.epoch = epoch
        self.max_iter = max_iter
        self.no_improvements = no_improvements
        self.valley_condition = valley_condition
        self.machine = machine
        self.trainer = trainer

    def __call__(self):
        return self.make_mlp()

    def make_mlp(self):

        # of the minimum devel. set RMSE detected so far
        VALLEY_CONDITION = self.valley_condition
        last_devel_rmse = 0

        def stop_condition(min_devel_rmse, devel_rmse, last_devel_rmse):
            """This method will detect a valley in the devel set RMSE"""
            stop = (VALLEY_CONDITION * devel_rmse) > (min_devel_rmse) or \
                abs(devel_rmse - last_devel_rmse) / \
                (devel_rmse + last_devel_rmse) < 0.00001
            return stop

        target = [
            numpy.array([+1], 'float64'),
            numpy.array([-1], 'float64'),
        ]

        logger.info("Preparing analysis framework...")
        analyze = Analyzer(self.train, self.devel, target)

        logger.info("Setting up training infrastructure...")
        shuffler = bob.learn.mlp.DataShuffler(self.train, target)
        shuffler.auto_stdnorm = True

        # shape = (shuffler.data_width, nhidden, 1)
        # machine = bob.learn.mlp.Machine(self.shape)
        # machine.activation = bob.learn.activation.HyperbolicTangent() #the
        # defaults are anyway Hyperbolic Tangent for hidden and output layer
        self.machine.input_subtract, self.machine.input_divide = \
            shuffler.stdnorm()

        # trainer = bob.learn.mlp.RProp(
        #     self.batch_size,
        #     bob.learn.mlp.SquareError(machine.output_activation), machine)

        self.trainer.train_biases = True

        continue_training = True
        iteration = 0
        min_devel_rmse = sys.float_info.max
        self.best_machine = bob.learn.mlp.Machine(self.machine)  # deep copy
        best_machine_iteration = 0

        # temporary training data selected by the shuffer
        shuffled_input = numpy.ndarray(
            (self.batch_size, shuffler.data_width), 'float64')
        shuffled_target = numpy.ndarray(
            (self.batch_size, shuffler.target_width), 'float64')

        logger.info(analyze.str_header())

        try:
            while continue_training:

                analyze(self.machine, iteration)

                logger.info(analyze)

                avg_devel_rmse = (analyze.data['real-devel-rmse'][-1] +
                                  analyze.data['attack-devel-rmse'][-1]) / 2

                # save best network, record minima
                if avg_devel_rmse < min_devel_rmse:
                    best_machine_iteration = iteration
                    self.best_machine = bob.learn.mlp.Machine(
                        self.machine)  # deep copy
                    logger.info("%d: Saving best network so far with average "
                                "devel. RMSE = %.4e", iteration, avg_devel_rmse)
                    min_devel_rmse = avg_devel_rmse
                    logger.info("%d: New valley stop threshold set to %.4e",
                                iteration, avg_devel_rmse / VALLEY_CONDITION)
                if stop_condition(min_devel_rmse, avg_devel_rmse, last_devel_rmse) \
                        or numpy.allclose(avg_devel_rmse / VALLEY_CONDITION, 0):
                    logger.info("%d: Stopping on devel valley condition", iteration)
                    logger.info("%d: Best machine happened on iteration %d with average "
                                "devel. RMSE of %.4e", iteration, best_machine_iteration,
                                min_devel_rmse)

                    break
                last_devel_rmse = avg_devel_rmse

                # train for 'epoch' times w/o stopping for tests
                for i in range(self.epoch):
                    shuffler(data=shuffled_input, target=shuffled_target)
                    self.trainer.batch_size = len(shuffled_input)
                    self.trainer.train(
                        self.machine, shuffled_input, shuffled_target)
                    iteration += 1

                if self.max_iter > 0 and iteration > self.max_iter:
                    logger.info("%d: Stopping on max. iterations condition", iteration)
                    logger.info("%d: Best machine happened on iteration %d with average "
                                "devel. RMSE of %.4e", iteration, best_machine_iteration,
                                min_devel_rmse)
                    break

                if self.no_improvements > 0 and \
                        (iteration - best_machine_iteration) > self.no_improvements:
                    logger.info("%d: Stopping because did not observe MLP performance "
                                "improvements for %d iterations",
                                iteration, iteration - best_machine_iteration)
                    logger.info("%d: Best machine happened on iteration %d with average "
                                "devel. RMSE of %.4e",
                                iteration, best_machine_iteration, min_devel_rmse)
                    break

        except KeyboardInterrupt:
            logger.info("%d: User interruption captured - exiting in a clean way",
                        iteration)
            logger.info("%d: Best machine happened on iteration %d "
                        "with average devel. RMSE of %.4e",
                        iteration, best_machine_iteration, min_devel_rmse)

            analyze(self.machine, iteration)

        return self.best_machine, analyze
