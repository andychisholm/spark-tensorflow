import math
import numpy
import random
import types
from itertools import izip, tee, imap
from operator import itemgetter, add
from cStringIO import StringIO

import cPickle as pickle
import tensorflow as tf

class Graph(object):
    def transform_batch(self, data):
        raise NotImplemented
    def train_and_loss_for_batch(self, data):
        return self.train, self.loss

class DistributedGraph(object):
    def __init__(self, sc, partitions, model):
        self.sc = sc
        self.partitions = partitions
        self.params = {n:sc.broadcast(v) for n, v in self.get_initial_params(model)}

    @staticmethod
    def get_initial_params(builder):
        with tf.Graph().as_default(), tf.Session() as session:
            builder()
            session.run(tf.initialize_all_variables())
            return [(v.name, session.run(v)) for v in tf.all_variables()]

    def initialize_all_variables(self, session):
        for v in tf.all_variables():
            session.run(v.assign(self.params[v.name].value))

    @staticmethod
    def shuffle_and_batch(dataset, partitions, batch_size, fraction=1.0):
        return dataset\
            .sample(False, fraction)\
            .zipWithIndex()\
            .map(lambda (k,v): (v % partitions, k))\
            .partitionBy(partitions)\
            .map(lambda (k,v): v)\
            .mapPartitions(lambda items: izip(*([iter(items)] * batch_size)), preservesPartitioning=True)

    @staticmethod
    def get_worker_op(graph_init, params, op):
        def worker_op(partition):
            with tf.Graph().as_default(), tf.Session() as s:
                g = graph_init()
                for v in tf.all_variables():
                    s.run(v.assign(params[v.name].value))
                for r in op(s, g, partition):
                    yield r
        return worker_op

    def evaluate(self, dataset, graph_init):
        def eval_partition(s, g, items):
            for b in items:
                yield s.run(g.evaluate, g.transform_batch(b))

        return dataset\
            .mapPartitions(self.get_worker_op(graph_init, self.params, eval_partition))\
            .flatMap(lambda rs: rs)\
            .mean()

    def train(self, dataset, graph_init, worker_epochs = 2):
        # todo: add option to train until a timeout elapses instead of fixed number of epochs
        # this should reduce latency from variance in batch runtime between workers
        num_partitions = self.partitions
        def train_on_partition(s, g, batches):
            loss = 0.
            # todo: only transform batch once
            batches = list(batches)
            num_batches = len(batches) * worker_epochs
            for i in xrange(worker_epochs):
                for batch in batches:
                    loss += s.run(g.train_and_loss_for_batch(batch), g.transform_batch(batch))[1]
                if (i + 1) != worker_epochs:
                    random.shuffle(batches)

            for v in tf.trainable_variables():
                yield v.name, s.run(v/num_partitions)

            yield '_loss', loss / num_batches

        results = dict(dataset\
            .mapPartitions(self.get_worker_op(graph_init, self.params, train_on_partition))\
            .reduceByKey(add)\
            .collect())

        total_loss = results.pop('_loss')

        for n, v in results.iteritems():
            self.params[n].unpersist()
            self.params[n] = self.sc.broadcast(v)
        return total_loss / self.partitions
