from __future__ import print_function

import collections
import time

import tensorflow as tf

from tensorflow.python.ops import lookup_ops

from .utils import misc_utils as utils


class TrainModel(
    collections.namedtuple(
        "TrainModel", ("graph", "model", "iterator",
                       "skip_count_placeholder"))):
    pass


class EvalModel(
    collections.namedtuple(
        "EvalModel", ("graph", "model", "src_file_placeholder",
                      "tgt_file_placeholder", "iterator"))):
    pass


class InferModel(
    collections.namedtuple(
        "InferModel", ("graph", "model", "src_placeholder",
                       "batch_size_placeholder", "iterator"))):
    pass


def create_train_model(
        model_creator, hparams, scope=None, num_workers=1, jobid=0):
    """Create train graph, model, and iterator."""
    src_file = "%s.%s" % (hparams.train_prefix, hparams.src)
    tgt_file = "%s.%s" % (hparams.train_prefix, hparams.tgt)
    src_vocab_file = hparams.src_vocab_file
    tgt_vocab_file = hparams.tgt_vocab_file

    graph = tf.Graph()

    with graph.as_default(), tf.container(scope or "train"):
        src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
            src_vocab_file, tgt_vocab_file, hparams.share_vocab)

        src_dataset = tf.data.TextLineDataset(src_file)
        tgt_dataset = tf.data.TextLineDataset(tgt_file)
        skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)

        iterator = iterator_utils.get_iterator(
            src_dataset,
            tgt_dataset,
            src_vocab_table,
            tgt_vocab_table,
            batch_size=hparams.batch_size,
            sos=hparams.sos,
            eos=hparams.eos,
            source_reverse=hparams.source_reverse,
            random_seed=hparams.random_seed,
            num_buckets=hparams.num_buckets,
            src_max_len=hparams.src_max_len,
            tgt_max_len=hparams.tgt_max_len,
            skip_count=skip_count_placeholder,
            num_shards=num_workers,
            shard_index=jobid)

        model = model_creator(
            hparams,
            iterator=iterator,
            mode=tf.contrib.learn.ModeKeys.TRAIN,
            source_vocab_table=src_vocab_table,
            target_vocab_table=tgt_vocab_table,
            scope=scope)

    return TrainModel(
        graph=graph,
        model=model,
        iterator=iterator,
        skip_count_placeholder=skip_count_placeholder)


def create_eval_model(model_creator, hparams, scope=None, extra_args=None):
    """Create train graph, model, src/tgt file holders, and iterator."""
    src_vocab_file = hparams.src_vocab_file
    tgt_vocab_file = hparams.tgt_vocab_file
    graph = tf.Graph()

    with graph.as_default(), tf.container(scope or "eval"):
        src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
            src_vocab_file, tgt_vocab_file, hparams.share_vocab)
        src_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
        tgt_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
        src_dataset = tf.data.TextLineDataset(src_file_placeholder)
        tgt_dataset = tf.data.TextLineDataset(tgt_file_placeholder)

        iterator = iterator_utils.get_iterator(
            src_dataset,
            tgt_dataset,
            src_vocab_table,
            tgt_vocab_table,
            hparams.batch_size,
            sos=hparams.sos,
            eos=hparams.eos,
            source_reverse=hparams.source_reverse,
            random_seed=hparams.random_seed,
            num_buckets=hparams.num_buckets,
            src_max_len=hparams.src_max_len_infer,
            tgt_max_len=hparams.tgt_max_len_infer)

        model = model_creator(
            hparams,
            iterator=iterator,
            mode=tf.contrib.learn.ModeKeys.EVAL,
            source_vocab_table=src_vocab_table,
            target_vocab_table=tgt_vocab_table,
            scope=scope,
            extra_args=extra_args)

    return EvalModel(
        graph=graph,
        model=model,
        src_file_placeholder=src_file_placeholder,
        tgt_file_placeholder=tgt_file_placeholder,
        iterator=iterator)


def create_infer_model(model_creator, hparams, scope=None):
    """Create inference model."""
    graph = tf.Graph()
    src_vocab_file = hparams.src_vocab_file
    tgt_vocab_file = hparams.tgt_vocab_file

    with graph.as_default(), tf.container(scope or "infer"):
        src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
            src_vocab_file, tgt_vocab_file, hparams.share_vocab)
        reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
            tgt_vocab_file, default_value=vocab_utils.UNK)

        batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
        src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        src_dataset = tf.data.Dataset.from_tensor_slices(src_placeholder)

        iterator = iterator_utils.get_infer_iterator(
            src_dataset,
            src_vocab_table,
            batch_size=batch_size_placeholder,
            eos=hparams.eos,
            source_reverse=hparams.source_reverse,
            src_max_len=hparams.src_max_len_infer)

        model = model_creator(
            hparams,
            iterator=iterator,
            mode=tf.contrib.learn.ModeKeys.INFER,
            source_vocab_table=src_vocab_table,
            target_vocab_table=tgt_vocab_table,
            reverse_target_vocab_table=reverse_tgt_vocab_table,
            scope=scope)

    return InferModel(
        graph=graph,
        model=model,
        src_placeholder=src_placeholder,
        batch_size_placeholder=batch_size_placeholder,
        iterator=iterator)
