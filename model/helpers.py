from __future__ import print_function

import collections
import time

import tensorflow as tf

from tensorflow.python.ops import lookup_ops

from .utils import misc_utils as utils

__all__ = [
    "get_initializer", "get_device_str",
    "get_learning_rate_warmup", "get_learning_rate_decay",
    "create_emb_for_encoder_and_decoder", "gradient_clip",
    "create_or_load_model", "load_model", "compute_perplexity"
]


def get_initializer(init_op, seed=None, init_weight=None):
    """Create an initializer. init_weight is only for uniform."""
    if init_op == "uniform":
        assert init_weight
        return tf.random_uniform_initializer(
            -init_weight, init_weight, seed=seed)
    elif init_op == "glorot_normal":
        return tf.keras.initializers.glorot_normal(
            seed=seed)
    elif init_op == "glorot_uniform":
        return tf.keras.initializers.glorot_uniform(
            seed=seed)
    else:
        raise ValueError("Unknown init_op %s" % init_op)


def get_device_str(device_id, num_gpus):
    """Return a device string for multi-GPU setup."""
    if num_gpus == 0:
        return "/cpu:0"
    device_str_output = "/gpu:%d" % (device_id % num_gpus)
    return device_str_output


def get_learning_rate_warmup(global_step, learning_rate, hparams):
    """Get learning rate warmup."""
    warmup_steps = hparams.warmup_steps
    warmup_scheme = hparams.warmup_scheme

    if warmup_scheme == "t2t":
        # 0.01^(1/warmup_steps): we start with a lr, 100 times smaller
        warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
        inv_decay = warmup_factor ** (
            tf.to_float(warmup_steps - global_step))
    else:
        raise ValueError("Unknown warmup scheme %s" % warmup_scheme)

    return tf.cond(
        global_step < hparams.warmup_steps,
        lambda: inv_decay * learning_rate,
        lambda: learning_rate,
        name="learning_rate_warump_cond")


def get_learning_rate_decay(global_step, learning_rate, hparams):
    """Get learning rate decay."""
    if hparams.learning_rate_decay_scheme in ["luong", "luong10"]:
        start_factor = 2
        start_decay_step = int(hparams.num_train_steps / start_factor)
        decay_factor = 0.5

        # decay 5 times
        if hparams.learning_rate_decay_scheme == "luong":
            decay_steps = int(hparams.num_train_steps / (5 * start_factor))
        # decay 10 times
        elif hparams.learning_rate_decay_scheme == "luong10":
            decay_steps = int(hparams.num_train_steps / (10 * start_factor))
    else:
        start_decay_step = hparams.start_decay_step
        decay_steps = hparams.decay_steps
        decay_factor = hparams.decay_factor

    return tf.cond(
        global_step < start_decay_step,
        lambda: learning_rate,
        lambda: tf.train.exponential_decay(
            learning_rate,
            (global_step - start_decay_step),
            decay_steps, decay_factor, staircase=True),
        name="learning_rate_decay_cond")


def create_emb_for_encoder_and_decoder(share_vocab,
                                       src_vocab_size,
                                       tgt_vocab_size,
                                       src_embed_size,
                                       tgt_embed_size,
                                       dtype=tf.float32,
                                       num_partitions=0,
                                       scope=None):
    """Create embedding matrix for both encoder and decoder.

    Args:
      share_vocab: A boolean. Whether to share embedding matrix for both
        encoder and decoder.
      src_vocab_size: An integer. The source vocab size.
      tgt_vocab_size: An integer. The target vocab size.
      src_embed_size: An integer. The embedding dimension for the encoder's
        embedding.
      tgt_embed_size: An integer. The embedding dimension for the decoder's
        embedding.
      dtype: dtype of the embedding matrix. Default to float32.
      num_partitions: number of partitions used for the embedding vars.
      scope: VariableScope for the created subgraph. Default to "embedding".

    Returns:
      embedding_encoder: Encoder's embedding matrix.
      embedding_decoder: Decoder's embedding matrix.

    Raises:
      ValueError: if use share_vocab but source and target have different vocab
        size.
    """

    if num_partitions <= 1:
        partitioner = None
    else:
        partitioner = tf.fixed_size_partitioner(num_partitions)

    with tf.variable_scope(
            scope or "embeddings", dtype=dtype, partitioner=partitioner) as scope:
        # Share embedding
        if share_vocab:
            if src_vocab_size != tgt_vocab_size:
                raise ValueError("Share embedding but different src/tgt vocab sizes"
                                 " %d vs. %d" % (src_vocab_size, tgt_vocab_size))
            print("# Use the same source embeddings for target")
            embedding = tf.get_variable(
                "embedding_share", [src_vocab_size, src_embed_size], dtype)
            embedding_encoder = embedding
            embedding_decoder = embedding
        else:
            with tf.variable_scope("encoder", partitioner=partitioner):
                embedding_encoder = tf.get_variable(
                    "embedding_encoder", [src_vocab_size, src_embed_size], dtype)

            with tf.variable_scope("decoder", partitioner=partitioner):
                embedding_decoder = tf.get_variable(
                    "embedding_decoder", [tgt_vocab_size, tgt_embed_size], dtype)

    return embedding_encoder, embedding_decoder


def gradient_clip(gradients, max_gradient_norm):
    """Clipping gradients of a model."""
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(
        gradients, max_gradient_norm)
    gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
    gradient_norm_summary.append(
        tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))
    return clipped_gradients, gradient_norm_summary, gradient_norm


def load_model(model, ckpt, session, name):
    start_time = time.time()
    model.saver.restore(session, ckpt)
    session.run(tf.tables_initializer())
    print("  loaded %s model parameters from %s, time %.2fs" %
          (name, ckpt, time.time() - start_time))
    return model


def create_or_load_model(model, model_dir, session, name):
    """Create translation model and initialize or load parameters in session."""
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt:
        model = load_model(model, latest_ckpt, session, name)
    else:
        start_time = time.time()
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print("  created %s model with fresh parameters, time %.2fs" %
              (name, time.time() - start_time))

    global_step = model.global_step.eval(session=session)
    return model, global_step


def compute_perplexity(model, sess, name):
    """Compute perplexity of the output of the model."""
    total_loss = 0
    total_predict_count = 0
    start_time = time.time()

    while True:
        try:
            loss, predict_count, batch_size = model.eval(sess)
            total_loss += loss * batch_size
            total_predict_count += predict_count
        except tf.errors.OutOfRangeError:
            break

    perplexity = utils.safe_exp(total_loss / total_predict_count)
    print("  eval %s: perplexity %.2f" % (name, perplexity))
    return perplexity
