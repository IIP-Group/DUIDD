
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Layers for channel decoding and utility functions."""

## Modified by R. Wiesmayr in October 2022:
# Extended the Sionna implementation of the LDPC BP decoder by message damping

import tensorflow as tf
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.utils import llr2mi

from source.decoder_v1 import LDPC5GDecoder1


class dampenedLDPC5GDecoder(LDPC5GDecoder1):
    # pylint: disable=line-too-long
    r"""LDPC5GDecoder(encoder, trainable=False, cn_type='boxplus-phi', hard_out=True, track_exit=False,
    return_infobits=True, prune_pcm=True, num_iter=20, stateful=False, output_dtype=tf.float32, **kwargs)

    (Iterative) belief propagation decoder for 5G NR LDPC codes.

    Inherits from :class:`~sionna.fec.ldpc.decoding.LDPCBPDecoder` and provides
    a wrapper for 5G compatibility, i.e., automatically handles puncturing and
    shortening according to [3GPPTS38212_LDPC]_.

    Note that for full 5G 3GPP NR compatibility, the correct puncturing and
    shortening patterns must be applied and, thus, the encoder object is
    required as input.

    If required the decoder can be made trainable and is differentiable
    (the training of some check node types may be not supported) following the
    concept of "weighted BP" [Nachmani]_.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        encoder: LDPC5GEncoder
            An instance of :class:`~sionna.fec.ldpc.encoding.LDPC5GEncoder`
            containing the correct code parameters.

        trainable: bool
            Defaults to False. If True, every outgoing variable node message is
            scaled with a trainable scalar.

        cn_type: str
            A string defaults to '"boxplus-phi"'. One of
            {`"boxplus"`, `"boxplus-phi"`, `"minsum"`} where
            '"boxplus"' implements the single-parity-check APP decoding rule.
            '"boxplus-phi"' implements the numerical more stable version of
            boxplus [Ryan]_.
            '"minsum"' implements the min-approximation of the CN
            update rule [Ryan]_.

        hard_out: bool
            Defaults to True. If True, the decoder provides hard-decided
            codeword bits instead of soft-values.

        track_exit: bool
            Defaults to False. If True, the decoder tracks EXIT characteristics.
            Note that this requires the all-zero CW as input.

        return_infobits: bool
            Defaults to True. If True, only the `k` info bits (soft or
            hard-decided) are returned. Otherwise all `n` positions are
            returned.

        prune_pcm: bool
            Defaults to True. If True, all punctured degree-1 VNs and
            connected check nodes are removed from the decoding graph (see
            [Cammerer]_ for details). Besides numerical differences, this should
            yield the same decoding result but improved the decoding throughput
            and reduces the memory footprint.

        num_iter: int
            Defining the number of decoder iteration (no early stopping used at
            the moment!).

        stateful: bool
            Defaults to False. If True, the internal VN messages ``msg_vn``
            from the last decoding iteration are returned, and ``msg_vn`` or
            `None` needs to be given as a second input when calling the decoder.
            This is required for iterative demapping and decoding.

        output_dtype: tf.DType
            Defaults to tf.float32. Defines the output datatype of the layer
            (internal precision remains tf.float32).

    Input
    -----
    llrs_ch or (llrs_ch, msg_vn):
        Tensor or Tuple (only required if ``stateful`` is True):

    llrs_ch: [...,n], tf.float32
        2+D tensor containing the channel logits/llr values.

    msg_vn: None or RaggedTensor, tf.float32
        Ragged tensor of VN messages.
        Required only if ``stateful`` is True.

    Output
    ------
        : [...,n] or [...,k], tf.float32
            2+D Tensor of same shape as ``inputs`` containing
            bit-wise soft-estimates (or hard-decided bit-values) of all
            codeword bits. If ``return_infobits`` is True, only the `k`
            information bits are returned.

        : RaggedTensor, tf.float32:
            Tensor of VN messages.
            Returned only if ``stateful`` is set to True.
    Raises
    ------
        ValueError
            If the shape of ``pcm`` is invalid or contains other
            values than `0` or `1`.

        AssertionError
            If ``trainable`` is not `bool`.

        AssertionError
            If ``track_exit`` is not `bool`.

        AssertionError
            If ``hard_out`` is not `bool`.

        AssertionError
            If ``return_infobits`` is not `bool`.

        AssertionError
            If ``encoder`` is not an instance of
            :class:`~sionna.fec.ldpc.encoding.LDPC5GEncoder`.

        ValueError
            If ``output_dtype`` is not {tf.float16, tf.float32, tf.
            float64}.

        ValueError
            If ``inputs`` is not of shape `[batch_size, n]`.

        ValueError
            If ``num_iter`` is not an integer greater (or equal) `0`.

        InvalidArgumentError
            When rank(``inputs``)<2.

    Note
    ----
        As decoding input logits
        :math:`\operatorname{log} \frac{p(x=1)}{p(x=0)}` are assumed for
        compatibility with the learning framework, but
        internally llrs with definition
        :math:`\operatorname{log} \frac{p(x=0)}{p(x=1)}` are used.

        The decoder is not (particularly) optimized for Quasi-cyclic (QC) LDPC
        codes and, thus, supports arbitrary parity-check matrices.

        The decoder is implemented by using '"ragged Tensors"' [TF_ragged]_ to
        account for arbitrary node degrees. To avoid a performance degradation
        caused by a severe indexing overhead, the batch-dimension is shifted to
        the last dimension during decoding.

        If the decoder is made trainable [Nachmani]_, for performance
        improvements only variable to check node messages are scaled as the VN
        operation is linear and, thus, would not increase the expressive power
        of the weights.
    """

    def __init__(self,
                 encoder,
                 trainable=False,
                 cn_type='boxplus-phi',
                 hard_out=True,
                 track_exit=False,
                 return_infobits=True,
                 prune_pcm=True,
                 num_iter=20,
                 stateful=False,
                 output_dtype=tf.float32,
                 alpha0=0,
                 beta0=0,
                 trainDamping=False,
                 constrainAlpha=True,
                 constrainBeta=True,
                 **kwargs):

        super().__init__(encoder,
                 trainable=trainable,
                 cn_type=cn_type,
                 hard_out=hard_out,
                 track_exit=track_exit,
                 return_infobits=return_infobits,
                 prune_pcm=prune_pcm,
                 num_iter=num_iter,
                 stateful=stateful,
                 output_dtype=output_dtype,
                 **kwargs)

        self._trainable = trainDamping or trainable
        # with constraints
        if constrainAlpha:
            self._alpha = tf.Variable(alpha0*tf.ones([num_iter]), dtype=output_dtype, trainable=trainDamping, name="alpha_damping", constraint=lambda x: tf.clip_by_value(x, 0.0,1.0))
        else:
            self._alpha = tf.Variable(alpha0 * tf.ones([num_iter]), dtype=output_dtype, trainable=trainDamping,
                                      name="alpha_damping")
        if constrainBeta:
            self._beta = tf.Variable(beta0*tf.ones([num_iter]), dtype=output_dtype, trainable=trainDamping, name="beta_damping", constraint=lambda x: tf.clip_by_value(x, 0.0,1.0))
        else:
            self._beta = tf.Variable(beta0 * tf.ones([num_iter]), dtype=output_dtype, trainable=trainDamping,
                                     name="beta_damping")


    #########################################
    # Public methods and properties
    #########################################

    @property
    def alpha(self):
        """Alpha values for dampening."""
        return self._alpha

    @property
    def beta(self):
        """Alpha values for dampening."""
        return self._beta


    # def build(self, input_shape):
    #     """Build model."""
    #     if self._stateful:
    #         assert(len(input_shape)==2), \
    #             "For stateful decoding, a tuple of two inputs is expected."
    #         input_shape = input_shape[0]
    #
    #     # check input dimensions for consistency
    #     assert (input_shape[-1]==self.encoder.n), \
    #                             'Last dimension must be of length n.'
    #     assert (len(input_shape)>=2), 'The inputs must have at least rank 2.'
    #
    #     self._old_shape_5g = input_shape
    #     # self._alpha = tf.Variable(self._alpha, dtype=self._output_dtype, trainable=self._trainDamping, name="alpha_damping")


    def super_call(self, inputs):
        """Iterative BP decoding function.

        This function performs ``num_iter`` belief propagation decoding
        iterations and returns the estimated codeword.

        Args:
            inputs (tf.float32): Tensor of shape `[...,n]` containing the
                channel logits/llr values.

        Returns:
            `tf.float32`: Tensor of shape `[...,n]` containing
            bit-wise soft-estimates (or hard-decided bit-values) of all
            codeword bits.

        Raises:
            ValueError: If ``inputs`` is not of shape `[batch_size, n]`.

            InvalidArgumentError: When rank(``inputs``)<2.
        """

        # Extract inputs
        if self._stateful:
            llr_ch, msg_vn = inputs
        else:
            llr_ch = inputs

        tf.debugging.assert_type(llr_ch, self.dtype, 'Invalid input dtype.')

        # internal calculations still in tf.float32
        llr_ch = tf.cast(llr_ch, tf.float32)

        # last dim must be of length n
        tf.debugging.assert_equal(tf.shape(llr_ch)[-1],
                                  self._num_vns,
                                  'Last dimension must be of length n.')

        llr_ch_shape = llr_ch.get_shape().as_list()
        new_shape = [-1, self._num_vns]
        llr_ch_reshaped = tf.reshape(llr_ch, new_shape)

        # must be done during call, as XLA fails otherwise due to ragged
        # indices placed on the CPU device.
        # create permutation index from cn perspective
        self._cn_mask_tf = tf.ragged.constant(self._gen_node_mask(self._cn_con),
                                              row_splits_dtype=tf.int32)

        # batch dimension is last dimension due to ragged tensor representation
        llr_ch = tf.transpose(llr_ch_reshaped, (1,0))

        llr_ch = -1. * llr_ch # logits are converted into "true" llrs

        # init internal decoder state if not explicitly
        # provided (e.g., required to restore decoder state for iterative
        # detection and decoding)
        # load internal state from previous iteration
        # required for iterative det./dec.
        if not self._stateful or msg_vn is None:
            msg_shape = tf.stack([tf.constant(self._num_edges),
                                  tf.shape(llr_ch)[1]],
                                 axis=0)
            msg_vn = tf.zeros(msg_shape, dtype=tf.float32)
        else:
            msg_vn = msg_vn.flat_values

        # track exit decoding trajectory; requires all-zero cw?
        if self._track_exit:
            self._ie_c = tf.zeros(self._num_iter+1)
            self._ie_v = tf.zeros(self._num_iter+1)

        # perform one decoding iteration
        # Remark: msg_vn cannot be ragged as input for tf.while_loop as
        # otherwise XLA will not be supported (with TF 2.5)
        def dec_iter(llr_ch, msg_vn, it):
            it += 1
            # msg_vn_old are the cn2vn messages from the previous iteration
            msg_vn_old = tf.RaggedTensor.from_row_splits(
                        values=msg_vn,
                        row_splits=tf.constant(self._vn_row_splits, tf.int32))
            # variable node update
            # msg_vn are now the vn2cn messages from the vn perspective
            msg_vn = self._vn_update(msg_vn_old, llr_ch)

            # track exit decoding trajectory; requires all-zero cw
            if self._track_exit:
                # neg values as different llr def is expected
                mi = llr2mi(-1. * msg_vn.flat_values)
                self._ie_v = tf.tensor_scatter_nd_add(self._ie_v,
                                                     tf.reshape(it, (1, 1)),
                                                     tf.reshape(mi, (1)))

            # scale outgoing vn messages (weighted BP); only if activated
            if self._has_weights:
                msg_vn = tf.ragged.map_flat_values(self._mult_weights,
                                                   msg_vn)
            # permute edges into CN perspective
            msg_cn = tf.gather(msg_vn.flat_values, self._cn_mask_tf, axis=None)

            # check node update using the pre-defined function
            msg_cn = self._cn_update(msg_cn)

            # track exit decoding trajectory; requires all-zero cw?
            if self._track_exit:
                # neg values as different llr def is expected
                mi = llr2mi(-1.*msg_cn.flat_values)
                # update pos i+1 such that first iter is stored as 0
                self._ie_c = tf.tensor_scatter_nd_add(self._ie_c,
                                                     tf.reshape(it, (1, 1)),
                                                     tf.reshape(mi, (1)))

            # re-permute edges to variable node perspective + damping via vn2cn messages + damping via old and new state (cn2vn messages)
            msg_vn = (1-self.alpha[it-1]-self.beta[it-1])*tf.gather(msg_cn.flat_values, self._ind_cn_inv, axis=None) + \
                     self.alpha[it-1]*msg_vn.flat_values + \
                     self.beta[it-1]*msg_vn_old.flat_values
            return llr_ch, msg_vn, it

        # stopping condition (required for tf.while_loop)
        def dec_stop(llr_ch, msg_vn, it): # pylint: disable=W0613
            return tf.less(it, self._num_iter)

        # start decoding iterations
        it = tf.constant(0)
        # maximum_iterations required for XLA
        _, msg_vn, _ = tf.while_loop(dec_stop,
                                     dec_iter,
                                     (llr_ch, msg_vn, it),
                                     parallel_iterations=1,
                                     maximum_iterations=self._num_iter)


        # raggedTensor for final marginalization
        msg_vn = tf.RaggedTensor.from_row_splits(
                        values=msg_vn,
                        row_splits=tf.constant(self._vn_row_splits, tf.int32))

        # marginalize and remove ragged Tensor
        x_hat = tf.add(llr_ch, tf.reduce_sum(msg_vn, axis=1))

        # restore batch dimension to first dimension
        x_hat = tf.transpose(x_hat, (1,0))

        x_hat = -1. * x_hat # convert llrs back into logits

        if self._hard_out: # hard decide decoder output if required
            x_hat = tf.cast(tf.less(0.0, x_hat), self._output_dtype)

        # Reshape c_short so that it matches the original input dimensions
        output_shape = llr_ch_shape
        output_shape[0] = -1 # overwrite batch dim (can be None in Keras)

        x_reshaped = tf.reshape(x_hat, output_shape)

        # cast output to output_dtype
        x_out = tf.cast(x_reshaped, self._output_dtype)

        if not self._stateful:
            return x_out
        else:
            return x_out, msg_vn

    def call(self, inputs):
        """Iterative BP decoding function.

        This function performs ``num_iter`` belief propagation decoding
        iterations and returns the estimated codeword.

        Args:
            inputs (tf.float32): Tensor of shape `[...,n]` containing the
                channel logits/llr values.

        Returns:
            `tf.float32`: Tensor of shape `[...,n]` or `[...,k]`
            (``return_infobits`` is True) containing bit-wise soft-estimates
            (or hard-decided bit-values) of all codeword bits (or info
            bits, respectively).

        Raises:
            ValueError: If ``inputs`` is not of shape `[batch_size, n]`.

            ValueError: If ``num_iter`` is not an integer greater (or equal)
                `0`.

            InvalidArgumentError: When rank(``inputs``)<2.
        """
        # Modified from sionna code: super().call ==> implements other signature for vn_update function (also takes in iterations count)

        # Extract inputs
        if self._stateful:
            llr_ch, msg_vn = inputs
        else:
            llr_ch = inputs

        tf.debugging.assert_type(llr_ch, self.dtype, 'Invalid input dtype.')

        llr_ch_shape = llr_ch.get_shape().as_list()
        new_shape = [-1, llr_ch_shape[-1]]
        llr_ch_reshaped = tf.reshape(llr_ch, new_shape)
        batch_size = tf.shape(llr_ch_reshaped)[0]

        # invert if rate-matching output interleaver was applied as defined in
        # Sec. 5.4.2.2 in 38.212
        if self._encoder.num_bits_per_symbol is not None:
            llr_ch_reshaped = tf.gather(llr_ch_reshaped,
                                        self._encoder.out_int_inv,
                                        axis=-1)

        # undo puncturing of the first 2*Z bit positions
        llr_5g = tf.concat(
            [tf.zeros([batch_size, 2 * self.encoder.z], self._output_dtype),
             llr_ch_reshaped],
            1)

        # undo puncturing of the last positions
        # total length must be n_ldpc, while llr_ch has length n
        # first 2*z positions are already added
        # -> add n_ldpc - n - 2Z punctured positions
        k_filler = self.encoder.k_ldpc - self.encoder.k  # number of filler bits
        nb_punc_bits = ((self.encoder.n_ldpc - k_filler)
                        - self.encoder.n - 2 * self.encoder.z)

        llr_5g = tf.concat([llr_5g,
                            tf.zeros([batch_size, nb_punc_bits - self._nb_pruned_nodes],
                                     self._output_dtype)],
                           1)

        # undo shortening (= add 0 positions after k bits, i.e. LLR=LLR_max)
        # the first k positions are the systematic bits
        x1 = tf.slice(llr_5g, [0, 0], [batch_size, self.encoder.k])

        # parity part
        nb_par_bits = (self.encoder.n_ldpc - k_filler
                       - self.encoder.k - self._nb_pruned_nodes)
        x2 = tf.slice(llr_5g,
                      [0, self.encoder.k],
                      [batch_size, nb_par_bits])

        # negative sign due to logit definition
        z = -self._llr_max * tf.ones([batch_size, k_filler], self._output_dtype)

        llr_5g = tf.concat([x1, z, x2], 1)

        # and execute the decoder (modified super-call because of damping)
        if not self._stateful:
            x_hat = self.super_call(llr_5g)
        else:
            x_hat, msg_vn = self.super_call([llr_5g, msg_vn])

        if self._return_infobits:  # return only info bits
            # reconstruct u_hat # code is systematic
            u_hat = tf.slice(x_hat, [0, 0], [batch_size, self.encoder.k])
            # Reshape u_hat so that it matches the original input dimensions
            output_shape = llr_ch_shape[0:-1] + [self.encoder.k]
            # overwrite first dimension as this could be None (Keras)
            output_shape[0] = -1
            u_reshaped = tf.reshape(u_hat, output_shape)

            # enable other output datatypes than tf.float32
            u_out = tf.cast(u_reshaped, self._output_dtype)

            if not self._stateful:
                return u_out
            else:
                return u_out, msg_vn

        else:  # return all codeword bits
            # the transmitted CW bits are not the same as used during decoding
            # cf. last parts of 5G encoding function

            # remove last dim
            x = tf.reshape(x_hat, [batch_size, self._n_pruned])

            # remove filler bits at pos (k, k_ldpc)
            x_no_filler1 = tf.slice(x, [0, 0], [batch_size, self.encoder.k])

            x_no_filler2 = tf.slice(x,
                                    [0, self.encoder.k_ldpc],
                                    [batch_size,
                                     self._n_pruned - self.encoder.k_ldpc])

            x_no_filler = tf.concat([x_no_filler1, x_no_filler2], 1)

            # shorten the first 2*Z positions and end after n bits
            x_short = tf.slice(x_no_filler,
                               [0, 2 * self.encoder.z],
                               [batch_size, self.encoder.n])

            # if used, apply rate-matching output interleaver again as
            # Sec. 5.4.2.2 in 38.212
            if self._encoder.num_bits_per_symbol is not None:
                x_short = tf.gather(x_short, self._encoder.out_int, axis=-1)

            # Reshape x_short so that it matches the original input dimensions
            # overwrite first dimension as this could be None (Keras)
            llr_ch_shape[0] = -1
            x_short = tf.reshape(x_short, llr_ch_shape)

            # enable other output datatypes than tf.float32
            x_out = tf.cast(x_short, self._output_dtype)

            if not self._stateful:
                return x_out
            else:
                return x_out, msg_vn

class llrTradeOffDampenedLDPC5GDecoder(dampenedLDPC5GDecoder):
    def __init__(self,
                 encoder,
                 trainableWeights=False,
                 cn_type='boxplus-phi',
                 hard_out=True,
                 track_exit=False,
                 return_infobits=True,
                 prune_pcm=True,
                 num_iter=20,
                 stateful=False,
                 output_dtype=tf.float32,
                 alpha0=0,
                 beta0=0,
                 trainDamping=False,
                 constrainAlpha=True,
                 constrainBeta=True,
                 trainLLRTradeOff=False,
                 alpha_llr_0 = 1,
                 beta_llr_0 = 0,
                 **kwargs):

        super().__init__(encoder,
                 trainable=trainableWeights,
                 cn_type=cn_type,
                 hard_out=False,
                 track_exit=track_exit,
                 return_infobits=return_infobits,
                 prune_pcm=prune_pcm,
                 num_iter=num_iter,
                 stateful=stateful,
                 constrainAlpha=constrainAlpha,
                 constrainBeta=constrainBeta,
                 output_dtype=output_dtype,
                 trainDamping=trainDamping,
                 alpha0=alpha0, beta0=beta0,
                 **kwargs)

        self._hard_out_ = hard_out
        self._trainLlrTradeOff = trainLLRTradeOff
        self._trainable = trainDamping or trainableWeights or trainLLRTradeOff
        self._alpha_llr = tf.Variable(alpha_llr_0, dtype=output_dtype, trainable=trainLLRTradeOff, name="alpha_llr_tradeoff")
        self._beta_llr = tf.Variable(beta_llr_0, dtype=output_dtype, trainable=trainLLRTradeOff, name="beta_llr_tradeoff")

    @property
    def alpha_llr(self):
        """Alpha values for dampening."""
        return self._alpha_llr

    @property
    def beta_llr(self):
        """Alpha values for dampening."""
        return self._beta_llr

    def call(self, inputs):
        """Iterative BP decoding function.

        This function performs ``num_iter`` belief propagation decoding
        iterations and returns the estimated codeword.

        Args:
            inputs (tf.float32): Tensor of shape `[...,n]` containing the
                channel logits/llr values.

        Returns:
            `tf.float32`: Tensor of shape `[...,n]` or `[...,k]`
            (``return_infobits`` is True) containing bit-wise soft-estimates
            (or hard-decided bit-values) of all codeword bits (or info
            bits, respectively).

        Raises:
            ValueError: If ``inputs`` is not of shape `[batch_size, n]`.

            ValueError: If ``num_iter`` is not an integer greater (or equal)
                `0`.

            InvalidArgumentError: When rank(``inputs``)<2.
        """
        # Modified from sionna code: super().call ==> implements other signature for vn_update function (also takes in iterations count)

        # Extract inputs
        if self._stateful:
            llr_ch, msg_vn = inputs
        else:
            llr_ch = inputs

        tf.debugging.assert_type(llr_ch, self.dtype, 'Invalid input dtype.')

        llr_ch_shape = llr_ch.get_shape().as_list()
        new_shape = [-1, llr_ch_shape[-1]]
        llr_ch_reshaped = tf.reshape(llr_ch, new_shape)
        batch_size = tf.shape(llr_ch_reshaped)[0]

        # invert if rate-matching output interleaver was applied as defined in
        # Sec. 5.4.2.2 in 38.212
        if self._encoder.num_bits_per_symbol is not None:
            llr_ch_reshaped = tf.gather(llr_ch_reshaped,
                                        self._encoder.out_int_inv,
                                        axis=-1)

        # undo puncturing of the first 2*Z bit positions
        llr_5g = tf.concat(
            [tf.zeros([batch_size, 2 * self.encoder.z], self._output_dtype),
             llr_ch_reshaped],
            1)

        # undo puncturing of the last positions
        # total length must be n_ldpc, while llr_ch has length n
        # first 2*z positions are already added
        # -> add n_ldpc - n - 2Z punctured positions
        k_filler = self.encoder.k_ldpc - self.encoder.k  # number of filler bits
        nb_punc_bits = ((self.encoder.n_ldpc - k_filler)
                        - self.encoder.n - 2 * self.encoder.z)

        llr_5g = tf.concat([llr_5g,
                            tf.zeros([batch_size, nb_punc_bits - self._nb_pruned_nodes],
                                     self._output_dtype)],
                           1)

        # undo shortening (= add 0 positions after k bits, i.e. LLR=LLR_max)
        # the first k positions are the systematic bits
        x1 = tf.slice(llr_5g, [0, 0], [batch_size, self.encoder.k])

        # parity part
        nb_par_bits = (self.encoder.n_ldpc - k_filler
                       - self.encoder.k - self._nb_pruned_nodes)
        x2 = tf.slice(llr_5g,
                      [0, self.encoder.k],
                      [batch_size, nb_par_bits])

        # negative sign due to logit definition
        z = -self._llr_max * tf.ones([batch_size, k_filler], self._output_dtype)

        llr_5g = tf.concat([x1, z, x2], 1)

        # and execute the decoder
        # and execute the decoder (modified super-call because of damping)
        if not self._stateful:
            x_hat = self.super_call(llr_5g)
        else:
            x_hat, msg_vn = self.super_call([llr_5g, msg_vn])
        # Intrinsic/Extrinsic LLR trade-off
        x_hat = self._alpha_llr * x_hat - self._beta_llr * llr_5g

        if self._hard_out_:  # hard decide decoder output if required
            x_hat = tf.cast(tf.less(0.0, x_hat), self._output_dtype)

        if self._return_infobits:  # return only info bits
            # reconstruct u_hat # code is systematic
            u_hat = tf.slice(x_hat, [0, 0], [batch_size, self.encoder.k])
            # Reshape u_hat so that it matches the original input dimensions
            output_shape = llr_ch_shape[0:-1] + [self.encoder.k]
            # overwrite first dimension as this could be None (Keras)
            output_shape[0] = -1
            u_reshaped = tf.reshape(u_hat, output_shape)

            # enable other output datatypes than tf.float32
            u_out = tf.cast(u_reshaped, self._output_dtype)

            if not self._stateful:
                return u_out
            else:
                return u_out, msg_vn

        else:  # return all codeword bits
            # the transmitted CW bits are not the same as used during decoding
            # cf. last parts of 5G encoding function

            # remove last dim
            x = tf.reshape(x_hat, [batch_size, self._n_pruned])

            # remove filler bits at pos (k, k_ldpc)
            x_no_filler1 = tf.slice(x, [0, 0], [batch_size, self.encoder.k])

            x_no_filler2 = tf.slice(x,
                                    [0, self.encoder.k_ldpc],
                                    [batch_size,
                                     self._n_pruned - self.encoder.k_ldpc])

            x_no_filler = tf.concat([x_no_filler1, x_no_filler2], 1)

            # shorten the first 2*Z positions and end after n bits
            x_short = tf.slice(x_no_filler,
                               [0, 2 * self.encoder.z],
                               [batch_size, self.encoder.n])

            # if used, apply rate-matching output interleaver again as
            # Sec. 5.4.2.2 in 38.212
            if self._encoder.num_bits_per_symbol is not None:
                x_short = tf.gather(x_short, self._encoder.out_int, axis=-1)

            # Reshape x_short so that it matches the original input dimensions
            # overwrite first dimension as this could be None (Keras)
            llr_ch_shape[0] = -1
            x_short = tf.reshape(x_short, llr_ch_shape)

            # enable other output datatypes than tf.float32
            x_out = tf.cast(x_short, self._output_dtype)

            if not self._stateful:
                return x_out
            else:
                return x_out, msg_vn