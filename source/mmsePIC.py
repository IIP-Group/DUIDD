# -----------------------------------------------------
# -- Implementation of the MMSE PIC and LoCo PIC Data Detectors
# --
# -- The implementation of the MMSE PIC is based on
# -- C. Studer, S. Fateh, and D. Seethaler, “ASIC Implementation of Soft-Input Soft-Output MIMO Detection Using MMSE
# -- Parallel Interference Cancellation,” IEEE Journal of Solid-State Circuits, vol. 46, no. 7, pp. 1754–1765, July 2011,
# -- available https://www.csl.cornell.edu/~studer/papers/11JSSC-mmsepic.pdf
# --
# -- October 2022 (c) Reinhard Wiesmayr (wiesmayr@iis.ee.ethz.ch)
# -- The code is based on the Sionna implementation of the LMMSE detector.
# -----------------------------------------------------

import platform
from tensorflow.keras.layers import Layer
import sionna
from sionna.mapping import *
from sionna.ofdm import RemoveNulledSubcarriers
from sionna.utils import split_dim, flatten_dims, expand_to_rank, flatten_last_dims, matrix_inv, matrix_sqrt_inv
import numpy as np


def selectDataCarryingOFDMSymbols(data_vec, rg_dim, data_ind, num_ofdm_data_symbols, num_effective_subcarriers):
    data_vec = flatten_dims(data_vec, 2, rg_dim)
    # data_ind carries indices for all data streams, we assume that they are all the same and only select the first one
    data_vec = tf.gather(data_vec, data_ind, axis=2)
    return split_dim(data_vec, [num_ofdm_data_symbols, num_effective_subcarriers], axis=rg_dim)


class SisoMmsePicDetector(Layer):
    # pylint: disable=line-too-long
    """
    Soft-Input Soft-Output Minimum Mean Squared Error (MMSE) Parallel Interference Cancellation Detector, based on
    C. Studer, S. Fateh, and D. Seethaler, “ASIC Implementation of Soft-Input Soft-Output MIMO Detection Using MMSE
    Parallel Interference Cancellation,” IEEE Journal of Solid-State Circuits, vol. 46, no. 7, pp. 1754–1765, July 2011,
    available at https://www.csl.cornell.edu/~studer/papers/11JSSC-mmsepic.pdf

    This implementation does NOT support XLA mode. Refer to the latest Sionna release for an implementation of MMSE PIC
    that supports XLA.
    """
    def __init__(self,
                 resource_grid,
                 stream_management,
                 demapping_method,
                 constellation: sionna.mapping.Constellation,
                 dtype=tf.complex64, low_complexity=False,
                 regularizationEpsilon=1e-4, data_carrying_whitened_inputs = False,
                 hyper_parameter_err_var_num_ofdm_symbols = -1, training=False,
                 **kwargs):
        super().__init__(**kwargs)
        assert isinstance(resource_grid, sionna.ofdm.ResourceGrid)
        assert isinstance(stream_management, sionna.mimo.StreamManagement)
        self._resource_grid = resource_grid
        self._stream_management = stream_management
        self._removed_nulled_scs = RemoveNulledSubcarriers(self._resource_grid)
        self._constellation = constellation
        self._dtype = dtype
        self._epsilon = regularizationEpsilon
        self._low_complexity = low_complexity
        self._removed_nulled_scs = RemoveNulledSubcarriers(self._resource_grid)
        self._data_carrying_whitened_inputs = data_carrying_whitened_inputs

        # Precompute indices to extract data symbols
        mask = resource_grid.pilot_pattern.mask
        num_data_symbols = resource_grid.pilot_pattern.num_data_symbols
        data_ind = tf.argsort(flatten_last_dims(mask), direction="ASCENDING")
        self._data_ind = data_ind[..., :num_data_symbols]

        num_bits_per_symbol = self._constellation.num_bits_per_symbol
        num_points = int(2 ** num_bits_per_symbol)
        a = np.zeros([num_points, num_bits_per_symbol])
        for i in range(0, num_points):
            a[i, :] = np.array(list(np.binary_repr(i, num_bits_per_symbol)),
                               dtype=np.int16)

        self._a = a
        self._aBool = tf.cast(self._a, tf.bool)

        # Compute symbol indices for which the bits are 0 or 1
        c0 = np.zeros([int(num_points / 2), num_bits_per_symbol])
        c1 = np.zeros([int(num_points / 2), num_bits_per_symbol])
        for i in range(num_bits_per_symbol - 1, -1, -1):
            c0[:, i] = np.where(a[:, i] == 0)[0]
            c1[:, i] = np.where(a[:, i] == 1)[0]
        self._c0 = tf.constant(c0, dtype=tf.int32)  # Symbols with ith bit=0
        self._c1 = tf.constant(c1, dtype=tf.int32)  # Symbols with ith bit=1

        if constellation.normalize:
            n = int(num_bits_per_symbol / 2)
            qam_var = 1 / (2 ** (n - 2)) * np.sum(np.linspace(1, 2 ** n - 1, 2 ** (n - 1)) ** 2)
            self._qam_normalization_factor = 1 / np.sqrt(qam_var)

        else:
            self._qam_normalization_factor = 1

        if demapping_method == "app":
            self._reduce = tf.reduce_logsumexp
        else:
            self._reduce = tf.reduce_max

        if hyper_parameter_err_var_num_ofdm_symbols > 0:
            self._eta = tf.Variable(tf.ones([1, 1, hyper_parameter_err_var_num_ofdm_symbols, 1, 1, 1]), trainable=training, dtype=tf.float32, name="eta")
        else:
            self._eta = 1

    def soft_symbols(self, llr_a, points_reshaped, batch_size, num_ofdm_symbols, num_effective_subcarriers, num_tx,
                     num_streams):

        p0 = 0.5 * (1 - tf.math.tanh(
            0.5 * llr_a))

        if self._low_complexity and self._constellation._constellation_type == "qam" and self._constellation.num_bits_per_symbol in [
            1, 2, 4, 6]:
            p1 = 1 - p0
            if self._constellation.num_bits_per_symbol == 1:
                # BPSK
                s_real = (1 - 2 * tf.gather(p1, indices=0, axis=-1))
                s_imag = 0

                c = 1
                d = 0
            elif self._constellation.num_bits_per_symbol == 2:
                # QPSK
                s_real = (1 - 2 * tf.gather(p1, indices=0, axis=-1))
                s_imag = (1 - 2 * tf.gather(p1, indices=1, axis=-1))

                c = 2
                d = 0
            elif self._constellation.num_bits_per_symbol == 4:
                # 16-QAM
                s_real = (1 - 2 * tf.gather(p1, indices=0, axis=-1)) * (1 + 2 * tf.gather(p1, indices=2, axis=-1))
                s_imag = (1 - 2 * tf.gather(p1, indices=1, axis=-1)) * (1 + 2 * tf.gather(p1, indices=3, axis=-1))

                c = 1 + 8 * tf.gather(p1, indices=2, axis=-1)
                d = 1 + 8 * tf.gather(p1, indices=3, axis=-1)
            elif self._constellation.num_bits_per_symbol == 6:
                # 64-QAM
                raise Exception('constellation order not implemented')
            else:
                raise Exception('unsupported constellation order')

            s_hat = self._qam_normalization_factor * tf.complex(s_real, s_imag)
            # normalization can be included in previous scaling factor...
            error_var = self._qam_normalization_factor ** 2 * ((c + d) - tf.square(s_real) - tf.square(s_imag))

            log_P_C = None
        else:
            p0 = tf.expand_dims(p0, axis=-2)
            p1 = 1 - p0
            oneBits_reshaped = tf.reshape(self._aBool, [1, 1, 1, 1, 1] + self._constellation.points.shape +
                                          self._constellation.num_bits_per_symbol)
            pC_bits = tf.where(oneBits_reshaped, p1, p0)

            P_C = tf.reduce_prod(pC_bits, axis=-1)

            # numerically stable way to calculate log_pC (log of constellation symbol probabilities)
            # following (22), (23) from C. Studer et al., "Soft–Input Soft–Output Single Tree-Search
            # Sphere Decoding," IEEE TRANS. ON INFORMATION THEORY, VOL. 56, NO. 10, OCTOBER 2010
            abs_llrs = tf.math.abs(llr_a)
            K_i_tilde = tf.reduce_sum(0.5 * abs_llrs + tf.math.log(1 + tf.math.exp(-abs_llrs)), axis=-1,
                                      keepdims=True)  # @TODO: check axis right?

            x_ib = 2 * (tf.cast(oneBits_reshaped, dtype=tf.float32) - 0.5)
            log_P_C = - (K_i_tilde - tf.reduce_sum(0.5 * x_ib * tf.expand_dims(llr_a, axis=-2), axis=-1))

            # s_hat [batch_size, num_tx, num_streams, num_ofdm_symbols, num_effective_subcarriers]
            s_hat = tf.reduce_sum(points_reshaped * tf.cast(P_C, tf.complex64), axis=-1)

            # Calculate Error Variance Estimate
            squared_error = tf.math.pow(
                tf.maximum(tf.abs(tf.expand_dims(s_hat, axis=-1) - points_reshaped), self._epsilon), 2)
            error_var = tf.reduce_sum(squared_error * P_C, axis=-1)

        # transform s_hat and error_var to [batch_size, 1, num_ofdm_symbols, num_effective_subcarriers,
        # num_tx*num_streams, 1]
        s_hat = tf.transpose(s_hat, [0, 3, 4, 1, 2])
        error_var = tf.transpose(error_var, [0, 3, 4, 1, 2])
        s_int_shape = tf.concat(
            [[batch_size], [1], [num_ofdm_symbols], [num_effective_subcarriers], [num_tx * num_streams, 1]], 0)
        s_hat = tf.reshape(s_hat, s_int_shape)
        error_var = tf.reshape(error_var, s_int_shape)

        return [s_hat, error_var, log_P_C]

    def LLLCalculation(self, z_i, rho_i, points_reshaped, log_P_C):
        # z_i is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx, 1]
        if self._low_complexity and self._constellation._constellation_type == "qam" and self._constellation.num_bits_per_symbol in [
            1, 2, 4, 6]:
            # transform z_i to constellation w/o unit-energy scaling
            z_i = z_i / self._qam_normalization_factor

            if self._constellation.num_bits_per_symbol == 1:
                # BPSK
                lambda_b_1 = 4 * tf.math.real(z_i)
                lambda_b = lambda_b_1
            elif self._constellation.num_bits_per_symbol == 2:
                # QPSK
                lambda_b_1 = 4 * tf.math.real(z_i)
                lambda_b_2 = 4 * tf.math.imag(z_i)
                lambda_b = tf.concat([lambda_b_1, lambda_b_2], axis=-1)
            elif self._constellation.num_bits_per_symbol == 4:
                # 16-QAM
                z_i_real = tf.math.real(z_i)
                z_i_imag = tf.math.imag(z_i)
                lambda_b_1 = tf.where(tf.math.less_equal(tf.abs(z_i_real), 2), 4 * z_i_real,
                                      8 * z_i_real - 8 * tf.sign(z_i_real))
                lambda_b_2 = 8 - 4 * tf.abs(z_i_real)
                lambda_b_3 = tf.where(tf.math.less_equal(tf.abs(z_i_imag), 2), 4 * z_i_imag,
                                      8 * z_i_imag - 8 * tf.sign(z_i_imag))
                lambda_b_4 = 8 - 4 * tf.abs(z_i_imag)
                lambda_b = tf.concat([lambda_b_1, lambda_b_3, lambda_b_2, lambda_b_4], axis=-1)
            elif self._constellation.num_bits_per_symbol == 6:
                # 64-QAM
                raise Exception('constellation order not implemented')
            else:
                raise Exception('unsupported constellation order')

            lambda_b = self._qam_normalization_factor ** 2 * lambda_b
            llr_d = - rho_i * lambda_b  # minus because of inverse LLR definition
        else:
            squared_dist = tf.math.pow(tf.math.abs(z_i - points_reshaped), 2)

            squared_dist = tf.maximum(squared_dist, self._epsilon ** 2)

            if log_P_C is not None:
                exponents = -squared_dist * rho_i + log_P_C  # intrinsic
            else:
                exponents = -squared_dist * rho_i  # extrinsic

            exp0 = tf.gather(exponents, self._c0, axis=-1, batch_dims=0)
            exp1 = tf.gather(exponents, self._c1, axis=-1, batch_dims=0)

            # transform
            # llr_d is [batch_size, 1, num_ofdm_symbols, num_effective_subcarriers, num_tx*num_streams, num_bits_per_symbol]
            llr_d = self._reduce(exp1, axis=-2) - self._reduce(exp0, axis=-2)

        return llr_d

    def call(self, inputs):
        y, h_hat, err_var, no, llr_a, G = inputs
        # y has shape:
        # [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]

        # h_hat has shape:
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams,...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]

        # err_var has a shape that is broadcastable to h_hat

        # llr_a None | [batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol], tf.float

        # no has shape [batch_size, num_rx] => assumed constant noise var across all Rx Antennas
        # or just the first n dimensions of this

        # prepare variables for shape
        batch_size = tf.shape(y)[0]
        num_effective_subcarriers = self._resource_grid.num_effective_subcarriers
        num_ofdm_data_symbols = int(self._resource_grid.num_data_symbols / num_effective_subcarriers)
        num_bits_per_symbol = self._constellation.num_bits_per_symbol
        num_tx = self._resource_grid.num_tx
        num_points = int(self._constellation.points.shape[0])
        num_streams = self._resource_grid.num_streams_per_tx
        num_data_symbols = int(self._resource_grid.num_data_symbols)
        _type_float = tf.float32
        data_ind = self._data_ind[0, 0, :]

        if not self._data_carrying_whitened_inputs:
            # Remove nulled subcarriers from y (guards, dc). New shape:
            # [batch_size, num_rx, num_rx_ant, ...
            #  ..., num_ofdm_symbols, num_effective_subcarriers]
            y_eff = self._removed_nulled_scs(y)
            ####################################################
            ### Prepare the observation y for MIMO detection ###
            ####################################################
            # Transpose y_eff to put num_rx_ant last. New shape:
            # [batch_size, num_rx, num_ofdm_symbols,...
            #  ..., num_effective_subcarriers, num_rx_ant]
            y_dt = tf.transpose(y_eff, [0, 1, 3, 4, 2])
            y_dt = tf.cast(y_dt, self._dtype)

            # Gather only data-carrying symbols
            # New shape:
            # [batch_size, num_rx, num_ofdm_data_symbols,...
            #  ..., num_effective_subcarriers, num_rx_ant]
            y_dt = selectDataCarryingOFDMSymbols(y_dt, 2, data_ind, num_ofdm_data_symbols, num_effective_subcarriers)

            ##############################################
            ### Prepare the err_var for MIMO detection ###
            ##############################################
            # New shape is:
            # [batch_size, num_rx, num_ofdm_data_symbols,...
            #  ..., num_effective_subcarriers, num_rx_ant, num_tx*num_streams]
            err_var_dt = tf.broadcast_to(err_var, tf.shape(h_hat))
            err_var_dt = tf.transpose(err_var_dt, [0, 1, 5, 6, 2, 3, 4])
            err_var_dt = flatten_last_dims(err_var_dt, 2)
            err_var_dt = tf.cast(err_var_dt, self._dtype)
            err_var_dt = selectDataCarryingOFDMSymbols(err_var_dt, 2, data_ind, num_ofdm_data_symbols,
                                                       num_effective_subcarriers)
            err_var_dt = err_var_dt * tf.cast(self._eta, dtype=tf.complex64)

            ###############################
            ### Construct MIMO channels ###
            ###############################

            # Reshape h_hat for the construction of desired/interfering channels:
            # [num_rx, num_tx, num_streams_per_tx, batch_size, num_rx_ant, ,...
            #  ..., num_ofdm_symbols, num_effective_subcarriers]
            perm = [1, 3, 4, 0, 2, 5, 6]
            h_dt = tf.transpose(h_hat, perm)

            # Flatten first three dimensions:
            # [num_rx*num_tx*num_streams_per_tx, batch_size, num_rx_ant, ...
            #  ..., num_ofdm_symbols, num_effective_subcarriers]
            h_dt = flatten_dims(h_dt, 3, 0)

            # Gather desired and undesired channels
            ind_desired = self._stream_management.detection_desired_ind
            ind_undesired = self._stream_management.detection_undesired_ind
            h_dt_desired = tf.gather(h_dt, ind_desired, axis=0)
            h_dt_undesired = tf.gather(h_dt, ind_undesired, axis=0)

            # Split first dimension to separate RX and TX:
            # [num_rx, num_streams_per_rx, batch_size, num_rx_ant, ...
            #  ..., num_ofdm_symbols, num_effective_subcarriers]
            h_dt_desired = split_dim(h_dt_desired, [self._stream_management.num_rx, -1], 0)
            h_dt_undesired = split_dim(h_dt_undesired, [self._stream_management.num_rx, -1], 0)

            # Permutate dims to
            # [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,..
            #  ..., num_rx_ant, num_streams_per_rx(num_Interfering_streams_per_rx)]
            perm = [2, 0, 4, 5, 3, 1]
            h_dt_desired = tf.transpose(h_dt_desired, perm)
            h_dt_desired = tf.cast(h_dt_desired, self._dtype)
            h_dt_undesired = tf.transpose(h_dt_undesired, perm)
            h_dt_desired = selectDataCarryingOFDMSymbols(h_dt_desired, 2, data_ind, num_ofdm_data_symbols,
                                                         num_effective_subcarriers)
            h_dt_undesired = selectDataCarryingOFDMSymbols(h_dt_undesired, 2, data_ind, num_ofdm_data_symbols,
                                                           num_effective_subcarriers)

            ##################################
            ### Prepare the noise variance ###
            ##################################
            # no is first broadcast to [batch_size, num_rx, num_rx_ant]
            # then the rank is expanded to that of y
            # then it is transposed like y to the final shape
            # [batch_size, num_rx, num_ofdm_symbols,...
            #  ..., num_effective_subcarriers, num_rx_ant]
            no_dt = expand_to_rank(no, 3, -1)
            no_dt = tf.broadcast_to(no_dt, tf.shape(y)[:3])
            no_dt = expand_to_rank(no_dt, tf.rank(y), -1)
            no_dt = tf.transpose(no_dt, [0, 1, 3, 4, 2])
            no_dt = tf.cast(no_dt, self._dtype)

            ##################################################
            ### Compute the interference covariance matrix ###
            ##################################################
            # Covariance of undesired transmitters
            s_inf = tf.matmul(h_dt_undesired, h_dt_undesired, adjoint_b=True)

            # Thermal noise
            s_no = tf.linalg.diag(no_dt)

            # Channel estimation errors
            # As we have only error variance information for each element,
            # we simply sum them across transmitters and build a
            # diagonal covariance matrix from this
            s_csi = tf.linalg.diag(tf.reduce_sum(err_var_dt, -1))

            # Final covariance matrix
            s = s_inf + s_no + s_csi
            s = tf.cast(s, self._dtype)

            # Noise+Interference Whitening
            s_inv_1_2 = matrix_sqrt_inv(s)

            # Whiten the observation
            y_dt = tf.expand_dims(y_dt, -1)
            y_dt_whitened = tf.matmul(s_inv_1_2, y_dt)

            # Compute channel after whitening
            h_dt_desired_whitened = tf.matmul(s_inv_1_2, h_dt_desired)

            # Step 1: Compute Gram matrix
            # h_dt_desired is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,..
            #  ..., num_rx_ant, num_streams_per_rx(num_Interfering_streams_per_rx)]
            # G is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx, num_streams_per_rx]
        else:
            h_dt_desired_whitened = h_hat
            y_dt_whitened = y
        if G is None:
            G = tf.linalg.matmul(h_dt_desired_whitened, h_dt_desired_whitened, adjoint_a=True)

            # y_MF is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx]
            y_MF = tf.linalg.matmul(h_dt_desired_whitened, y_dt_whitened, adjoint_a=True)
        else:
            y_MF = y
        ############################################################
        #### SISO LMMSE PIC ###
        # following Algorithm 1 from [1]
        ############################################################

        # Calculate Soft Symbols
        points_reshaped = tf.reshape(self._constellation.points, [1] * 5 + [num_points])

        if llr_a is None:
            # no a priori LLR => no parallel interference cancellation
            y_hat_i_MF = y_MF
            # _lambda = None
            _error_var_row_vec = None
            log_P_C = None
            error_var = 1
            llr_a_out = 0
            _A = G
        else:
            # Step 2: Calculte soft-symbols and variances

            # llr_a is [batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol]
            # reshape to [batch_size, num_tx, num_streams, num_ofdm_symbols, num_effective_subcarriers, num_bits_per_symbol]
            llr_a_out = llr_a
            llr_a = tf.expand_dims(llr_a, axis=-1)
            llr_a = tf.expand_dims(llr_a, axis=-3)
            llr_int_shape = tf.concat(
                [tf.shape(llr_a)[:-3], [num_ofdm_data_symbols, num_effective_subcarriers, num_bits_per_symbol]], 0)
            llr_a = tf.reshape(llr_a, llr_int_shape)

            # Compute log(P(points)) from llr_a
            # [batch_size, num_tx, num_streams, num_ofdm_symbols, num_effective_subcarriers, num_constellation]
            # logPb0 = np.log(1 + np.exp(llr_a))  # numerical instability exp(large) is inf => Jacobi Logarithm

            [s_hat, error_var, log_P_C] = self.soft_symbols(llr_a, points_reshaped, batch_size, num_ofdm_data_symbols,
                                                            num_effective_subcarriers, num_tx, num_streams)

            # Step 3: Perform PIC
            # H^H y_hat_i = y_MF - sum_j!=i gj s_hat_j = y + g_i s_hat_i - sum_j g_j s_hat_j
            # y_MF is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx]
            # G is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx, num_streams_per_rx]
            _g_j_s_hat_j = tf.linalg.matmul(G, s_hat)
            _s_hat = tf.transpose(s_hat, [0, 1, 2, 3, 5, 4])
            y_hat_i_MF = y_MF + G * _s_hat - _g_j_s_hat_j
            # y_hat_i_MF is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_tx*num_streams,
            # num_tx*num_streams]

            # Step 4: Compute A
            # Calculate MMSE Filter (efficiently)
            # W^H = A^-1 H^H
            # A = H^H H \Lambda + N_0 I_Mt
            # \Lambda_ii = E_i = error_var

            _error_var_row_vec = tf.linalg.matrix_transpose(error_var)
            # _lambda = tf.linalg.diag(tf.squeeze(error_var, axis=-1))
            # _lambda is [batch_size, 1, num_ofdm_symbols, num_effective_subcarriers, num_tx*num_streams, num_tx*num_streams]
            # _A = tf.matmul(G, tf.cast(_lambda, dtype=self.dtype))
            _A = G * tf.cast(_error_var_row_vec, dtype=self.dtype)

        # calculate LMMSE filter (unit power Tx signals, perfect CSI)
        # _A is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx, num_streams_per_rx]
        # _I_NT is [1, 1, 1, 1, num_streams_per_rx, num_streams_per_rx]
        _I_NT = tf.linalg.eye(tf.shape(_A)[-1], dtype=self.dtype)
        _I_NT = tf.reshape(_I_NT, tf.concat([[1] * (_A._rank() - 2), tf.shape(_I_NT)], 0))
        # thermal noise is identity after noise whitening
        _A = _A + _I_NT

        # Step 5: compute MMSE filter and outputs, calculate A\H^H
        # A_inv is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx, num_streams_per_rx]
        # calculating inverse explicitly is necessary
        A_inv = tf.linalg.inv(_A)
        # A_inv_Hermitian = tf.transpose(A_inv, conjugate=True, perm=[0, 1, 2, 3, 5, 4])

        # G and [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx, num_streams_per_rx]
        # mu_i = a_i^H g_i
        _G_trans = tf.linalg.matrix_transpose(G)
        mu_i = tf.math.real(tf.reduce_sum(A_inv * _G_trans, axis=-1, keepdims=True))
        # mu_i is [batch_size, 1, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx, 1]

        rho_i = tf.divide(mu_i, tf.maximum(1 - error_var * mu_i, self._epsilon))
        # z_i = W^H y_dt = mu_i^-1 a_i^H y_hat_i_MF
        # y_hat_i_MF is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_tx*num_streams,
        # num_tx*num_streams]

        # h_i^H h_i
        # channel_strengths = tf.linalg.diag_part(G)
        # normalization_chan_strength = tf.linalg.diag(1 / channel_strengths)

        if llr_a is not None:
            # z_i = tf.linalg.matmul(A_inv / tf.cast(mu_i, dtype=self.dtype), y_hat_i_MF)
            # z_i = tf.linalg.diag_part(z_i)
            y_hat_i_MF_trans = tf.linalg.matrix_transpose(y_hat_i_MF)
            z_i = tf.squeeze(
                tf.reduce_sum(A_inv * y_hat_i_MF_trans, axis=-1, keepdims=True) / tf.cast(mu_i, dtype=self.dtype),
                axis=-1)

            ### LMMSE calculation done => continue with LLR calculation

            # Step 6: calculate LLRs

            # calculate exponents
            # Compute squared distances from y to all points

            # log_P_C is [batch_size, num_tx, num_streams, num_ofdm_symbols, num_effective_subcarriers,
            # num_constellation] transform log_P_C to [batch_size, 1, num_ofdm_symbols, num_effective_subcarriers,
            # num_tx*num_streams, num_constellation]
            if log_P_C is not None:
                log_P_C = tf.transpose(log_P_C, [0, 3, 4, 1, 2, 5])
                log_P_C_int_shape = tf.concat(
                    [[batch_size], [1], [num_ofdm_data_symbols], [num_effective_subcarriers], [num_tx * num_streams],
                     [num_points]], 0)
                log_P_C = tf.reshape(log_P_C, log_P_C_int_shape)

            z_i = tf.expand_dims(z_i, axis=-1)
        else:
            z_i = tf.linalg.matmul(A_inv, y_hat_i_MF) / tf.cast(mu_i, dtype=self.dtype)
        # z_i is [batch_size, num_rx, num_ofdm_data_symbols, num_effective_subcarriers, num_streams_per_rx, 1]
        llr_d = self.LLLCalculation(z_i, rho_i, points_reshaped, log_P_C)

        # llr_d = tf.reduce_logsumexp(exp1, axis=-2) - tf.reduce_logsumexp(exp0, axis=-2)

        # internal llr_a shape [batch_size, num_tx, num_streams, num_ofdm_symbols, num_effective_subcarriers,
        # num_bits_per_symbol]
        # outer llr_a shape is [batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol]
        # convert llr_d to out-shape
        llr_d = tf.squeeze(llr_d, axis=[1])
        tmp_shape = tf.concat([[batch_size], [num_ofdm_data_symbols], [num_effective_subcarriers], [num_tx],
                               [num_streams], [num_bits_per_symbol]], 0)
        llr_d = tf.reshape(llr_d, tmp_shape)
        llr_d = tf.transpose(llr_d, [0, 3, 4, 1, 2, 5])
        out_shape = tf.concat([[batch_size], [num_tx], [num_streams], [num_data_symbols * num_bits_per_symbol]], 0)
        llr_d = tf.reshape(llr_d, out_shape)

        # subtract llr_a => llr_e = llr_d - llr_a
        if self._low_complexity:
            llr_e = llr_d
        else:
            llr_e = llr_d - llr_a_out

        return [llr_e, y_MF, h_dt_desired_whitened, G]

# SISO LoCo PIC
class sisoLoCoPicDetector(SisoMmsePicDetector):
    # pylint: disable=line-too-long
    """
    Soft-Input Soft-Output Low-Complexity (LoCo) Parallel Interference Cancellation Detector
     with trainable parameter \alpha (and \beta).
    """

    def __init__(self,
                 resource_grid,
                 stream_management,
                 demapping_method,
                 constellation=None,
                 trainable=False,
                 alpha0=1,
                 dtype=tf.complex64,
                 regularizationEpsilon=1e-4,
                 data_carrying_whitened_inputs = False,
                 low_complexity=False,
                 two_variables=False,
                 beta0=0,
                 error_var_term="default"):
        super().__init__(resource_grid=resource_grid, stream_management=stream_management,
                         demapping_method=demapping_method, data_carrying_whitened_inputs=data_carrying_whitened_inputs,
                         constellation=constellation, low_complexity=low_complexity,
                         dtype=dtype)
        assert isinstance(resource_grid, sionna.ofdm.ResourceGrid)
        assert isinstance(stream_management, sionna.mimo.StreamManagement)
        self._resource_grid = resource_grid
        self._stream_management = stream_management
        self._removed_nulled_scs = RemoveNulledSubcarriers(self._resource_grid)

        self._trainable = trainable

        self._two_variables = two_variables

        self._epsilon = regularizationEpsilon

        # Precompute indices to extract data symbols
        mask = resource_grid.pilot_pattern.mask
        num_data_symbols = resource_grid.pilot_pattern.num_data_symbols
        data_ind = tf.argsort(flatten_last_dims(mask), direction="ASCENDING")
        self._data_ind = data_ind[..., :num_data_symbols]

        num_bits_per_symbol = self._constellation.num_bits_per_symbol
        num_points = int(2 ** num_bits_per_symbol)
        a = np.zeros([num_points, num_bits_per_symbol])
        for i in range(0, num_points):
            a[i, :] = np.array(list(np.binary_repr(i, num_bits_per_symbol)),
                               dtype=np.int16)

        self._error_var_term = error_var_term

        self._a = a
        self._aBool = tf.cast(self._a, tf.bool)

        self._alpha0 = alpha0
        self._beta0 = beta0

        self._alpha = tf.Variable(self._alpha0, dtype=tf.as_dtype(self._dtype).real_dtype, trainable=self._trainable,
                                  name="alpha_lmmse")
        if two_variables:
            self._beta = tf.Variable(self._beta0, dtype=tf.as_dtype(self._dtype).real_dtype, trainable=self._trainable,
                                  name="beta_mf")
        else:
            self._beta = beta0

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    def call(self, inputs):
        y, h_hat, err_var, no, llr_a, A_inv, G, mu_i = inputs  # attention: obey right order of input variables!!!
        # y has shape:
        # [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]

        # h_hat has shape:
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams,...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]

        # err_var has a shape that is broadcastable to h_hat

        # llr_a None | [batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol], tf.float

        # no has shape [batch_size, num_rx] => assumed constant noise var across all Rx Antennas
        # or just the first n dimensions of this
        # prepare variables for shape
        batch_size = tf.shape(y)[0]
        num_effective_subcarriers = self._resource_grid.num_effective_subcarriers
        num_ofdm_data_symbols = int(self._resource_grid.num_data_symbols / num_effective_subcarriers)

        num_bits_per_symbol = self._constellation.num_bits_per_symbol
        num_tx = self._resource_grid.num_tx
        num_points = np.cast[int](self._constellation.points.shape)[0]
        num_streams = self._resource_grid.num_streams_per_tx
        num_data_symbols = int(self._resource_grid.num_data_symbols)
        _type_float = tf.float32
        data_ind = self._data_ind[0, 0, :]

        if not self._data_carrying_whitened_inputs:
            # Remove nulled subcarriers from y (guards, dc). New shape:
            # [batch_size, num_rx, num_rx_ant, ...
            #  ..., num_ofdm_symbols, num_effective_subcarriers]
            y_eff = self._removed_nulled_scs(y)
            ####################################################
            ### Prepare the observation y for MIMO detection ###
            ####################################################
            # Transpose y_eff to put num_rx_ant last. New shape:
            # [batch_size, num_rx, num_ofdm_symbols,...
            #  ..., num_effective_subcarriers, num_rx_ant]
            y_dt = tf.transpose(y_eff, [0, 1, 3, 4, 2])
            y_dt = tf.cast(y_dt, self._dtype)

            # Gather only data-carrying symbols
            # New shape:
            # [batch_size, num_rx, num_ofdm_data_symbols,...
            #  ..., num_effective_subcarriers, num_rx_ant]
            y_dt = selectDataCarryingOFDMSymbols(y_dt, 2, data_ind, num_ofdm_data_symbols, num_effective_subcarriers)
            ##############################################
            ### Prepare the err_var for MIMO detection ###
            ##############################################
            # New shape is:
            # [batch_size, num_rx, num_ofdm_symbols,...
            #  ..., num_effective_subcarriers, num_rx_ant, num_tx*num_streams]
            err_var_dt = tf.broadcast_to(err_var, tf.shape(h_hat))
            err_var_dt = tf.transpose(err_var_dt, [0, 1, 5, 6, 2, 3, 4])
            err_var_dt = flatten_last_dims(err_var_dt, 2)
            err_var_dt = tf.cast(err_var_dt, self._dtype)
            err_var_dt = selectDataCarryingOFDMSymbols(err_var_dt, 2, data_ind, num_ofdm_data_symbols,
                                                       num_effective_subcarriers)

            ###############################
            ### Construct MIMO channels ###
            ###############################

            # Reshape h_hat for the construction of desired/interfering channels:
            # [num_rx, num_tx, num_streams_per_tx, batch_size, num_rx_ant, ,...
            #  ..., num_ofdm_symbols, num_effective_subcarriers]
            perm = [1, 3, 4, 0, 2, 5, 6]
            h_dt = tf.transpose(h_hat, perm)

            # Flatten first three dimensions:
            # [num_rx*num_tx*num_streams_per_tx, batch_size, num_rx_ant, ...
            #  ..., num_ofdm_symbols, num_effective_subcarriers]
            h_dt = flatten_dims(h_dt, 3, 0)

            # Gather desired and undesired channels
            ind_desired = self._stream_management.detection_desired_ind
            ind_undesired = self._stream_management.detection_undesired_ind
            h_dt_desired = tf.gather(h_dt, ind_desired, axis=0)
            h_dt_undesired = tf.gather(h_dt, ind_undesired, axis=0)

            # Split first dimension to separate RX and TX:
            # [num_rx, num_streams_per_rx, batch_size, num_rx_ant, ...
            #  ..., num_ofdm_symbols, num_effective_subcarriers]
            h_dt_desired = split_dim(h_dt_desired, [self._stream_management.num_rx, -1], 0)
            h_dt_undesired = split_dim(h_dt_undesired, [self._stream_management.num_rx, -1], 0)

            # Permutate dims to
            # [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,..
            #  ..., num_rx_ant, num_streams_per_rx(num_Interfering_streams_per_rx)]
            perm = [2, 0, 4, 5, 3, 1]
            h_dt_desired = tf.transpose(h_dt_desired, perm)
            h_dt_desired = tf.cast(h_dt_desired, self._dtype)
            h_dt_undesired = tf.transpose(h_dt_undesired, perm)
            h_dt_desired = selectDataCarryingOFDMSymbols(h_dt_desired, 2, data_ind, num_ofdm_data_symbols,
                                                         num_effective_subcarriers)
            h_dt_undesired = selectDataCarryingOFDMSymbols(h_dt_undesired, 2, data_ind, num_ofdm_data_symbols,
                                                           num_effective_subcarriers)

            ##################################
            ### Prepare the noise variance ###
            ##################################
            # no is first broadcast to [batch_size, num_rx, num_rx_ant]
            # then the rank is expanded to that of y
            # then it is transposed like y to the final shape
            # [batch_size, num_rx, num_ofdm_symbols,...
            #  ..., num_effective_subcarriers, num_rx_ant]
            no_dt = expand_to_rank(no, 3, -1)
            no_dt = tf.broadcast_to(no_dt, tf.shape(y)[:3])
            no_dt = expand_to_rank(no_dt, tf.rank(y), -1)
            no_dt = tf.transpose(no_dt, [0, 1, 3, 4, 2])
            no_dt = tf.cast(no_dt, self._dtype)
            ##################################################
            ### Compute the interference covariance matrix ###
            ##################################################
            # Covariance of undesired transmitters
            s_inf = tf.matmul(h_dt_undesired, h_dt_undesired, adjoint_b=True)

            # Thermal noise
            s_no = tf.linalg.diag(no_dt)

            # Channel estimation errors
            # As we have only error variance information for each element,
            # we simply sum them across transmitters and build a
            # diagonal covariance matrix from this
            s_csi = tf.linalg.diag(tf.reduce_sum(err_var_dt, -1))

            # Final covariance matrix
            s = s_inf + s_no + s_csi
            s = tf.cast(s, self._dtype)

            # Noise+Interference Whitening
            s_inv_1_2 = matrix_sqrt_inv(s)

            # Whiten the observation
            y_dt = tf.expand_dims(y_dt, -1)
            y_dt_whitened = tf.matmul(s_inv_1_2, y_dt)

            # Compute channel after whitening
            h_dt_desired_whitened = tf.matmul(s_inv_1_2, h_dt_desired)

            # Step 1: Compute Gram matrix
            # h_dt_desired is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,..
            #  ..., num_rx_ant, num_streams_per_rx(num_Interfering_streams_per_rx)]
            # G is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx, num_streams_per_rx]
        else:
            h_dt_desired_whitened = h_hat
            y_dt_whitened = y
        ##################################
        ### Prepare the noise variance ###
        ##################################
        # no is broadcast to [batch_size, num_rx, num_rx_ant]
        # no_dt = expand_to_rank(no, 3, -1)
        # no_dt = tf.broadcast_to(no_dt, tf.shape(y)[:3])

        ############################################################
        #### SISO LMMSE PIC ###
        ############################################################
        if G is None:
            G = tf.linalg.matmul(h_dt_desired_whitened, h_dt_desired_whitened, adjoint_a=True)
            # y_MF is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx]
            y_MF = tf.linalg.matmul(h_dt_desired_whitened, y_dt_whitened, adjoint_a=True)
        else:
            y_MF = y


        ## following Algorithm 1 from [1]
        # Step 1: Compute Gram matrix
        # h_dt_desired is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,..
        #  ..., num_rx_ant, num_streams_per_rx(num_Interfering_streams_per_rx)]
        # G is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx, num_streams_per_rx]

        # _I_Nt is [1, 1, 1, 1, num_streams_per_rx, num_streams_per_rx]
        _I_Nt = tf.linalg.eye(tf.shape(G)[-1], dtype=self.dtype)
        _I_Nt = tf.reshape(_I_Nt, tf.concat([[1] * (G._rank() - 2), tf.shape(_I_Nt)], 0))

        # Calculate Soft Symbols
        points_reshaped = tf.reshape(self._constellation.points, [1] * 5 + [num_points])

        if llr_a is None:
            # no a priori LLR => no parallel interference cancellation
            y_hat_i_MF = y_MF
            log_pC = None  # tf.math.log(tf.cast(1 / num_points, dtype=tf.float32))
            error_var = 1
            llr_a_out = 0
            _lambda = _I_Nt
        else:
            # Step 2: Calculte soft-symbols and variances

            # llr_a is [batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol]
            # reshape to [batch_size, num_tx, num_streams, num_ofdm_symbols, num_effective_subcarriers, num_bits_per_symbol]
            llr_a_out = llr_a
            llr_a = tf.expand_dims(llr_a, axis=-1)
            llr_a = tf.expand_dims(llr_a, axis=-3)
            llr_int_shape = tf.concat(
                [tf.shape(llr_a)[:-3], [num_ofdm_data_symbols, num_effective_subcarriers, num_bits_per_symbol]], 0)
            llr_a = tf.reshape(llr_a, llr_int_shape)

            [s_hat, error_var, log_pC] = self.soft_symbols(llr_a, points_reshaped, batch_size, num_ofdm_data_symbols,
                                                           num_effective_subcarriers, num_tx, num_streams)

            # Step 3: Perform PIC
            # H^H y_hat_i = y_MF - sum_j!=i gj s_hat_j = y + g_i s_hat_i - sum_j g_j s_hat_j
            # y_MF is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx]
            # G is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx, num_streams_per_rx]
            # _g_j_s_hat_j = tf.linalg.matmul(G, s_hat)
            _s_hat = tf.linalg.matrix_transpose(s_hat)
            # y_hat_i_MF_old = tf.expand_dims(y_MF, axis=-1) + G * _s_hat - _g_j_s_hat_j

            _G_times_s_hat = G * _s_hat
            _g_j_s_hat_j = tf.reduce_sum(_G_times_s_hat, axis=-1, keepdims=True)
            y_hat_i_MF = y_MF + _G_times_s_hat - _g_j_s_hat_j  # @TODO Debug and verify

            # y_hat_i_MF is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_tx*num_streams,
            # num_tx*num_streams]

            # Step 4: Compute A
            # Calculate MMSE MF Filter (efficiently)
            # W = (alpha * (1/diag(A^-1 G)) A^-1 + beta * (1/diag(G))) * H^H
            # Z_bar = alpha * (1/diag(A^-1 G)) A^-1 + beta * (1/diag(G))

            # \Lambda_ii = E_i = error_var
            _lambda = tf.linalg.diag(tf.squeeze(error_var, axis=-1))
            # _lambda is [batch_size, 1, num_ofdm_symbols, num_effective_subcarriers, num_tx*num_streams, num_tx*num_streams]

        no_dt = 1  # expand_to_rank(no, tf.rank(_I_Nt), -1)
        no_dt_complex = tf.cast(no_dt, dtype=self.dtype)
        if A_inv is None:
            # calculate LMMSE filter (unit power Tx signals, perfect CSI)
            # _A is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx, num_streams_per_rx]
            _A = G
            _A = _A + no_dt_complex * _I_Nt

            # Step 5: compute MMSE filter and outputs, calculate A\H^H
            # A_inv is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx, num_streams_per_rx]
            A_inv = matrix_inv(_A)
            # A_inv_Hermitian = tf.transpose(A_inv, conjugate=True, perm=[0, 1, 2, 3, 5, 4])


        # G is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx, num_streams_per_rx]
        # Calculate normalized partial filter
        # normalizing_lmmse = tf.linalg.diag(1 / tf.linalg.diag_part(tf.matmul(A_inv, G)))

        # preprocessing
        if mu_i is None:
            mu_i = tf.reduce_sum(A_inv * tf.linalg.matrix_transpose(G), axis=-1, keepdims=True)
        mu_i_real = tf.math.real(mu_i)
        normalizing_lmmse_col_vec = 1 / mu_i

        normalizing_mf_vec = 1 / tf.linalg.diag_part(G)
        # normalizing_mf = tf.linalg.diag(normalizing_mf_vec)
        # normalizing_mf_col_vec = tf.expand_dims(1 / tf.linalg.diag_part(G), axis=-1)

        # z_i = W^H y_dt = mu_i^-1 a_i^H y_hat_i_MF
        # if PIC: y_hat_i_MF is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_tx*num_streams,
        # num_tx*num_streams]; else:[batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,num_tx*num_streams, 1]
        alpha = tf.cast(self.alpha, dtype=self.dtype)
        if self._two_variables:
            beta = tf.cast(self.beta, dtype=self.dtype)
        else:
            beta = 1 - alpha
        # beta = tf.cast(self.beta, dtype=self.dtype)
        Z_bar = (alpha * normalizing_lmmse_col_vec) * A_inv + tf.linalg.diag(beta * normalizing_mf_vec)
        # rho_i = 1
        if llr_a is not None:  # i.e. PIC
            # calculate error variance
            # error_i is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx, num_streams_per_rx]

            # if PIC: s_hat_i is [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_tx*num_streams,
            # num_tx*num_streams]; else:[batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,num_tx*num_streams, 1]
            G_abs = tf.math.abs(G)
            G_uu = tf.expand_dims(tf.linalg.diag_part(G_abs), axis=-1)
            if self._error_var_term == "default":
                _lambda = tf.cast(_lambda, dtype=self.dtype)
                theta_i = tf.linalg.diag_part(tf.matmul(Z_bar @ ((G @ _lambda + no_dt_complex * _I_Nt) @ G), Z_bar,
                                                        adjoint_b=True) - _lambda)
                theta_i = tf.math.real(theta_i)
                rho_i = tf.expand_dims(1 / theta_i, axis=-1)
            elif self._error_var_term == "sinr_heuristic":
                rho_i = G_uu / (no_dt + G_abs @ error_var)
            elif self._error_var_term == "sinr_heuristic2":
                rho_i =  G_uu / (no_dt + G_uu * error_var)
            elif self._error_var_term == "sinr_heuristic3":     # by inspection heuristic after PIC and matched-filtering
                rho_i =  G_uu / (no_dt + (G_abs - tf.linalg.diag(tf.squeeze(G_uu, axis=-1))) @ error_var)
            elif self._error_var_term == "sinr_heuristic4":  # by inspection variance after PIC and matched-filtering
                G_uu_squared = tf.square(G_uu)
                G_abs_squared = tf.square(G_abs)
                rho_i = G_uu_squared / (G_uu * no_dt + (G_abs_squared - tf.linalg.diag(
                    tf.squeeze(G_uu_squared, axis=-1))) @ error_var)
            elif self._error_var_term == "ocd_paper":
                mu_tilde_i = G_uu / (G_uu + no_dt)
                rho_i = mu_tilde_i/(1-mu_tilde_i)
            elif self._error_var_term == "ocd_paper2":
                rho_i = G_uu/no_dt
            elif self._error_var_term == "lmmse":   # MMSE PIC NPI expression - suboptimal (only optimal if MMSE PIC)
                rho_i = tf.divide(mu_i_real, tf.maximum(1 - error_var * mu_i_real, self._epsilon))
            else:
                raise Exception('unsupported error variance term')
            # print(str(np.min(llr_a_out.numpy())))
            # filtering
            s_hat_i = tf.reduce_sum(Z_bar * tf.linalg.matrix_transpose(y_hat_i_MF), axis=-1)

            # Step 6: calculate LLRs

            # calculate exponents
            # Compute squared distances from y to all points

            # pC is [batch_size, num_tx, num_streams, num_ofdm_symbols, num_effective_subcarriers,
            # num_constellation]; transform pC to [batch_size, 1, num_ofdm_symbols, num_effective_subcarriers,
            # num_tx*num_streams, num_constellation]
            if log_pC is not None:      # is none in low-complexity mode
                log_pC = tf.transpose(log_pC, [0, 3, 4, 1, 2, 5])
                log_pC_int_shape = tf.concat(
                    [[batch_size], [1], [num_ofdm_data_symbols], [num_effective_subcarriers], [num_tx * num_streams],
                     [num_points]], 0)
                log_pC = tf.reshape(log_pC, log_pC_int_shape)

            s_hat_i = tf.expand_dims(s_hat_i, axis=-1)
            # squared_dist = tf.math.pow(tf.math.abs(tf.expand_dims(s_hat_i, axis=-1) - points_reshaped), 2)
        else:       # preprocessing / first IDD iteration
            # intially (when LMMSE filtering) --> apply LMMSE error variance
            # rho_i = tf.divide(mu_i_real, tf.maximum(1 - error_var * mu_i_real, self._epsilon))
            rho_i = tf.divide(mu_i_real, tf.maximum(1 - mu_i_real, self._epsilon))
            s_hat_i = tf.linalg.matmul(Z_bar, y_hat_i_MF)
            # squared_dist = tf.math.pow(tf.math.abs(s_hat_i - points_reshaped), 2)

        llr_d = self.LLLCalculation(s_hat_i, rho_i, points_reshaped, log_pC)

        # internal llr_a shape [batch_size, num_tx, num_streams, num_ofdm_symbols, num_effective_subcarriers,
        # num_bits_per_symbol]
        # outer llr_a shape is [batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol]
        # convert llr_d to out-shape
        llr_d = tf.squeeze(llr_d, axis=[1])
        tmp_shape = tf.concat([[batch_size], [num_ofdm_data_symbols], [num_effective_subcarriers], [num_tx], [num_streams],
                               [num_bits_per_symbol]], 0)
        llr_d = tf.reshape(llr_d, tmp_shape)
        llr_d = tf.transpose(llr_d, [0, 3, 4, 1, 2, 5])
        out_shape = tf.concat([[batch_size], [num_tx], [num_streams], [num_data_symbols * num_bits_per_symbol]], 0)
        llr_d = tf.reshape(llr_d, out_shape)

        # subtract llr_a => llr_e = llr_d - llr_a
        if log_pC is not None:
            llr_e = llr_d - llr_a_out  # extrinsic LLRs
        else:
            llr_e = llr_d

        return [llr_e, A_inv, G, y_MF, mu_i, h_dt_desired_whitened]
