"""
Simulation script defining all simulation parameters, model definitions, training and the performance benchmark

This implementation does NOT support XLA mode. Refer to the latest Sionna release for an implementation of MMSE PIC that
supports XLA.
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model

# import Sionna components
import sionna
from sionna.channel.tr38901 import PanelArray, UMi, UMa, RMa
from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LMMSEEqualizer, LSChannelEstimator, PilotPattern, \
    RemoveNulledSubcarriers
from sionna.mapping import Constellation, Mapper, Demapper
from sionna.fec.ldpc import LDPC5GEncoder
from sionna.utils import BinarySource, ebnodb2no, sim_ber, expand_to_rank
from sionna.channel import RayleighBlockFading, OFDMChannel, gen_single_sector_topology

# import self-implemented and extended Sionna compontents
from source.mmsePIC import SisoMmsePicDetector, sisoLoCoPicDetector
from source.customOFDMChannel import customOFDMChannel
from source.dampenedLdpc5gDecoder import dampenedLDPC5GDecoder
from source.simulationFunctions import save_data, save_weights, load_weights, train_model_BLER, genRandIdx

#####################################################################################################################
## Simulation Parameters
#####################################################################################################################

# Select either of the two simulation scenarios presented in the ASILOMAR paper
simulation_scenario = "8x4 Rayleigh Perf-CSI"
# simulation_scenario = "16x4 REMCOM CEst"

if simulation_scenario == "16x4 REMCOM CEst":
    n_ue = 4
    n_bs_ant = 16  # 16 BS antennas
    ebno_db_min = -5  # min EbNo value in dB for training
    ebno_db_max = 15
    stepsize = 2
    perfect_csi = False
    # REMCOM channels
    REMCOM_CHANNELS = True
    channel_model_str = None  # None for REMCOM; alternative "UMi", "UMa", "RMa"
elif simulation_scenario == "8x4 Rayleigh Perf-CSI":
    n_ue = 4
    n_bs_ant = 8  # 8 BS antennas
    ebno_db_min = -5  # min EbNo value in dB for training
    ebno_db_max = 5
    stepsize = 1
    perfect_csi = True
    # iid Rayleigh fading channels
    REMCOM_CHANNELS = False
    channel_model_str = "Rayleigh"  # alternative "UMi", "UMa", "RMa"

# set the total number of LDPC iterations to study
num_ldpc_iter = 12
gamma_0 = 1  # 0 is resetting decoder, 1 is non-resetting decoder

GPU_NUM = 0
# Debug => smaller batchsize, fewer training and Monte Carlo iterations
DEBUG = False
# If REMCOM channels are selected, random subsampling
RAND_SUBSAMPLING = True

normalizing_channels = True
low_complexity = True
OPTIMIZED_LDPC_INTERLEAVER = True

# LoS True only line of sight, False: none-los, none: mix of los and none-los
LoS = True
# Antenna_Array = "Single-Pol-Omni-ULA"
# Antenna_Array = "Single-Pol-ULA"
Antenna_Array = "Dual-Pol-ULA"
error_var_term = "sinr_heuristic4"
MOBILITY = False

# Select GPU 0 to run TF/Sionna
gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available :', len(gpus))
if gpus:
    gpu_num = GPU_NUM  # Number of the GPU to be used
    try:
        # tf.config.set_visible_devices([], 'GPU')
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        print('Only GPU number', gpu_num, 'used.')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)

# Set seeds (TF and NP)
tf.random.set_seed(1)
np.random.seed(1)

# OFDM Waveform Settings
# 30 kHz subcarrier spacing
subcarrier_spacing = 30e3
# with 30MHz sc spacing 273 in 100 MHz n78 channel
num_resource_blocks = 273
num_subcarriers = num_resource_blocks * 12
# we use only a subset of resource blocks for our simulations (here, some smaller number 5-15)
rb_used = 5
sc_used = rb_used * 12

carrier_freq = 3.75e9   # 3.7GHz-3.8GHz n78 100MHz band
bandwidth = 100e6
# effective_bandwidth = subcarrier_spacing * num_subcarriers
# 14 OFDM TIME symbols is one 5G OFDM frame
num_ofdm_symbols = 14
num_pilot_symbols = 4

# simulation parameters
batch_size = int(1e3)  # number of symbols to be analyzed
num_iter = 100  # number of Monte Carlo Iterations (total number of Monte Carlo runs is num_iter*batch_size)

num_pretraining_iterations = 2500
num_BLER_training_iterations = 2500

training_batch_size = int(40)
# path where to save the learnt parameters

if low_complexity:
    demapping_method = "maxlog"
    ldpc_cn_update_func = "minsum"
else:
    demapping_method = "app"
    ldpc_cn_update_func = "boxplus"

if DEBUG:
    batch_size = int(1e1)
    num_iter = 1
    num_pretraining_iterations = 10
    num_BLER_training_iterations = 10
    num_iter_per_epoch = 5
    stepsize = 5
    training_batch_size = int(2e0)
    tf.config.run_functions_eagerly(True)
    mi_stepsize = 0.25
    rb_used = 1
else:
    tf.config.run_functions_eagerly(False)

num_bits_per_symbol = 4  # bits per modulated symbol, i.e., 2^4 = 16-QAM
_num_const_bits_ldpc = num_bits_per_symbol
if not OPTIMIZED_LDPC_INTERLEAVER:
    _num_const_bits_ldpc = None
num_streams_per_tx = 1

case = "asilomar_range_non_resetting" + simulation_scenario

if REMCOM_CHANNELS:
    # load remcom channels
    mat = scipy.io.loadmat('../data/channels/dualPolULA_30kHz-SC.mat')
    remcom_scenario_str = "MIMO Outdoor Grid "
    channels = mat["H_Freq_set"]

    print("tic")
    # generate the UE indices combinations (samples) for random subsampling
    chanIdxComb_training = genRandIdx(np.shape(channels)[0], n_ue, training_batch_size * (num_pretraining_iterations +
                                                                                          num_BLER_training_iterations))
    chanIdxComb_validation = genRandIdx(np.shape(channels)[0], n_ue, batch_size * num_iter)
    print("toc")
    channel_model_str = " REMCOM " + remcom_scenario_str
else:
    chanIdxComb_training = None
    chanIdxComb_validation = None
    if channel_model_str in ["UMi", "UMa", "RMa"]:
        bs_array = None
        if Antenna_Array == "4x4":
            bs_array = PanelArray(num_rows_per_panel=int(n_bs_ant / 4),
                                  num_cols_per_panel=4,
                                  polarization='single',
                                  polarization_type='V',
                                  antenna_pattern='38.901',
                                  carrier_frequency=carrier_freq)
        elif Antenna_Array == "Single-Pol-ULA":
            bs_array = PanelArray(num_rows_per_panel=1,
                                  num_cols_per_panel=n_bs_ant,
                                  polarization='single',
                                  polarization_type='V',
                                  antenna_pattern='38.901',
                                  carrier_frequency=carrier_freq)
        elif Antenna_Array == "Single-Pol-Omni-ULA":
            bs_array = PanelArray(num_rows_per_panel=1,
                                  num_cols_per_panel=n_bs_ant,
                                  polarization='single',
                                  polarization_type='V',
                                  antenna_pattern='omni',
                                  carrier_frequency=carrier_freq)
        elif Antenna_Array == "Dual-Pol-ULA":
            bs_array = PanelArray(num_rows_per_panel=1,
                                  num_cols_per_panel=int(n_bs_ant/2),
                                  polarization='dual',
                                  polarization_type='cross',
                                  antenna_pattern='38.901',
                                  carrier_frequency=carrier_freq)
        else:
            bs_array = PanelArray(num_rows_per_panel=int(n_bs_ant/2/8),
                num_cols_per_panel = 8,
                polarization = 'dual',
                polarization_type = 'cross',
                antenna_pattern = '38.901',
                carrier_frequency = carrier_freq)
        ut_array = PanelArray(num_rows_per_panel=1,
            num_cols_per_panel = 1,
            polarization = 'single',
            polarization_type = 'V',
            antenna_pattern = 'omni',
            carrier_frequency = carrier_freq)
    elif channel_model_str =="Rayleigh":
        pass
    else:
        raise NameError('channel_model_string not found')

model_weights_path = '../data/weights/duidd_Perf-CSI_' + str(perfect_csi) + channel_model_str + "_" + str(ebno_db_min) + "_" + \
                     str(ebno_db_max) + "_" + str(n_bs_ant) + "x" + str(n_ue) + case
max_ut_velocity = 50.0/3.6
if not MOBILITY:
    max_ut_velocity = 0

# LDPC ENCODING DECODING
# LDPC code parameters
r = 0.5  # rate 1/2
n = int(sc_used * (num_ofdm_symbols - num_pilot_symbols) * num_bits_per_symbol) # code length
k = int(n * r)  # number of information bits per codeword

# Constellation 16 QAM
# initialize mapper (and demapper) for constellation object
constellation = Constellation("qam", num_bits_per_symbol=num_bits_per_symbol)

# Define MU-MIMO System
rx_tx_association = np.zeros([1, n_ue])
rx_tx_association[0, :] = 1

# stream management stores a mapping from Rx and Tx
sm = StreamManagement(rx_tx_association, num_streams_per_tx)

# 2x Kronecker Pilot Pattern with UE 0,1 on OFDM symbol 2,11 and UE 2,3 on OFDM symbol 3,12
pilot_mask = [2,3,11,12]
mask = np.zeros([   n_ue, 1, num_ofdm_symbols, sc_used], bool)
mask[...,pilot_mask,:] = True
pilots = np.zeros([n_ue, 1, np.sum(mask[0,0])])
pilots[0,0, 0*sc_used:1*sc_used:2] = 1
pilots[0,0, 2*sc_used:3*sc_used:2] = 1
pilots[1,0,1+0*sc_used:1*sc_used:2] = 1
pilots[1,0,1+2*sc_used:3*sc_used:2] = 1
pilots[2,0, 1*sc_used:2*sc_used:2] = 1
pilots[2,0, 3*sc_used:4*sc_used:2] = 1
pilots[3,0,1+1*sc_used:2*sc_used:2] = 1
pilots[3,0,1+3*sc_used:4*sc_used:2] = 1
pilot_pattern = PilotPattern(mask, pilots, normalize=True)

rg_chan_est = ResourceGrid(num_ofdm_symbols=14, fft_size=sc_used,
                           subcarrier_spacing=subcarrier_spacing, cyclic_prefix_length=20,
                           num_tx=n_ue, pilot_ofdm_symbol_indices=pilot_mask,
                           num_streams_per_tx=num_streams_per_tx, pilot_pattern=pilot_pattern,
                           )
rg_chan_est.show()
rg_chan_est.pilot_pattern.show()
plt.show()


#####################################################################################################################
## Define Models
#####################################################################################################################
class BaseModel(Model):
    def __init__(self, num_bp_iter=5, REMCOM_ChanIdxComb=None, loss_fun="BCE", training=False):
        super().__init__()
        num_bp_iter = int(num_bp_iter)
        ######################################
        ## Transmitter
        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(k, n, num_bits_per_symbol=_num_const_bits_ldpc)
        self._mapper = Mapper(constellation=constellation)
        self._rg_mapper = ResourceGridMapper(rg_chan_est)
        self._lossFun = loss_fun
        self._training = training

        ######################################
        ## Channel
        if REMCOM_CHANNELS:
            self._channel = customOFDMChannel(channel_set=channels,
                                              resource_grid=rg_chan_est,
                                              add_awgn=True,
                                              normalize_channel=True, return_channel=True, chanIdxComb=REMCOM_ChanIdxComb,
                                              randomSubSamplingChanIdx=RAND_SUBSAMPLING)
            self._channel_model = None
        else:
            if channel_model_str == "UMi":
                self._channel_model = UMi(carrier_frequency=carrier_freq,
                                    o2i_model='low',
                                    ut_array=ut_array,
                                    bs_array=bs_array,
                                    direction='uplink')
            elif channel_model_str == "UMa":
                self._channel_model = UMa(carrier_frequency=carrier_freq,
                                          o2i_model='low',
                                          ut_array=ut_array,
                                          bs_array=bs_array,
                                          direction='uplink')
            elif channel_model_str == "RMa":
                self._channel_model = RMa(carrier_frequency=carrier_freq,
                                          ut_array=ut_array,
                                          bs_array=bs_array,
                                          direction='uplink')
            elif channel_model_str == "Rayleigh":
                self._channel_model = RayleighBlockFading(num_rx=1, num_rx_ant=n_bs_ant, num_tx=n_ue, num_tx_ant=1)
            self._channel = OFDMChannel(channel_model=self._channel_model,
                                        resource_grid=rg_chan_est,
                                        add_awgn=True,
                                        normalize_channel=normalizing_channels, return_channel=True)
    def new_topology(self, batch_size):
        """Set new topology"""
        if channel_model_str in ["UMi", "UMa", "RMa"]:
            # sensible values according to 3GPP standard, no mobility by default
            topology = gen_single_sector_topology(batch_size,
                                                  n_ue, max_ut_velocity=max_ut_velocity,
                                                  scenario=channel_model_str.lower())
            self._channel_model.set_topology(*topology, los=LoS)
    def computeLoss(self, b, c, b_hat):
        # b are the information (inf.) bits transmitted, c the code bits
        # b_hat are inf. bit estimates (training=False) and estimated LLRs of the inf. bits (training=True)
        if self._training:
            cost = 0
            if 'BCE' in self._lossFun:
                bce = tf.nn.sigmoid_cross_entropy_with_logits(b, b_hat)     # training on information bits
                # Notice: we observed slightly better performance in the REMCOM scenario, when considering all code-
                # bits for training; therefore, use: bce = tf.nn.sigmoid_cross_entropy_with_logits(c, b_hat)
                if 'LogSumExp' in self._lossFun:
                    if 'normalized' in self._lossFun:
                        x_max = tf.reduce_max(bce, axis=-1, keepdims=True)
                        x_max_ = tf.squeeze(x_max, axis=-1)
                        cost = x_max_ + tf.math.log(
                            tf.math.reduce_sum(tf.exp(bce - x_max), axis=-1) - tf.exp(-x_max_) * tf.cast(
                                tf.shape(bce)[-1] - 1, tf.float32))
                    else:
                        cost = tf.math.reduce_logsumexp(bce, axis=-1)
                elif 'pNorm_2' in self._lossFun:
                    if 'noRoot' in self._lossFun:
                        cost = tf.pow(bce, 2)
                    else:
                        bce = tf.concat([bce, expand_to_rank(tf.ones(tf.shape(bce)[:-1]) * (1e-4), tf.rank(bce))],
                                        axis=-1)
                        cost = tf.norm(bce, ord=2, axis=-1)
                elif 'pNorm_4' in self._lossFun:
                    if 'noRoot' in self._lossFun:
                        cost = tf.pow(bce, 4)
                    else:
                        bce = tf.concat([bce, expand_to_rank(tf.ones(tf.shape(bce)[:-1]) * (1e-4), tf.rank(bce))],
                                        axis=-1)
                        cost = tf.norm(bce, ord=4, axis=-1)
                elif 'max' in self._lossFun:
                    cost = tf.math.reduce_max(bce, axis=-1)
                elif 'softMax' in self._lossFun:
                    # alpha = 0 corresponds to arithmetic mean (standard BCE)
                    alpha = 0.0
                    if '0_1' in self._lossFun:
                        alpha = 0.1
                    if '0_5' in self._lossFun:
                        alpha = 0.5
                    elif '1' in self._lossFun:
                        alpha = 1.0
                    elif '2' in self._lossFun:
                        alpha = 2.0
                    x_max = tf.reduce_max(bce, axis=-1, keepdims=True)
                    _exp_alpha_bce = tf.exp(alpha * (bce - x_max))
                    cost = tf.reduce_sum(bce * _exp_alpha_bce, axis=-1) / tf.reduce_sum(_exp_alpha_bce, axis=-1)
                else:
                    cost = bce
            elif 'PrBlckErr' in self._lossFun:
                # Not numerically stable: cost = 1 - tf.reduce_prod(1/(1+tf.exp(b_hat*(2*c-1))), axis=-1)
                cost = 1 - tf.reduce_prod(0.5 - 0.5 * tf.tanh(-b_hat * (b - 0.5)), axis=-1)
                if 'Log' in self._lossFun:
                    cost = tf.math.log(tf.reduce_mean(cost))
            elif 'MSE' in self._lossFun:
                p = 0.5 * (1 - tf.tanh(-b_hat / 2.0))
                cost = tf.reduce_mean(tf.pow(b - p, 2.0), axis=-1)
                # cost = tf.keras.losses.MSE(c, p)
            else:
                raise NotImplementedError('Not implemented:' + self._lossFun)
            if 'deweighting_SNR' in self._lossFun:
                cost = tf.reduce_mean(cost, axis=range(1, tf.rank(cost)))
            else:
                cost = tf.reduce_mean(cost)
            return cost
        else:
            return b, b_hat  # Ground truth and reconstructed information bits returned for BER/BLER computation

class ChanEstBaseModel(BaseModel):
    def __init__(self, num_bp_iter=5, perfect_csi=False, REMCOM_ChanIdxComb=None, loss_fun="BCE", training=False):
        super().__init__(num_bp_iter=num_bp_iter, REMCOM_ChanIdxComb=REMCOM_ChanIdxComb, loss_fun=loss_fun, training=training)
        self._num_idd_iter = 3

        ######################################
        ## Transmitter
        self._encoder = LDPC5GEncoder(k, n, num_bits_per_symbol=_num_const_bits_ldpc)
        self._rg_mapper = ResourceGridMapper(rg_chan_est)

        ######################################
        ## Receiver
        ######################################
        ## Receiver
        self._perfect_csi = perfect_csi
        if self._perfect_csi:
            self._removeNulledSc = RemoveNulledSubcarriers(rg_chan_est)
        else:
            self._ls_est = LSChannelEstimator(rg_chan_est, interpolation_type="lin_time_avg")
            # time averaging due to no mobility

class LmmseBaselineModelChanEst(ChanEstBaseModel):
    def __init__(self, num_bp_iter=5, perfect_csi=False, REMCOM_ChanIdxComb=None):
        super().__init__(num_bp_iter, perfect_csi=perfect_csi, REMCOM_ChanIdxComb=REMCOM_ChanIdxComb)
        ######################################
        ## Receiver
        self._equalizer = LMMSEEqualizer(rg_chan_est, sm)
        self._demapper = Demapper(demapping_method=demapping_method, constellation=constellation)
        self._LDPCDec0 = dampenedLDPC5GDecoder(self._encoder,
                                       cn_type=ldpc_cn_update_func, return_infobits=True,  # using minsum ldpc function
                                       num_iter=int(num_bp_iter),
                                       hard_out=True)

    @tf.function()
    def call(self, batch_size, ebno_db):
        self.new_topology(batch_size)
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)

        ######################################
        ## Transmitter
        no = ebnodb2no(ebno_db, num_bits_per_symbol, r)
        b = self._binary_source([batch_size, n_ue, num_streams_per_tx, k])
        c = self._encoder(b)
        # Modulation
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        ######################################
        ## Channel
        no_ = expand_to_rank(no, tf.rank(x_rg))
        y, h = self._channel([x_rg, no_])

        ######################################
        ## Receiver
        if self._perfect_csi:# simulation_scenario = "16x4 REMCOM CEst"

            # nulled subcarriers must be removed, either by ls_est or manually by removedNulledSc
            # No channel estimation error when perfect CSI knowledge is assumed
            h_hat = self._removeNulledSc(h)
            chan_est_var = tf.zeros(tf.shape(h_hat), dtype=tf.float32)
        else:
            h_hat, chan_est_var = self._ls_est([y, no])

        [x_hat, no_eff] = self._equalizer([y, h_hat, chan_est_var, no])
        llr_ch = self._demapper([x_hat, no_eff])
        b_hat = self._LDPCDec0(llr_ch)

        return self.computeLoss(b, c, b_hat)  # Ground truth and reconstructed information bits returned for BER/BLER computation

class OneIterModelLmmseChanEst(ChanEstBaseModel):
    def __init__(self, training=False, num_bp_iter=5, perfect_csi=False, loss_fun='BCE', REMCOM_ChanIdxComb=None):
        super().__init__(num_bp_iter, perfect_csi=perfect_csi, REMCOM_ChanIdxComb=REMCOM_ChanIdxComb, loss_fun=loss_fun, training=training)

        self._detector0 = SisoMmsePicDetector(rg_chan_est, sm, demapping_method=demapping_method,
                                                 constellation=constellation, low_complexity=low_complexity)
        self._LDPCDec0 = dampenedLDPC5GDecoder(self._encoder,  cn_type=ldpc_cn_update_func,
                                               return_infobits=True, num_iter=int(num_bp_iter),
                                               stateful=False, alpha0=0.0,trainDamping=training,
                                               hard_out=not (self._training))
        # For training with all code-bits, select return_infobits=not (self._training)

        self._eta1 = tf.Variable(1.0, trainable=training, dtype=tf.float32, name="eta1")

    @property
    def eta1(self):
        return self._eta1

    @tf.function()
    def call(self, batch_size, ebno_db):
        self.new_topology(batch_size)
        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)

        ######################################
        ## Transmitter
        no = ebnodb2no(ebno_db, num_bits_per_symbol, r)
        # Outer coding is only performed if not training
        b = self._binary_source([batch_size, n_ue, num_streams_per_tx, k])
        c = self._encoder(b)
        # Modulation
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        ######################################
        ## Channel
        # A batch of new channel realizations is sampled and applied at every inference
        no_ = expand_to_rank(no, tf.rank(x_rg))
        y, h = self._channel([x_rg, no_])

        ######################################
        ## Receiver
        if self._perfect_csi:
            h_hat = self._removeNulledSc(h)
            chan_est_var = tf.zeros(tf.shape(h_hat), dtype=tf.float32)
            # No channel estimation error when perfect CSI knowledge is assumed
        else:
            h_hat, chan_est_var = self._ls_est([y, no])

        chan_est_var = self.eta1 * chan_est_var

        [llr_ch, y_MF, h_dt_desired_whitened, G] = self._detector0(
            [y, h_hat, chan_est_var, no, None, None])  # A_inv and G none => will calculate itself then

        b_hat = self._LDPCDec0(llr_ch)

        return self.computeLoss(b, c, b_hat)

# MMSE PIC
# two iterations IDD model applying LoCo PIC as first detector, MMSE PIC as second
class TwoIterModelMmsePicChanEst(ChanEstBaseModel):
    def __init__(self, training=False, num_bp_iter=5, perfect_csi=False, loss_fun='BCE', REMCOM_ChanIdxComb=None):
        super().__init__(num_bp_iter=num_bp_iter, perfect_csi=perfect_csi, REMCOM_ChanIdxComb=REMCOM_ChanIdxComb, loss_fun=loss_fun, training=training)

        self._detector0 = sisoLoCoPicDetector(rg_chan_est, sm, demapping_method=demapping_method,
                                                           constellation=constellation, low_complexity=low_complexity,
                                                           error_var_term=error_var_term, trainable=training)
        self._detector1 = SisoMmsePicDetector(rg_chan_est, sm, demapping_method=demapping_method,
                                                 constellation=constellation, low_complexity=low_complexity,
                                                  data_carrying_whitened_inputs=True)
        self._LDPCDec0 = dampenedLDPC5GDecoder(self._encoder, cn_type=ldpc_cn_update_func, return_infobits=False,
                                               num_iter=int(num_bp_iter), stateful=True, alpha0=0.0,
                                               trainDamping=training,  hard_out=False)
        self._LDPCDec1 = dampenedLDPC5GDecoder(self._encoder, cn_type=ldpc_cn_update_func,
                                               return_infobits=True,
                                               num_iter=int(num_bp_iter), stateful=True, alpha0=0.0,
                                               trainDamping=training,
                                               hard_out=not (self._training))
        # For training with all code-bits, select return_infobits=not (self._training)

        self._alpha1 = tf.Variable(1, trainable=training, dtype=tf.float32, name="alpha1")
        self._alpha2 = tf.Variable(1, trainable=training, dtype=tf.float32, name="alpha2")

        self._beta1 = tf.Variable(0, trainable=training, dtype=tf.float32, name="beta1")
        self._beta2 = tf.Variable(0, trainable=training, dtype=tf.float32, name="beta2")

        self._gamma1 = tf.Variable(gamma_0, trainable=training, dtype=tf.float32, name="gamma1")

        self._eta1 = tf.Variable(1.0, trainable=training, dtype=tf.float32, name="eta1")

    @property
    def eta1(self):
        return self._eta1

    @property
    def gamma1(self):
        return self._gamma1

    @property
    def alpha1(self):
        return self._alpha1

    @property
    def beta1(self):
        return self._beta1

    @property
    def alpha2(self):
        return self._alpha2

    @property
    def beta2(self):
        return self._beta2

    @tf.function()
    def call(self, batch_size, ebno_db):
        self.new_topology(batch_size)
        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)

        ######################################
        ## Transmitter
        no = ebnodb2no(ebno_db, num_bits_per_symbol, r)
        # Outer coding is only performed if not training
        b = self._binary_source([batch_size, n_ue, num_streams_per_tx, k])
        c = self._encoder(b)
        # Modulation
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        ######################################
        ## Channel
        # A batch of new channel realizations is sampled and applied at every inference
        no_ = expand_to_rank(no, tf.rank(x_rg))
        y, h = self._channel([x_rg, no_])

        ######################################
        ## Receiver
        if self._perfect_csi:
            h_hat = self._removeNulledSc(h)
            chan_est_var = tf.zeros(tf.shape(h_hat), dtype=tf.float32)  # No channel estimation error when perfect CSI knowledge is assumed
        else:
            h_hat, chan_est_var = self._ls_est([y, no])

        chan_est_var = self.eta1 * chan_est_var

        [llr_ch, _, G, y_MF, _, h_dt_desired_whitened] = self._detector0(
            [y, h_hat, chan_est_var, no, None, None, None, None])  # A_inv and G none => will calculate itself then

        [llr_dec, msg_vn] = self._LDPCDec0([llr_ch, None])
        llr_a_det = self.alpha1 * llr_dec - self.beta1 * llr_ch

        [llr_ch, _, _, _] = self._detector1([y_MF, h_dt_desired_whitened, chan_est_var, no, llr_a_det, G])
        llr_a_dec = self.alpha2 * llr_ch - self.beta2 * llr_a_det

        [b_hat, _] = self._LDPCDec1([llr_a_dec, self.gamma1*msg_vn])

        return self.computeLoss(b, c, b_hat)


# three iterations IDD model applying LoCo PIC as first detector, MMSE PIC as second and third
class ThreeIterModelMmsePicChanEst(ChanEstBaseModel):
    def __init__(self, training=False, num_bp_iter=5, perfect_csi=False, loss_fun='BCE', REMCOM_ChanIdxComb=None):
        super().__init__(num_bp_iter, perfect_csi=perfect_csi, REMCOM_ChanIdxComb=REMCOM_ChanIdxComb, loss_fun=loss_fun, training=training)

        self._detector0 = sisoLoCoPicDetector(rg_chan_est, sm, demapping_method=demapping_method,
                                                           constellation=constellation, low_complexity=low_complexity,
                                                           error_var_term=error_var_term, trainable=training)
        self._detector1 = SisoMmsePicDetector(rg_chan_est, sm, demapping_method=demapping_method,
                                                 constellation=constellation, low_complexity=low_complexity,
                                                  data_carrying_whitened_inputs=True)
        self._LDPCDec0 = dampenedLDPC5GDecoder(self._encoder,
                                               cn_type=ldpc_cn_update_func, return_infobits=False,
                                               num_iter=int(num_bp_iter), stateful=True, alpha0=0.0,
                                               trainDamping=training,
                                               hard_out=False)
        self._LDPCDec1 = dampenedLDPC5GDecoder(self._encoder,
                                               cn_type=ldpc_cn_update_func, return_infobits=False,
                                               num_iter=int(num_bp_iter), stateful=True, alpha0=0.0,
                                               trainDamping=training,
                                               hard_out=False)
        self._LDPCDec2 = dampenedLDPC5GDecoder(self._encoder,
                                               cn_type=ldpc_cn_update_func,
                                               return_infobits=True,
                                               num_iter=int(num_bp_iter), stateful=True, alpha0=0.0,
                                               trainDamping=training,
                                               hard_out=not (self._training))
        # For training with all code-bits, select return_infobits=not (self._training)

        self._alpha1 = tf.Variable(1, trainable=training, dtype=tf.float32, name="alpha1")
        self._alpha2 = tf.Variable(1, trainable=training, dtype=tf.float32, name="alpha2")
        self._alpha3 = tf.Variable(1, trainable=training, dtype=tf.float32, name="alpha3")
        self._alpha4 = tf.Variable(1, trainable=training, dtype=tf.float32, name="alpha4")

        self._beta1 = tf.Variable(0, trainable=training, dtype=tf.float32, name="beta1")
        self._beta2 = tf.Variable(0, trainable=training, dtype=tf.float32, name="beta2")
        self._beta3 = tf.Variable(0, trainable=training, dtype=tf.float32, name="beta3")
        self._beta4 = tf.Variable(0, trainable=training, dtype=tf.float32, name="beta4")

        self._gamma1 = tf.Variable(gamma_0, trainable=training, dtype=tf.float32, name="gamma1")
        self._gamma2 = tf.Variable(gamma_0, trainable=training, dtype=tf.float32, name="gamma2")

        self._eta1 = tf.Variable(1.0, trainable=training, dtype=tf.float32, name="eta1")

    @property
    def eta1(self):
        return self._eta1

    @property
    def gamma1(self):
        return self._gamma1

    @property
    def gamma2(self):
        return self._gamma2

    @property
    def alpha1(self):
        return self._alpha1

    @property
    def beta1(self):
        return self._beta1

    @property
    def alpha2(self):
        return self._alpha2

    @property
    def beta2(self):
        return self._beta2

    @property
    def alpha3(self):
        return self._alpha3

    @property
    def beta3(self):
        return self._beta3

    @property
    def alpha4(self):
        return self._alpha4

    @property
    def beta4(self):
        return self._beta4

    @tf.function()
    def call(self, batch_size, ebno_db):
        self.new_topology(batch_size)
        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)

        ######################################
        ## Transmitter
        no = ebnodb2no(ebno_db, num_bits_per_symbol, r)
        # Outer coding is only performed if not training
        b = self._binary_source([batch_size, n_ue, num_streams_per_tx, k])
        c = self._encoder(b)
        # Modulation
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        ######################################
        ## Channel
        # A batch of new channel realizations is sampled and applied at every inference
        no_ = expand_to_rank(no, tf.rank(x_rg))
        y, h = self._channel([x_rg, no_])

        ######################################
        ## Receiver
        if self._perfect_csi:
            h_hat = self._removeNulledSc(h)
            chan_est_var = tf.zeros(tf.shape(h_hat), dtype=tf.float32)
            # No channel estimation error when perfect CSI knowledge is assumed
        else:
            h_hat, chan_est_var = self._ls_est([y, no])

        chan_est_var = self.eta1 * chan_est_var

        [llr_ch, _, G, y_MF, _, h_dt_desired_whitened] = self._detector0(
            [y, h_hat, chan_est_var, no, None, None, None, None])  # A_inv and G none => will calculate itself then

        [llr_dec, msg_vn] = self._LDPCDec0([llr_ch, None])
        llr_a_det = self.alpha1 * llr_dec - self.beta1 * llr_ch

        [llr_ch, _, _, _] = self._detector1([y_MF, h_dt_desired_whitened, chan_est_var, no, llr_a_det, G])
        llr_a_dec = self.alpha2 * llr_ch - self.beta2 * llr_a_det

        [llr_dec, msg_vn] = self._LDPCDec1([llr_a_dec, self.gamma1*msg_vn])
        llr_a_det = self.alpha3 * llr_dec - self.beta3 * llr_a_dec

        [llr_ch, _, _, _] = self._detector1([y_MF, h_dt_desired_whitened, chan_est_var, no, llr_a_det, G])
        llr_a_dec = self.alpha4 * llr_ch - self.beta4 * llr_a_det

        [b_hat, _] = self._LDPCDec2([llr_a_dec, self.gamma2*msg_vn])

        return self.computeLoss(b, c, b_hat)

# four iterations IDD model applying LoCo PIC as first detector, MMSE PIC as second, third and fourth
class FourIterModelMmsePicChanEst(ChanEstBaseModel):
    def __init__(self, training=False, num_bp_iter=5, perfect_csi=False, loss_fun='BCE', REMCOM_ChanIdxComb=None):
        super().__init__(num_bp_iter, perfect_csi=perfect_csi, REMCOM_ChanIdxComb=REMCOM_ChanIdxComb, loss_fun=loss_fun, training=training)

        self._detector0 = sisoLoCoPicDetector(rg_chan_est, sm, demapping_method=demapping_method,
                                                           constellation=constellation, low_complexity=low_complexity,
                                                           error_var_term=error_var_term, trainable=training)
        self._detector1 = SisoMmsePicDetector(rg_chan_est, sm, demapping_method=demapping_method,
                                                 constellation=constellation, low_complexity=low_complexity,
                                                  data_carrying_whitened_inputs=True)
        self._LDPCDec0 = dampenedLDPC5GDecoder(self._encoder,
                                               cn_type=ldpc_cn_update_func, return_infobits=False,
                                               num_iter=int(num_bp_iter), stateful=True, alpha0=0.0,
                                               trainDamping=training,
                                               hard_out=False)
        self._LDPCDec1 = dampenedLDPC5GDecoder(self._encoder,
                                               cn_type=ldpc_cn_update_func, return_infobits=False,
                                               num_iter=int(num_bp_iter), stateful=True, alpha0=0.0,
                                               trainDamping=training,
                                               hard_out=False)
        self._LDPCDec2 = dampenedLDPC5GDecoder(self._encoder,
                                               cn_type=ldpc_cn_update_func, return_infobits=False,
                                               num_iter=int(num_bp_iter), stateful=True, alpha0=0.0,
                                               trainDamping=training,
                                               hard_out=False)
        self._LDPCDec3 = dampenedLDPC5GDecoder(self._encoder,
                                               cn_type=ldpc_cn_update_func,
                                               return_infobits=True,
                                               num_iter=int(num_bp_iter), stateful=True, alpha0=0.0,
                                               trainDamping=training,
                                               hard_out=not (self._training))
        # For training with all code-bits, select return_infobits=not (self._training)

        self._alpha1 = tf.Variable(1, trainable=training, dtype=tf.float32, name="alpha1")
        self._alpha2 = tf.Variable(1, trainable=training, dtype=tf.float32, name="alpha2")
        self._alpha3 = tf.Variable(1, trainable=training, dtype=tf.float32, name="alpha3")
        self._alpha4 = tf.Variable(1, trainable=training, dtype=tf.float32, name="alpha4")
        self._alpha5 = tf.Variable(1, trainable=training, dtype=tf.float32, name="alpha5")
        self._alpha6 = tf.Variable(1, trainable=training, dtype=tf.float32, name="alpha6")

        self._beta1 = tf.Variable(0, trainable=training, dtype=tf.float32, name="beta1")
        self._beta2 = tf.Variable(0, trainable=training, dtype=tf.float32, name="beta2")
        self._beta3 = tf.Variable(0, trainable=training, dtype=tf.float32, name="beta3")
        self._beta4 = tf.Variable(0, trainable=training, dtype=tf.float32, name="beta4")
        self._beta5 = tf.Variable(0, trainable=training, dtype=tf.float32, name="beta5")
        self._beta6 = tf.Variable(0, trainable=training, dtype=tf.float32, name="beta6")

        self._gamma1 = tf.Variable(gamma_0, trainable=training, dtype=tf.float32, name="gamma1")
        self._gamma2 = tf.Variable(gamma_0, trainable=training, dtype=tf.float32, name="gamma2")
        self._gamma3 = tf.Variable(gamma_0, trainable=training, dtype=tf.float32, name="gamma3")

        self._eta1 = tf.Variable(1.0, trainable=training, dtype=tf.float32, name="eta1")

    @property
    def eta1(self):
        return self._eta1

    @property
    def gamma1(self):
        return self._gamma1

    @property
    def gamma2(self):
        return self._gamma2

    @property
    def gamma3(self):
        return self._gamma3

    @property
    def alpha1(self):
        return self._alpha1

    @property
    def beta1(self):
        return self._beta1

    @property
    def alpha2(self):
        return self._alpha2

    @property
    def beta2(self):
        return self._beta2

    @property
    def alpha3(self):
        return self._alpha3

    @property
    def beta3(self):
        return self._beta3

    @property
    def alpha4(self):
        return self._alpha4

    @property
    def beta4(self):
        return self._beta4

    @property
    def alpha5(self):
        return self._alpha5

    @property
    def beta5(self):
        return self._beta5

    @property
    def alpha6(self):
        return self._alpha6

    @property
    def beta6(self):
        return self._beta6

    @tf.function()
    def call(self, batch_size, ebno_db):
        self.new_topology(batch_size)
        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)

        ######################################
        ## Transmitter
        no = ebnodb2no(ebno_db, num_bits_per_symbol, r)
        # Outer coding is only performed if not training
        b = self._binary_source([batch_size, n_ue, num_streams_per_tx, k])
        c = self._encoder(b)
        # Modulation
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        ######################################
        ## Channel
        # A batch of new channel realizations is sampled and applied at every inference
        no_ = expand_to_rank(no, tf.rank(x_rg))
        y, h = self._channel([x_rg, no_])

        ######################################
        ## Receiver
        if self._perfect_csi:
            h_hat = self._removeNulledSc(h)
            chan_est_var = tf.zeros(tf.shape(h_hat), dtype=tf.float32)  # No channel estimation error when perfect CSI knowledge is assumed
        else:
            h_hat, chan_est_var = self._ls_est([y, no])

        chan_est_var = self.eta1 * chan_est_var

        [llr_ch, _, G, y_MF, _, h_dt_desired_whitened] = self._detector0(
            [y, h_hat, chan_est_var, no, None, None, None, None])  # A_inv and G none => will calculate itself then

        [llr_dec, msg_vn] = self._LDPCDec0([llr_ch, None])
        llr_a_det = self.alpha1 * llr_dec - self.beta1 * llr_ch

        [llr_ch, _, _, _] = self._detector1([y_MF, h_dt_desired_whitened, chan_est_var, no, llr_a_det, G])
        llr_a_dec = self.alpha2 * llr_ch - self.beta2 * llr_a_det

        [llr_dec, msg_vn] = self._LDPCDec1([llr_a_dec, self.gamma1*msg_vn])
        llr_a_det = self.alpha3 * llr_dec - self.beta3 * llr_a_dec

        [llr_ch, _, _, _] = self._detector1([y_MF, h_dt_desired_whitened, chan_est_var, no, llr_a_det, G])
        llr_a_dec = self.alpha4 * llr_ch - self.beta4 * llr_a_det

        [llr_dec, msg_vn] = self._LDPCDec2([llr_a_dec, self.gamma2*msg_vn])
        llr_a_det = self.alpha5 * llr_dec - self.beta5 * llr_a_dec

        [llr_ch, _, _, _] = self._detector1([y_MF, h_dt_desired_whitened, chan_est_var, no, llr_a_det, G])
        llr_a_dec = self.alpha6 * llr_ch - self.beta6 * llr_a_det

        [b_hat, _] = self._LDPCDec3([llr_a_dec, self.gamma3*msg_vn])

        return self.computeLoss(b, c, b_hat)

# Low Complexity LoCo PIC models
# one iteration LoCo PIC
class OneIterModelLoCoPicChanEst(OneIterModelLmmseChanEst):
    def __init__(self, training=False, num_bp_iter=5, perfect_csi=False, loss_fun='BCE', two_variables=False,
                 err_var_term=error_var_term, REMCOM_ChanIdxComb=None):
        super().__init__(training=training, num_bp_iter=num_bp_iter, perfect_csi=perfect_csi, loss_fun=loss_fun,
                         REMCOM_ChanIdxComb=REMCOM_ChanIdxComb)

        self._detector0 = sisoLoCoPicDetector(rg_chan_est, sm, demapping_method=demapping_method,
                                              two_variables=two_variables, constellation=constellation,
                                              low_complexity=low_complexity, error_var_term=err_var_term,
                                              trainable=training)

    @tf.function()
    def call(self, batch_size, ebno_db):
        self.new_topology(batch_size)
        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)

        ######################################
        ## Transmitter
        no = ebnodb2no(ebno_db, num_bits_per_symbol, r)
        # Outer coding is only performed if not training
        b = self._binary_source([batch_size, n_ue, num_streams_per_tx, k])
        c = self._encoder(b)
        # Modulation
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        ######################################
        ## Channel
        # A batch of new channel realizations is sampled and applied at every inference
        no_ = expand_to_rank(no, tf.rank(x_rg))
        y, h = self._channel([x_rg, no_])

        ######################################
        ## Receiver
        if self._perfect_csi:
            h_hat = self._removeNulledSc(h)
            chan_est_var = tf.zeros(tf.shape(h_hat), dtype=tf.float32)
            # No channel estimation error when perfect CSI knowledge is assumed
        else:
            h_hat, chan_est_var = self._ls_est([y, no])

        chan_est_var = self.eta1 * chan_est_var

        [llr_ch, _, _, _, _, _] = self._detector0(
            [y, h_hat, chan_est_var, no, None, None, None, None])  # A_inv and G none => will calculate itself then
        b_hat = self._LDPCDec0(llr_ch)

        return self.computeLoss(b, c, b_hat)

# two iteration LoCo PIC
class TwoIterModelLoCoPicChanEst(TwoIterModelMmsePicChanEst):
    def __init__(self, training=False, num_bp_iter=5, perfect_csi=False, two_variables=False, loss_fun='BCE',
                 err_var_term=error_var_term, REMCOM_ChanIdxComb=None):
        super().__init__(training=training, num_bp_iter=num_bp_iter, perfect_csi=perfect_csi, loss_fun=loss_fun,
                         REMCOM_ChanIdxComb=REMCOM_ChanIdxComb)

        self._detector0 = sisoLoCoPicDetector(rg_chan_est, sm, demapping_method=demapping_method,
                                              two_variables=two_variables, constellation=constellation,
                                              low_complexity=low_complexity, error_var_term=err_var_term,
                                              trainable=training)
        self._detector1 = sisoLoCoPicDetector(rg_chan_est, sm, demapping_method=demapping_method,
                                              two_variables=two_variables, constellation=constellation,
                                              low_complexity=low_complexity, error_var_term=err_var_term,
                                              trainable=training, alpha0=0.25, data_carrying_whitened_inputs=True)

    @tf.function()
    def call(self, batch_size, ebno_db):
        self.new_topology(batch_size)
        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)

        ######################################
        ## Transmitter
        no = ebnodb2no(ebno_db, num_bits_per_symbol, r)
        # Outer coding is only performed if not training
        b = self._binary_source([batch_size, n_ue, num_streams_per_tx, k])
        c = self._encoder(b)
        # Modulation
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        ######################################
        ## Channel
        # A batch of new channel realizations is sampled and applied at every inference
        no_ = expand_to_rank(no, tf.rank(x_rg))
        y, h = self._channel([x_rg, no_])

        ######################################
        ## Receiver
        if self._perfect_csi:
            h_hat = self._removeNulledSc(h)
            chan_est_var = tf.zeros(tf.shape(h_hat), dtype=tf.float32)
            # No channel estimation error when perfect CSI knowledge is assumed
        else:
            h_hat, chan_est_var = self._ls_est([y, no])

        chan_est_var = self.eta1 * chan_est_var

        [llr_ch, A_inv, G, y_MF, mu_i, h_dt_desired_whitened] = self._detector0(
            [y, h_hat, chan_est_var, no, None, None, None, None])  # A_inv and G none => will calculate itself then

        [llr_dec, msg_vn] = self._LDPCDec0([llr_ch, None])
        llr_a_det = self.alpha1 * llr_dec - self.beta1 * llr_ch

        [llr_ch, _, _, _, _, _] = self._detector1([y_MF, h_dt_desired_whitened, chan_est_var, no, llr_a_det, A_inv, G, mu_i])
        llr_a_dec = self.alpha2 * llr_ch - self.beta2 * llr_a_det

        [b_hat, _]  = self._LDPCDec1([llr_a_dec, self.gamma1*msg_vn])

        return self.computeLoss(b, c, b_hat)

# three iteration LoCo PIC
class ThreeIterModelLoCoPicChanEst(ThreeIterModelMmsePicChanEst):
    def __init__(self, training=False, num_bp_iter=5, perfect_csi=False, two_variables=False, loss_fun='BCE',
                 err_var_term=error_var_term, REMCOM_ChanIdxComb=None):
        super().__init__(training=training, num_bp_iter=num_bp_iter, perfect_csi=perfect_csi, loss_fun=loss_fun,
                         REMCOM_ChanIdxComb=REMCOM_ChanIdxComb)

        self._detector0 = sisoLoCoPicDetector(rg_chan_est, sm, demapping_method=demapping_method, two_variables=two_variables,
                                                           constellation=constellation, low_complexity=low_complexity,
                                                           error_var_term=err_var_term, trainable=training)
        self._detector1 = sisoLoCoPicDetector(rg_chan_est, sm, demapping_method=demapping_method, two_variables=two_variables,
                                                           constellation=constellation, low_complexity=low_complexity,
                                                           error_var_term=err_var_term, trainable=training, alpha0=0.25,
                                                           data_carrying_whitened_inputs=True)
        self._detector2 = sisoLoCoPicDetector(rg_chan_est, sm, demapping_method=demapping_method, two_variables=two_variables,
                                                           constellation=constellation, low_complexity=low_complexity,
                                                           error_var_term=err_var_term, trainable=training, alpha0=0.25,
                                                           data_carrying_whitened_inputs=True)

    @tf.function()
    def call(self, batch_size, ebno_db):
        self.new_topology(batch_size)
        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)

        ######################################
        ## Transmitter
        no = ebnodb2no(ebno_db, num_bits_per_symbol, r)
        # Outer coding is only performed if not training
        b = self._binary_source([batch_size, n_ue, num_streams_per_tx, k])
        c = self._encoder(b)
        # Modulation
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        ######################################
        ## Channel
        # A batch of new channel realizations is sampled and applied at every inference
        no_ = expand_to_rank(no, tf.rank(x_rg))
        y, h = self._channel([x_rg, no_])

        ######################################
        ## Receiver
        if self._perfect_csi:
            h_hat = self._removeNulledSc(h)
            chan_est_var = tf.zeros(tf.shape(h_hat), dtype=tf.float32)  # No channel estimation error when perfect CSI knowledge is assumed
        else:
            h_hat, chan_est_var = self._ls_est([y, no])

        chan_est_var = self.eta1 * chan_est_var


        [llr_ch, A_inv, G, y_MF, mu_i, h_dt_desired_whitened] = self._detector0(
            [y, h_hat, chan_est_var, no, None, None, None, None])  # A_inv and G none => will calculate itself then

        [llr_dec, msg_vn] = self._LDPCDec0([llr_ch, None])
        llr_a_det = self.alpha1 * llr_dec - self.beta1 * llr_ch

        [llr_ch, _, _, _, _, _] = self._detector1([y_MF, h_dt_desired_whitened, chan_est_var, no, llr_a_det, A_inv, G, mu_i])
        llr_a_dec = self.alpha2 * llr_ch - self.beta2 * llr_a_det

        [llr_dec, msg_vn] = self._LDPCDec1([llr_a_dec, self.gamma1*msg_vn])
        llr_a_det = self.alpha3 * llr_dec - self.beta3 * llr_a_dec

        [llr_ch, _, _, _, _, _] = self._detector2([y_MF, h_dt_desired_whitened, chan_est_var, no, llr_a_det, A_inv, G, mu_i])
        llr_a_dec = self.alpha4 * llr_ch - self.beta4 * llr_a_det

        [b_hat, _] = self._LDPCDec2([llr_a_dec, self.gamma2*msg_vn])

        return self.computeLoss(b, c, b_hat)

# four iteration LoCo PIC
class FourIterModelLoCoPicChanEst(FourIterModelMmsePicChanEst):
    def __init__(self, training=False, num_bp_iter=5, perfect_csi=False, two_variables=False, loss_fun='BCE',
                 err_var_term=error_var_term, REMCOM_ChanIdxComb=None):
        super().__init__(training=training, num_bp_iter=num_bp_iter, perfect_csi=perfect_csi, loss_fun=loss_fun,
                         REMCOM_ChanIdxComb=REMCOM_ChanIdxComb)

        self._detector0 = sisoLoCoPicDetector(rg_chan_est, sm, demapping_method=demapping_method, two_variables=two_variables,
                                                           constellation=constellation, low_complexity=low_complexity,
                                                           error_var_term=err_var_term, trainable=training)
        self._detector1 = sisoLoCoPicDetector(rg_chan_est, sm, demapping_method=demapping_method, two_variables=two_variables,
                                                           constellation=constellation, low_complexity=low_complexity,
                                                           error_var_term=err_var_term, trainable=training, alpha0=0.25,
                                                           data_carrying_whitened_inputs=True)
        self._detector2 = sisoLoCoPicDetector(rg_chan_est, sm, demapping_method=demapping_method, two_variables=two_variables,
                                                           constellation=constellation, low_complexity=low_complexity,
                                                           error_var_term=err_var_term, trainable=training, alpha0=0.25,
                                                           data_carrying_whitened_inputs=True)
        self._detector3 = sisoLoCoPicDetector(rg_chan_est, sm, demapping_method=demapping_method, two_variables=two_variables,
                                                           constellation=constellation, low_complexity=low_complexity,
                                                           error_var_term=err_var_term, trainable=training, alpha0=0.25,
                                                           data_carrying_whitened_inputs=True)

    @tf.function()
    def call(self, batch_size, ebno_db):
        self.new_topology(batch_size)
        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)

        ######################################
        ## Transmitter
        no = ebnodb2no(ebno_db, num_bits_per_symbol, r)
        # Outer coding is only performed if not training
        b = self._binary_source([batch_size, n_ue, num_streams_per_tx, k])
        c = self._encoder(b)
        # Modulation
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        ######################################
        ## Channel
        # A batch of new channel realizations is sampled and applied at every inference
        no_ = expand_to_rank(no, tf.rank(x_rg))
        y, h = self._channel([x_rg, no_])

        ######################################
        ## Receiver
        if self._perfect_csi:
            h_hat = self._removeNulledSc(h)
            chan_est_var = tf.zeros(tf.shape(h_hat), dtype=tf.float32)
            # No channel estimation error when perfect CSI knowledge is assumed
        else:
            h_hat, chan_est_var = self._ls_est([y, no])

        chan_est_var = self.eta1 * chan_est_var

        [llr_ch, A_inv, G, y_MF, mu_i, h_dt_desired_whitened] = self._detector0(
            [y, h_hat, chan_est_var, no, None, None, None, None])  # A_inv and G none => will calculate itself then

        [llr_dec, msg_vn] = self._LDPCDec0([llr_ch, None])
        llr_a_det = self.alpha1 * llr_dec - self.beta1 * llr_ch

        [llr_ch, _, _, _, _, _] = self._detector1([y_MF, h_dt_desired_whitened, chan_est_var, no, llr_a_det, A_inv, G, mu_i])
        llr_a_dec = self.alpha2 * llr_ch - self.beta2 * llr_a_det

        [llr_dec, msg_vn] = self._LDPCDec1([llr_a_dec, self.gamma1*msg_vn])
        llr_a_det = self.alpha3 * llr_dec - self.beta3 * llr_a_dec

        [llr_ch, _, _, _, _, _] = self._detector2([y_MF, h_dt_desired_whitened, chan_est_var, no, llr_a_det, A_inv, G, mu_i])
        llr_a_dec = self.alpha4 * llr_ch - self.beta4 * llr_a_det

        [llr_dec, msg_vn] = self._LDPCDec2([llr_a_dec, self.gamma2*msg_vn])
        llr_a_det = self.alpha5 * llr_dec - self.beta5 * llr_a_dec

        [llr_ch, _, _, _, _, _] = self._detector3(
            [y_MF, h_dt_desired_whitened, chan_est_var, no, llr_a_det, A_inv, G, mu_i])
        llr_a_dec = self.alpha6 * llr_ch - self.beta6 * llr_a_det

        [b_hat, _] = self._LDPCDec3([llr_a_dec, self.gamma3*msg_vn])

        return self.computeLoss(b, c, b_hat)


#####################################################################################################################
## Train Models
#####################################################################################################################
# LoCo PIC Models
# I=1
one_iter_LoCo_training = OneIterModelLoCoPicChanEst(training=True, two_variables=False, num_bp_iter=num_ldpc_iter,
                                                    REMCOM_ChanIdxComb=chanIdxComb_training, perfect_csi=perfect_csi)
one_iter_LoCo_training_BLER = OneIterModelLoCoPicChanEst(training=True, two_variables=False, num_bp_iter=num_ldpc_iter,
                                                         REMCOM_ChanIdxComb=chanIdxComb_training, perfect_csi=perfect_csi,
                                                         loss_fun="BCE_LogSumExp_normalized")
train_model_BLER(one_iter_LoCo_training, one_iter_LoCo_training_BLER, ebno_db_min, ebno_db_max, num_pretraining_iterations,
                 num_BLER_training_iterations, training_batch_size)
save_weights(one_iter_LoCo_training_BLER, model_weights_path + "_bler_full_OneIterModelLowComplexityLmmseMfChanEst")

# I=2
two_iter_LoCo_training = TwoIterModelLoCoPicChanEst(training=True, two_variables=False, num_bp_iter=int(num_ldpc_iter / 2),
                                                    REMCOM_ChanIdxComb=chanIdxComb_training, perfect_csi=perfect_csi)
two_iter_LoCo_training_BLER = TwoIterModelLoCoPicChanEst(training=True, two_variables=False, num_bp_iter=int(num_ldpc_iter / 2),
                                                         REMCOM_ChanIdxComb=chanIdxComb_training, perfect_csi=perfect_csi,
                                                         loss_fun="BCE_LogSumExp_normalized")
train_model_BLER(two_iter_LoCo_training, two_iter_LoCo_training_BLER, ebno_db_min, ebno_db_max, num_pretraining_iterations,
                 num_BLER_training_iterations, training_batch_size)
save_weights(two_iter_LoCo_training_BLER, model_weights_path + "_bler_full_TwoIterModelLowComplexityLmmseMfChanEst")

# I=3
three_iter_LoCo_training = ThreeIterModelLoCoPicChanEst(training=True, two_variables=False, num_bp_iter=int(num_ldpc_iter / 3),
                                                        REMCOM_ChanIdxComb=chanIdxComb_training, perfect_csi=perfect_csi)
three_iter_LoCo_training_BLER = ThreeIterModelLoCoPicChanEst(training=True, two_variables=False, num_bp_iter=int(num_ldpc_iter / 3),
                                                             REMCOM_ChanIdxComb=chanIdxComb_training, perfect_csi=perfect_csi,
                                                             loss_fun="BCE_LogSumExp_normalized")
train_model_BLER(three_iter_LoCo_training, three_iter_LoCo_training_BLER, ebno_db_min, ebno_db_max, num_pretraining_iterations,
                 num_BLER_training_iterations, training_batch_size)
save_weights(three_iter_LoCo_training_BLER, model_weights_path + "_bler_full_ThreeIterModelLowComplexityLmmseMfChanEst")

# I=4
four_iter_LoCo_training = FourIterModelLoCoPicChanEst(training=True, two_variables=False, num_bp_iter=int(num_ldpc_iter / 4),
                                                      REMCOM_ChanIdxComb=chanIdxComb_training, perfect_csi=perfect_csi)
four_iter_LoCo_training_BLER = FourIterModelLoCoPicChanEst(training=True, two_variables=False, num_bp_iter=int(num_ldpc_iter / 4),
                                                           REMCOM_ChanIdxComb=chanIdxComb_training, perfect_csi=perfect_csi,
                                                           loss_fun="BCE_LogSumExp_normalized")
train_model_BLER(four_iter_LoCo_training, four_iter_LoCo_training_BLER, ebno_db_min, ebno_db_max, num_pretraining_iterations,
                 num_BLER_training_iterations, training_batch_size)
save_weights(four_iter_LoCo_training_BLER, model_weights_path + "_bler_full_FourIterModelLowComplexityLmmseMfChanEst")

# MMSE PIC
# I=2
two_iter_mmse_pic_training = TwoIterModelMmsePicChanEst(training=True, num_bp_iter=int(num_ldpc_iter/2),
                                                        REMCOM_ChanIdxComb=chanIdxComb_training, perfect_csi=perfect_csi)
two_iter_mmse_pic_training_BLER = TwoIterModelMmsePicChanEst(training=True, num_bp_iter=int(num_ldpc_iter/2),
                                                             REMCOM_ChanIdxComb=chanIdxComb_training, perfect_csi=perfect_csi,
                                                             loss_fun="BCE_LogSumExp_normalized")
train_model_BLER(two_iter_mmse_pic_training, two_iter_mmse_pic_training_BLER, ebno_db_min, ebno_db_max, num_pretraining_iterations,
                 num_BLER_training_iterations, training_batch_size)
save_weights(two_iter_mmse_pic_training_BLER, model_weights_path+"_bler_full_TwoIterModelMmsePicChanEst")

# I=3
three_iter_mmse_pic_training = ThreeIterModelMmsePicChanEst(training=True, num_bp_iter=int(num_ldpc_iter/3),
                                                            REMCOM_ChanIdxComb=chanIdxComb_training, perfect_csi=perfect_csi)
three_iter_mmse_pic_training_BLER = ThreeIterModelMmsePicChanEst(training=True, num_bp_iter=int(num_ldpc_iter/3),
                                                                 REMCOM_ChanIdxComb=chanIdxComb_training, perfect_csi=perfect_csi,
                                                                 loss_fun="BCE_LogSumExp_normalized")
train_model_BLER(three_iter_mmse_pic_training, three_iter_mmse_pic_training_BLER, ebno_db_min, ebno_db_max,
                 num_pretraining_iterations, num_BLER_training_iterations, training_batch_size)
save_weights(three_iter_mmse_pic_training_BLER, model_weights_path+"_bler_full_ThreeIterModelMmsePicChanEst")

# I=4
four_iter_mmse_pic_training = FourIterModelMmsePicChanEst(training=True, num_bp_iter=int(num_ldpc_iter/4),
                                                          REMCOM_ChanIdxComb=chanIdxComb_training, perfect_csi=perfect_csi)
four_iter_mmse_pic_training_BLER = FourIterModelMmsePicChanEst(training=True, num_bp_iter=int(num_ldpc_iter/4),
                                                               REMCOM_ChanIdxComb=chanIdxComb_training, perfect_csi=perfect_csi,
                                                               loss_fun="BCE_LogSumExp_normalized")
train_model_BLER(four_iter_mmse_pic_training, four_iter_mmse_pic_training_BLER, ebno_db_min, ebno_db_max,
                 num_pretraining_iterations, num_BLER_training_iterations, training_batch_size)
save_weights(four_iter_mmse_pic_training_BLER, model_weights_path+"_bler_full_FourIterModelMmsePicChanEst")

#####################################################################################################################
## Define Benchmark Models
#####################################################################################################################
# LMMSE Baseline Model
lmmse_baseline = LmmseBaselineModelChanEst(num_bp_iter=num_ldpc_iter, REMCOM_ChanIdxComb=chanIdxComb_validation,
                                           perfect_csi=perfect_csi)

# LoCo PIC Models
one_iter_LoCo_trained = OneIterModelLoCoPicChanEst(training=False, two_variables=False, num_bp_iter=num_ldpc_iter,
                                                   REMCOM_ChanIdxComb=chanIdxComb_validation, perfect_csi=perfect_csi)
one_iter_LoCo_trained = load_weights(one_iter_LoCo_trained, model_weights_path + "_bler_full_OneIterModelLowComplexityLmmseMfChanEst")
# I=2
two_iter_LoCo_trained = TwoIterModelLoCoPicChanEst(training=False, two_variables=False, num_bp_iter=int(num_ldpc_iter / 2),
                                                   REMCOM_ChanIdxComb=chanIdxComb_validation, perfect_csi=perfect_csi)
two_iter_LoCo_trained = load_weights(two_iter_LoCo_trained, model_weights_path + "_bler_full_TwoIterModelLowComplexityLmmseMfChanEst")
# I=3
three_iter_LoCo_trained = ThreeIterModelLoCoPicChanEst(training=False, two_variables=False, num_bp_iter=int(num_ldpc_iter / 3),
                                                       REMCOM_ChanIdxComb=chanIdxComb_validation, perfect_csi=perfect_csi)
three_iter_LoCo_trained = load_weights(three_iter_LoCo_trained, model_weights_path + "_bler_full_ThreeIterModelLowComplexityLmmseMfChanEst")
# I=4
four_iter_LoCo_trained = FourIterModelLoCoPicChanEst(training=False, two_variables=False, num_bp_iter=int(num_ldpc_iter / 4),
                                                     REMCOM_ChanIdxComb=chanIdxComb_validation, perfect_csi=perfect_csi)
four_iter_LoCo_trained = load_weights(four_iter_LoCo_trained, model_weights_path + "_bler_full_FourIterModelLowComplexityLmmseMfChanEst")

# MMSE PIC (default=untrained, classical idd with a constant number of LDPC iterations per IDD iteration)
# I=2
two_iter_mmse_pic_classicalIDD_default = TwoIterModelMmsePicChanEst(training=False, num_bp_iter=int(num_ldpc_iter),
                                                                    REMCOM_ChanIdxComb=chanIdxComb_validation, perfect_csi=perfect_csi)
two_iter_mmse_pic_default = TwoIterModelMmsePicChanEst(training=False, num_bp_iter=int(num_ldpc_iter/2),
                                                       REMCOM_ChanIdxComb=chanIdxComb_validation, perfect_csi=perfect_csi)
two_iter_mmse_pic_trained = TwoIterModelMmsePicChanEst(training=False, num_bp_iter=int(num_ldpc_iter/2),
                                                       REMCOM_ChanIdxComb=chanIdxComb_validation, perfect_csi=perfect_csi)
two_iter_mmse_pic_trained = load_weights(two_iter_mmse_pic_trained, model_weights_path+"_bler_full_TwoIterModelMmsePicChanEst")

# I=3
three_iter_mmse_pic_classicalIDD_default = ThreeIterModelMmsePicChanEst(training=False, num_bp_iter=int(num_ldpc_iter),
                                                                        REMCOM_ChanIdxComb=chanIdxComb_validation, perfect_csi=perfect_csi)
three_iter_mmse_pic_default = ThreeIterModelMmsePicChanEst(training=False, num_bp_iter=int(num_ldpc_iter/3),
                                                           REMCOM_ChanIdxComb=chanIdxComb_validation, perfect_csi=perfect_csi)
three_iter_mmse_pic_trained = ThreeIterModelMmsePicChanEst(training=False, num_bp_iter=int(num_ldpc_iter/3),
                                                           REMCOM_ChanIdxComb=chanIdxComb_validation, perfect_csi=perfect_csi)
three_iter_mmse_pic_trained = load_weights(three_iter_mmse_pic_trained, model_weights_path+"_bler_full_ThreeIterModelMmsePicChanEst")

# I=4
four_iter_mmse_pic_classicalIDD_default = FourIterModelMmsePicChanEst(training=False, num_bp_iter=int(num_ldpc_iter),
                                                                      REMCOM_ChanIdxComb=chanIdxComb_validation, perfect_csi=perfect_csi)
four_iter_mmse_pic_default = FourIterModelMmsePicChanEst(training=False, num_bp_iter=int(num_ldpc_iter/4),
                                                         REMCOM_ChanIdxComb=chanIdxComb_validation, perfect_csi=perfect_csi)
four_iter_mmse_pic_trained = FourIterModelMmsePicChanEst(training=False, num_bp_iter=int(num_ldpc_iter/4),
                                                         REMCOM_ChanIdxComb=chanIdxComb_validation, perfect_csi=perfect_csi)
four_iter_mmse_pic_trained = load_weights(four_iter_mmse_pic_trained, model_weights_path+"_bler_full_FourIterModelMmsePicChanEst")


#####################################################################################################################
## Benchmark Models
#####################################################################################################################
snr_range = np.arange(ebno_db_min - 2, ebno_db_max + 2 + stepsize, stepsize)

BLER = {'snr_range': snr_range}
BER = {'snr_range': snr_range}

title = case + str(num_iter) + '_Perfect-CSI=' + str(perfect_csi) + " " + str(n_bs_ant) + 'x' + str(n_ue) + \
        channel_model_str + '_N_MP_' + str(num_ldpc_iter) + '_Antenna_' + Antenna_Array

models = [lmmse_baseline, one_iter_LoCo_trained, two_iter_LoCo_trained, three_iter_LoCo_trained, four_iter_LoCo_trained,
          two_iter_mmse_pic_default, two_iter_mmse_pic_classicalIDD_default, two_iter_mmse_pic_trained,
          three_iter_mmse_pic_classicalIDD_default, three_iter_mmse_pic_default, three_iter_mmse_pic_trained,
          four_iter_mmse_pic_default, four_iter_mmse_pic_classicalIDD_default, four_iter_mmse_pic_trained]
model_names = ["lmmse_baseline", "one_iter_LoCo_trained", "two_iter_LoCo_trained", "three_iter_LoCo_trained",
               "four_iter_LoCo_trained", "two_iter_mmse_pic_default", "two_iter_mmse_pic_classicalIDD_default",
               "two_iter_mmse_pic_trained", "three_iter_mmse_pic_classicalIDD_default", "three_iter_mmse_pic_default",
               "three_iter_mmse_pic_trained", "four_iter_mmse_pic_default", "four_iter_mmse_pic_classicalIDD_default",
               "four_iter_mmse_pic_trained"]

for i in range(len(models)):
    ber, bler = sim_ber(models[i], ebno_dbs=snr_range, batch_size=batch_size,
                        num_target_block_errors=None, max_mc_iter=num_iter, early_stop=False)
    BLER[model_names[i]] = bler.numpy()
    BER[model_names[i]] = ber.numpy()

BLER["snr_range"] = snr_range
save_data(title + "_BLER", BLER, path="../results/")
save_data(title + "_BER", BER, path="../results/")

## Plotting simulation results
plt.figure(figsize=(10, 6))
for i in range(len(models)):
    plt.semilogy(BLER["snr_range"], BLER[model_names[i]], 'o-', c=f'C'+str(i), label=model_names[i])
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.ylim((1e-4, 1.0))
plt.legend()
plt.tight_layout()
plt.title(title+" BLER")
plt.show()

plt.figure(figsize=(10, 6))
for i in range(len(models)):
    plt.semilogy(BER["snr_range"], BER[model_names[i]], 'o-', c=f'C'+str(i), label=model_names[i])
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BER")
plt.grid(which="both")
plt.ylim((1e-4, 1.0))
plt.legend()
plt.tight_layout()
plt.title(title+" BLER")
plt.show()
