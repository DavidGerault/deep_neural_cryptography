import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import time

from utils import AES_key_schedule, build_xor_weights_and_biases, get_linear_mapping, \
    build_256_corners, MUL2, MUL3, SBOX, SBOX_INV, MUL14, MUL9, MUL13, MUL11, integer_to_bitvector, state_as_ints, \
    bytes_matrix_to_binary_states, build_xor4_weights_and_biases

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 # torch.float16 for benchmarking

def build_parameters_from_matrices(w, b, ws=None):
    wt = torch.tensor(w, dtype=DTYPE, requires_grad=False).to(DEVICE)
    bt = torch.tensor(b.flatten(), dtype=DTYPE, requires_grad=False).to(DEVICE)
    if ws is None:
        return wt, bt, None
    wst = torch.tensor(ws, dtype=DTYPE, requires_grad=False).to(DEVICE)
    return wt, bt, wst

class ClippingLayer(nn.Module):
    def __init__(self, epsilon=1/4):
        '''
        Implements the step function with parameter epsilon as defined in the paper.
        '''
        assert epsilon == 1/4, "Untested for other values; some values of epsilon, such as 1/3, cause numerical " \
                               "precision issues and result in an incorrect implementation of the cryptographic " \
                               "primitive. If changing epsilon, ensure that ClippingLayer(epsilon).forward(x) == x, " \
                               "for x a tensor containing 0s and 1s only."
        super().__init__()
        self.register_buffer("epsilon", torch.tensor(epsilon))

    def forward(self, x):
        return torch.div(
            torch.sub(
                F.relu(torch.sub(x, self.epsilon)),
                F.relu(torch.sub(x, 1 - self.epsilon))
            )
            , (1 - 2 * self.epsilon))


class SumBumpsLayer(nn.Module):
    def __init__(self, epsilon=1/4):
        '''
        Implements the sumbumps function with parameter epsilon as defined in the paper.
        '''
        assert epsilon == 1/4, "Untested for other values; some values of epsilon, such as 1/3, cause numerical " \
                               "precision issues and result in an incorrect implementation of the cryptographic " \
                               "primitive. If changing epsilon, ensure that SumBumpsLayer(epsilon).forward(x) == 0, " \
                               "for x a tensor containing 0s and 1s only."
        super().__init__()
        self.register_buffer("epsilon", torch.tensor(epsilon, dtype=DTYPE))

    def forward(self, x):
        return torch.mul(
            torch.add(
                F.relu(torch.sub(x, 1)),
                torch.sub(
                    torch.sub(F.relu(x),
                              F.relu(torch.sub(x, self.epsilon))
                              ),
                    F.relu(torch.sub(x, 1 - self.epsilon))
                )
            )
            , 1 / self.epsilon).sum(-1)


class NeuralAESImplementation(nn.Module):
    def __init__(self, secret_key, direction='Encryption', c_parameter=1.0, number_of_rounds=10, protected=False,
                 epsilon=1/4):
        '''
        Builds a number_of_rounds instance of the AES embedded with the key secret_key, given as an integer. The corresponding NN uses c_parameter in the corner functions.
        If protected is set to True, the protection measure described in the paper is applied, with the specified epsilon value.
        '''
        super().__init__()
        self.number_of_rounds = number_of_rounds
        self.c_parameter = c_parameter

        # Init ARK Parameters
        self.round_keys = AES_key_schedule(secret_key)  # TODO: add number of rounds
        self.round_key_tensors = torch.tensor(bytes_matrix_to_binary_states(self.round_keys), dtype=DTYPE)
        self.register_buffer("buff_round_keys", self.round_key_tensors[:, None, :, :])

        self.xorw, self.xorb, _ = build_parameters_from_matrices(*build_xor_weights_and_biases(c_parameter))
        self.xor4w, self.xor4b, _ = build_parameters_from_matrices(*build_xor4_weights_and_biases(c_parameter))
        if protected:
            self.sum_bumps = SumBumpsLayer(epsilon)
            self.step = ClippingLayer(epsilon)
            if direction == 'Encryption':
                self.fw_function = self.encrypt_protected
            elif direction == 'Decryption':
                self.fw_function = self.decrypt_protected
        else:
            if direction == 'Encryption':
                self.fw_function = self.encrypt
            elif direction == 'Decryption':
                self.fw_function = self.decrypt

    def ARK(self, state, round_key):
        # Possible optimisation: precompute self.xorw[0, 1] * round_key + self.xorb[0] and self.xorw[1, 1] * round_key + self.xorb[1]
        corner_0 = self.xorw[0, 0] * state + self.xorw[0, 1] * round_key + self.xorb[0]
        corner_1 = self.xorw[1, 0] * state + self.xorw[1, 1] * round_key + self.xorb[1]
        return F.relu(corner_0) + F.relu(corner_1)

    def apply_op(self, x, w0, b0, w1=None):
        orig_shape = x.shape[:-1]
        h0 = F.relu(F.linear(x, w0, b0))
        if w1 is None:
            return h0.sum(dim=-1)
        else:
            return F.linear(h0, w1)

    def encrypt(self, state_tensor):
        '''
        Applies NNAES to a binary tensor of plaintexts with dimensions batch_size x 128.
        '''
        state_tensor = state_tensor.view(-1, 4, 4, 8).transpose(1, 2)
        state_tensor = self.ARK(state_tensor, self.buff_round_keys[0])
        # disp_state(state_tensor, "AK0")
        for r in range(1, self.number_of_rounds):
            state_tensor = self.AES_round(state_tensor)
            # disp_state(state_tensor, f"R{r}")

            state_tensor = self.ARK(state_tensor, self.buff_round_keys[r])
            # disp_state(state_tensor, f"AK{r}")

        state_tensor = self.AES_round(state_tensor, final=True)
        state_tensor = self.ARK(state_tensor, self.buff_round_keys[self.number_of_rounds])
        return state_tensor.transpose(1, 2).reshape(-1, 128)

    def decrypt(self, state_tensor):
        '''
        Applies NNAES to a binary tensor of ciphertexts with dimensions batch_size x 128.
        '''
        state_tensor = state_tensor.view(-1, 4, 4, 8).transpose(1, 2)
        state_tensor = self.ARK(state_tensor, self.buff_round_keys[self.number_of_rounds])
        state_tensor = self.AES_round_inverse(state_tensor, final=True)
        for r in range(1, self.number_of_rounds):
            state_tensor = self.ARK(state_tensor, self.buff_round_keys[self.number_of_rounds - r])
            state_tensor = self.AES_round_inverse(state_tensor)
        state_tensor = self.ARK(state_tensor, self.buff_round_keys[0])
        return state_tensor.transpose(1, 2).reshape(-1, 128)

    def encrypt_protected(self, state_tensor):
        '''
        Applies NNAES to a binary tensor of plaintexts with dimensions batch_size x 128.
        '''
        sum_bumps = self.sum_bumps(state_tensor)
        state_tensor = self.step(state_tensor)
        state_tensor = self.encrypt(state_tensor)
        state_tensor = F.relu(torch.sub(state_tensor, sum_bumps[:, None]))
        return state_tensor

    def decrypt_protected(self, state_tensor):
        '''
        Applies NNAES to a binary tensor of plaintexts with dimensions batch_size x 128.
        '''
        sum_bumps = self.sum_bumps(state_tensor)
        state_tensor = self.step(state_tensor)
        state_tensor = self.decrypt(state_tensor)
        state_tensor = F.relu(torch.sub(state_tensor, sum_bumps[:, None]))
        return state_tensor

    def forward(self, state_tensor):
        '''
        Applies NNAES to a binary tensor of plaintexts with dimensions batch_size x 128.
        '''
        return self.fw_function(state_tensor)


class NeuralAESBase(NeuralAESImplementation):
    def __init__(self, secret_key, direction='Encryption', c_parameter=1.0, number_of_rounds=10, protected=False,
                 epsilon=None, T_table_based=False):
        '''
        Builds a number_of_rounds instance of the AES embedded with the key secret_key, given as an integer. The corresponding NN uses c_parameter in the corner functions.
        If protected is set to True, the protection measure described in the paper is applied, with the specified epsilon value.
        '''
        super().__init__(secret_key, direction, c_parameter, number_of_rounds, protected, epsilon)

        # Init parameters
        self.byte_op_w, self.byte_op_b, _ = build_parameters_from_matrices(*build_256_corners(self.c_parameter))
        if direction == 'Encryption':
            self.sbw_s = torch.tensor(get_linear_mapping(SBOX), dtype=DTYPE, requires_grad=False).to(DEVICE)
            self.m2w_s = torch.tensor(get_linear_mapping(MUL2), dtype=DTYPE, requires_grad=False).to(DEVICE)
            self.m3w_s = torch.tensor(get_linear_mapping(MUL3), dtype=DTYPE, requires_grad=False).to(DEVICE)
        elif direction == 'Decryption':
            self.sbiw_s = torch.tensor(get_linear_mapping(SBOX_INV), dtype=DTYPE, requires_grad=False).to(
                DEVICE)
            self.m14w_s = torch.tensor(get_linear_mapping(MUL14), dtype=DTYPE, requires_grad=False).to(DEVICE)
            self.m9w_s = torch.tensor(get_linear_mapping(MUL9), dtype=DTYPE, requires_grad=False).to(DEVICE)
            self.m13w_s = torch.tensor(get_linear_mapping(MUL13), dtype=DTYPE, requires_grad=False).to(DEVICE)
            self.m11w_s = torch.tensor(get_linear_mapping(MUL11), dtype=DTYPE, requires_grad=False).to(DEVICE)

    def apply_op(self, x, w0, b0, w1=None):
        orig_shape = x.shape[:-1]
        h0 = F.relu(F.linear(x, w0, b0))
        if w1 is None:
            return h0.sum(dim=-1)
        else:
            return F.linear(h0, w1)

    def SR(self, state):
        SR = torch.cat((
            state[:, 0:1],
            state[:, 1:2, [1, 2, 3, 0]],
            state[:, 2:3, [2, 3, 0, 1]],
            state[:, 3:, [3, 0, 1, 2]])
            , dim=1
        )
        return SR

    def SB(self, state):
        return self.apply_op(state, self.byte_op_w, self.byte_op_b, self.sbw_s)

    def MC(self, state):
        all_corners = F.relu(F.linear(state, self.byte_op_w, self.byte_op_b))
        M2 = F.linear(all_corners, self.m2w_s)
        M3 = F.linear(all_corners, self.m3w_s)
        X = torch.stack((M2, torch.cat((M3[:, 1:], M3[:, :1]), dim=1), torch.cat((state[:, 2:], state[:, :2]), dim=1),
                         torch.cat((state[:, 3:], state[:, :3]), dim=1)), dim=-1)
        return self.apply_op(X, self.xor4w, self.xor4b)

    def AES_round(self, state, final=False):
        # Shift Rows
        X = self.SR(state)
        # Sbox
        X = self.SB(X)
        # Mix Columns
        if not final:
            X = self.MC(X)
        return X

    def MC_inv(self, state):
        all_corners = F.relu(F.linear(state, self.byte_op_w, self.byte_op_b))
        M9 = F.linear(all_corners, self.m9w_s)
        M11 = F.linear(all_corners, self.m11w_s)
        M13 = F.linear(all_corners, self.m13w_s)
        M14 = F.linear(all_corners, self.m14w_s)
        MC = F.relu(
            F.linear(
                torch.stack((
                            M14, torch.cat((M11[:, 1:], M11[:, :1]), dim=1), torch.cat((M13[:, 2:], M13[:, :2]), dim=1),
                            torch.cat((M9[:, 3:], M9[:, :3]), dim=1)), dim=-1),
                self.xor4w, self.xor4b)
        ).sum(dim=-1)
        return MC

    def SB_inv(self, state):
        return self.apply_op(state, self.byte_op_w, self.byte_op_b, self.sbiw_s)

    def SR_inv(self, state):
        SR = torch.cat((
            state[:, 0:1],
            state[:, 1:2, [3, 0, 1, 2]],
            state[:, 2:3, [2, 3, 0, 1]],
            state[:, 3:, [1, 2, 3, 0]])
            , dim=1
        )
        return SR

    def AES_round_inverse(self, X, final=False):
        if not final:
            X = self.MC_inv(X)
        X = self.SB_inv(X)
        X = self.SR_inv(X)
        return X


class TTablesNeuralAES(NeuralAESImplementation):
    def __init__(self, secret_key, direction='Encryption', c_parameter=1.0, number_of_rounds=10, protected=False,
                 epsilon=None, T_table_based=False):
        '''
        Builds a number_of_rounds instance of the AES embedded with the key secret_key, given as an integer. The corresponding NN uses c_parameter in the corner functions.
        If protected is set to True, the protection measure described in the paper is applied, with the specified epsilon value.
        '''
        super().__init__(secret_key, direction, c_parameter, number_of_rounds, protected, epsilon)

        # Init parameters
        self.byte_op_w, self.byte_op_b, _ = build_parameters_from_matrices(*build_256_corners(self.c_parameter))
        SM2 = np.uint8(MUL2)[np.uint8(SBOX)]
        SM3 = np.uint8(MUL3)[np.uint8(SBOX)]

        if direction == 'Encryption':
            self.sw_s = torch.tensor(get_linear_mapping(SBOX), dtype=DTYPE, requires_grad=False).to(DEVICE)
            self.m2w_s = torch.tensor(get_linear_mapping(SM2), dtype=DTYPE, requires_grad=False).to(DEVICE)
            self.m3w_s = torch.tensor(get_linear_mapping(SM3), dtype=DTYPE, requires_grad=False).to(DEVICE)
        elif direction == 'Decryption':
            self.sbiw_s = torch.tensor(get_linear_mapping(SBOX_INV), dtype=DTYPE, requires_grad=False).to(
                DEVICE)
            self.m14w_s = torch.tensor(get_linear_mapping(MUL14), dtype=DTYPE, requires_grad=False).to(DEVICE)
            self.m9w_s = torch.tensor(get_linear_mapping(MUL9), dtype=DTYPE, requires_grad=False).to(DEVICE)
            self.m13w_s = torch.tensor(get_linear_mapping(MUL13), dtype=DTYPE, requires_grad=False).to(DEVICE)
            self.m11w_s = torch.tensor(get_linear_mapping(MUL11), dtype=DTYPE, requires_grad=False).to(DEVICE)

    def apply_op(self, x, w0, b0, w1=None):
        orig_shape = x.shape[:-1]
        h0 = F.relu(F.linear(x, w0, b0))
        if w1 is None:
            return h0.sum(dim=-1)
        else:
            return F.linear(h0, w1)

    def SR(self, state):
        SR = torch.cat((
            state[:, 0:1],
            state[:, 1:2, [1, 2, 3, 0]],
            state[:, 2:3, [2, 3, 0, 1]],
            state[:, 3:, [3, 0, 1, 2]])
            , dim=1
        )
        return SR

    def SB(self, state):
        return self.apply_op(state, self.byte_op_w, self.byte_op_b, self.sw_s)

    def SBMMC(self, state):
        all_corners = F.relu(F.linear(state, self.byte_op_w, self.byte_op_b))
        SB = F.linear(all_corners, self.sw_s)
        M2 = F.linear(all_corners, self.m2w_s)
        M3 = F.linear(all_corners, self.m3w_s)
        X = torch.stack((M2, torch.cat((M3[:, 1:], M3[:, :1]), dim=1), torch.cat((SB[:, 2:], SB[:, :2]), dim=1),
                         torch.cat((SB[:, 3:], SB[:, :3]), dim=1)), dim=-1)
        return self.apply_op(X, self.xor4w, self.xor4b)

    def AES_round(self, state, final=False):
        # Shift Rows
        X = self.SR(state)
        # Sbox
        if not final:
            X = self.SBMMC(X)
        else:
            X = self.SB(X)
        return X

    def MC_inv(self, state):
        all_corners = F.relu(F.linear(state, self.byte_op_w, self.byte_op_b))
        M9 = F.linear(all_corners, self.m9w_s)
        M11 = F.linear(all_corners, self.m11w_s)
        M13 = F.linear(all_corners, self.m13w_s)
        M14 = F.linear(all_corners, self.m14w_s)
        MC = F.relu(
            F.linear(
                torch.stack((
                            M14, torch.cat((M11[:, 1:], M11[:, :1]), dim=1), torch.cat((M13[:, 2:], M13[:, :2]), dim=1),
                            torch.cat((M9[:, 3:], M9[:, :3]), dim=1)), dim=-1),
                self.xor4w, self.xor4b)
        ).sum(dim=-1)
        return MC

    def SB_inv(self, state):
        return self.apply_op(state, self.byte_op_w, self.byte_op_b, self.sbiw_s)

    def SR_inv(self, state):
        SR = torch.cat((
            state[:, 0:1],
            state[:, 1:2, [3, 0, 1, 2]],
            state[:, 2:3, [2, 3, 0, 1]],
            state[:, 3:, [1, 2, 3, 0]])
            , dim=1
        )
        return SR

    def AES_round_inverse(self, X, final=False):
        if not final:
            X = self.MC_inv(X)
        X = self.SB_inv(X)
        X = self.SR_inv(X)
        return X


def test_aes_testvectors(protected=False, epsilon=None):
    # Tests the neural implementation against test vectors
    # AES128 test vector (from https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197-upd1.pdf)
    p_fips = [0x3243f6a8885a308d313198a2e0370734]
    k_fips = 0x2b7e151628aed2a6abf7158809cf4f3c
    c_fips = [0x3925841d02dc09fbdc118597196a0b32]
    # First 5 KATs from D1 in https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Algorithm-Validation-Program/documents/aes/AESAVS.pdf
    k_aesavs = 0
    p_aesavs = [0x80000000000000000000000000000000, 0xc0000000000000000000000000000000,
                0xe0000000000000000000000000000000, 0xf0000000000000000000000000000000,
                0xf8000000000000000000000000000000]
    c_aesavs = [0x3ad78e726c1ec02b7ebfe92b23d9ec34, 0xaae5939c8efdf2f04e60b9fe7117b2c2,
                0xf031d4d74f5dcbf39daaf8ca3af6e527, 0x96d9fd5cc4f07441727df0f33e401a36,
                0x30ccdb044646d7e1f3ccea3dca08b8c0]
    for model_type in [NeuralAESBase, TTablesNeuralAES]:
        for c_parameter in [1.0, 0.5]:
            aes_enc = model_type(k_fips, c_parameter=c_parameter, protected=protected, epsilon=epsilon)
            ct = encrypt_list_of_plaintexts(p_fips, aes_enc)
            for i in range(1):
                assert ct[i] == c_fips[i]
            aes_enc = model_type(k_aesavs, c_parameter=c_parameter, protected=protected, epsilon=epsilon)
            ct = encrypt_list_of_plaintexts(p_aesavs, aes_enc)
            for i in range(5):
                assert ct[i] == c_aesavs[i]

        if protected:
            print(f"(Protected) Neural AES Test vector verified (forward direction, {model_type.__name__})")
        else:
            print(f"Neural AES Test vector verified (forward direction, {model_type.__name__})")

        for c_parameter in [1.0, 0.5]:
            aes_dec = model_type(k_fips, direction='Decryption', c_parameter=c_parameter, protected=protected,
                                 epsilon=epsilon)
            ct = encrypt_list_of_plaintexts(c_fips, aes_dec)
            for i in range(1):
                assert ct[i] == p_fips[i]
            aes_dec = model_type(k_aesavs, direction='Decryption', c_parameter=c_parameter, protected=protected,
                                 epsilon=epsilon)
            ct = encrypt_list_of_plaintexts(c_aesavs, aes_dec)
            for i in range(5):
                assert ct[i] == p_aesavs[i]

        if protected:
            print(f"(Protected) Neural AES Test vector verified (backward direction, {model_type.__name__})")
        else:
            print(f"Neural AES Test vector verified (backward direction, {model_type.__name__})")


def test_protected_aes_testvectors(epsilon):
    # Tests the neural implementation against test vectors
    test_aes_testvectors(protected=True, epsilon=epsilon)


def test_protected_aes_functionality(num_samples=10 ** 3, epsilon=1 / 4):
    for model_type in [NeuralAESBase, TTablesNeuralAES]:
        torch.cuda.empty_cache()
        k = 0x2b7e151628aed2a6abf7158809cf4f3c
        model = model_type(k, c_parameter=0.5, protected=True, epsilon=epsilon).to(DEVICE)
        model.eval()
        p = torch.randint(0, 2, (num_samples, 128), dtype=DTYPE).to(DEVICE)

        # Test 0: Integer inputs => integer outputs
        print("Test 0: Protected implementation should generate 0/1 output for 0/1 inputs...", end=" ")
        with torch.inference_mode():
            c = model(p).detach().cpu().numpy()
        assert np.all(np.logical_or(c == 0, c == 1))
        print(f"Passed ({model_type.__name__})")

        # Test 1: One input single bit toggled towards 0.5  -> expect relu(c-sumbumps)
        print("Test 1: Protected implementation should return relu(c-sumbumps) when at least one input is not 0/1...",
              end=" ")
        p1 = p.clone().to(DEVICE)
        p1[:, 3] = torch.abs(p[:, 3] - torch.div(torch.rand(num_samples, dtype=DTYPE), 2).to(DEVICE))
        expected_diff = SumBumpsLayer(epsilon=1 / 4).to(DEVICE)(p1)
        with torch.inference_mode():
            c = model(p)
            c0 = F.relu(torch.sub(c, expected_diff[:, None])).detach().cpu().numpy()
            c1 = model(p1).detach().cpu().numpy()
        assert np.all(c1 == c0)

        # Test 2: One input single bit away from 0.5  -> expect relu(c-sumbumps)
        p[:, 3] = 1
        p1 = p.clone().to(DEVICE)
        p1[:, 3] = p[:, 3] + torch.div(torch.rand(num_samples, dtype=DTYPE), 2).to(DEVICE)
        expected_diff = SumBumpsLayer(epsilon=1 / 4).to(DEVICE)(p1)

        with torch.inference_mode():
            c = model(p)
            c0 = F.relu(torch.sub(c, expected_diff[:, None])).detach().cpu().numpy()
            c1 = model(p1).detach().cpu().numpy()
        assert np.all(c1 == c0)

        # Test 3: Random small toggles (smaller than 1/4)  -> expect relu(c-sumbumps)
        p1 = p.clone().to(DEVICE)
        p1 = p + torch.div(torch.sub(torch.rand((num_samples, 128), dtype=DTYPE), 0.5), 2).to(DEVICE)
        expected_diff = SumBumpsLayer(epsilon=1 / 4).to(DEVICE)(p1)

        with torch.inference_mode():
            c = model(p)
            c0 = F.relu(torch.sub(c, expected_diff[:, None])).detach().cpu().numpy()
            c1 = model(p1).detach().cpu().numpy()
        assert np.all(c1 == c0)
        print(f"Passed ({model_type.__name__})")


def encrypt_array_of_plaintexts(plaintexts, model, dtype=DTYPE):
    '''
    Encrypt a numpy array of plaintexts of shape (n, 128), and returns the output of the model as an array.
    '''
    model = model.to(DEVICE)
    model.eval()
    plaintexts_tensor = torch.tensor(plaintexts, dtype=dtype).to(DEVICE)
    state_tensor = plaintexts_tensor.clone()
    with torch.inference_mode():#, torch.autocast(device_type=DEVICE, dtype=DTYPE):
        ciphertexts_tensor = model(state_tensor)
    return ciphertexts_tensor.detach().cpu().numpy()


def encrypt_list_of_plaintexts(plaintexts, model, dtype=DTYPE):
    '''
    Encrypt a list of plaintexts given in integer form, and returns the result as a list of integers.
    '''
    state = [integer_to_bitvector(x) for x in plaintexts]
    model = model.to(DEVICE)
    model.eval()
    state_tensor = torch.tensor(state, dtype=dtype).to(DEVICE)
    with torch.inference_mode():#, torch.autocast(device_type=DEVICE, dtype=DTYPE):
        state_tensor = model(state_tensor)
    res = state_as_ints(state_tensor)  # [bitvector_to_integer(c) for c in state_tensor.detach().cpu().numpy()]
    return res


def run_benchmark_models(models, batch_sizes=[1, 10000], seed=42, num_repetitions=100, warmup_runs=50):
    """
    Benchmark pre-instantiated models.

    Args:
        models: List of (model, model_name) tuples
        batch_sizes: List of batch sizes to test
        seed: Random seed
        num_repetitions: Number of benchmark iterations
        warmup_runs: Number of warmup runs

    Returns:
        List of benchmark results
    """
    random.seed(seed)
    results = []

    for model, model_name in models:
        model = model.to(DEVICE).eval()
        if torch.cuda.is_available():
            model = torch.compile(model, mode="reduce-overhead", fullgraph=True, dynamic=False)
        for batch_size in batch_sizes:
            with torch.inference_mode(): #, torch.autocast(device_type='cuda', dtype=torch.float16):
                data = torch.randint(0, 2, (batch_size, 128), device=DEVICE, dtype=DTYPE)
                for _ in range(warmup_runs):
                    _ = model(data)
                data = torch.randint(0, 2, (batch_size, 128), device=DEVICE, dtype=DTYPE)
                times = []

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    # Create CUDA events for timing
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                for _ in range(num_repetitions):
                    if torch.cuda.is_available():
                        start_event.record()
                        _ = model(data)
                        end_event.record()
                        end_event.synchronize()
                        times.append(start_event.elapsed_time(end_event))
                    else:
                        start = time.time()
                        _ = model(data)
                        times.append(time.time() - start)

                total_ms = sum(times)
                total_samples = num_repetitions * batch_size
                samples_per_second = 1000 * total_samples / total_ms
                mean_time_ms = np.mean(times)
                blocks_per_second = (batch_size * 1000) / mean_time_ms

                # Calculate throughput in Mbps (each AES block is 128 bits)
                bits_per_second = blocks_per_second * 128
                mbps = bits_per_second / 1_000_000

                # Calculate Mbps for each individual timing to get standard deviation
                individual_mbps = []
                for time_ms in times:
                    individual_blocks_per_second = (batch_size * 1000) / time_ms
                    individual_bits_per_second = individual_blocks_per_second * 128
                    individual_mbps.append(individual_bits_per_second / 1_000_000)
                mbps_std = np.std(individual_mbps)

                result = {}
                result['model'] = model_name
                result['batch_size'] = batch_size
                result['num_batches'] = num_repetitions
                result['total_samples'] = total_samples
                result['total_time_ms'] = round(total_ms, 3)
                result['mean_time_ms'] = round(mean_time_ms, 3)
                result['min_time_ms'] = round(np.min(times), 3)
                result['max_time_ms'] = round(np.max(times), 3)
                result['median_time_ms'] = round(np.median(times), 3)
                result['blocks_per_second'] = int(blocks_per_second)
                result['samples_per_second'] = int(samples_per_second)
                result['mbps'] = round(mbps, 2)
                result['mbps_std'] = round(mbps_std, 2)

                results.append(result)
                if batch_size == 10000:  # Show Mbps for 10k case
                    print(
                        f"{model_name} - Batch {batch_size}: {mean_time_ms:.3f}ms ({blocks_per_second:,.0f} blocks/sec, {mbps:.2f} ± {mbps_std:.2f} Mbps)")
                else:
                    print(
                        f"{model_name} - Batch {batch_size}: {mean_time_ms:.3f}ms ({blocks_per_second:,.0f} blocks/sec)")

    return results


def main():
    os.environ.pop("TORCH_LOGS", None)           # env var has precedence over the API
    import torch._logging as tlog
    tlog.set_logs(inductor=0, dynamo=0, aot=0)   # turn off compiler component logs
    print("===================================================================")
    print("Running testvectors...")
    test_aes_testvectors()

    import pandas as pd

    torch._dynamo.config.cache_size_limit = 256  # added to solve FailOnRecompileLimitHit

    print("===================================================================")
    print("Running Comprehensive Neural AES Benchmark...")

    # Test key
    key = 0x2b7e151628aed2a6abf7158809cf4f3c
    batch_sizes = [1, 10000]  # Test both small and large batch sizes

    # Create models for comprehensive comparison
    models_to_test = []

    # All model types - Encryption
    models_to_test.append((NeuralAESBase(secret_key=key, direction='Encryption'), "NeuralAESBase_Encryption"))
    models_to_test.append((TTablesNeuralAES(secret_key=key, direction='Encryption'), "TTablesNeuralAES_Encryption"))
    models_to_test.append((TTablesNeuralAES(secret_key=key, direction='Encryption', protected=True, epsilon=1 / 4),
                           "TTablesNeuralAES_Protected_Encryption"))

    # All model types - Decryption
    models_to_test.append((NeuralAESBase(secret_key=key, direction='Decryption'), "NeuralAESBase_Decryption"))
    models_to_test.append((NeuralAESBase(secret_key=key, direction='Decryption', protected=True, epsilon=1 / 4),
                           "NeuralAESBase_Protected_Decryption"))

    # Run benchmarks
    all_results = run_benchmark_models(models_to_test, batch_sizes=batch_sizes, num_repetitions=100, warmup_runs=50)

    # Create comprehensive results table
    print("\n" + "=" * 100)
    print("COMPREHENSIVE NEURAL AES BENCHMARK RESULTS")
    print("=" * 100)

    # Create a formatted table
    table_data = []

    model_types = ['NeuralAESBase', 'TTablesNeuralAES']
    directions = ['Encryption','Decryption']

    for model_type in model_types:
        for direction in directions:
            row = {'Model': model_type, 'Direction': direction}

            for batch_size in batch_sizes:
                # Find matching result
                if model_type == 'TTablesNeuralAES_Protected':
                    model_name = f"TTablesNeuralAES_Protected_{direction}"
                else:
                    model_name = f"{model_type}_{direction}"

                result = next((r for r in all_results
                               if r['model'] == model_name and r['batch_size'] == batch_size), None)

                if result:
                    row[f'Batch_{batch_size}_ms'] = result['mean_time_ms']
                    row[f'Batch_{batch_size}_blocks_per_sec'] = result['blocks_per_second']
                    if batch_size == 10000:  # Add Mbps for 10k case
                        row[f'Batch_{batch_size}_mbps'] = result['mbps']
                        row[f'Batch_{batch_size}_mbps_std'] = result['mbps_std']
                else:
                    row[f'Batch_{batch_size}_ms'] = 'N/A'
                    row[f'Batch_{batch_size}_blocks_per_sec'] = 'N/A'
                    if batch_size == 10000:
                        row[f'Batch_{batch_size}_mbps'] = 'N/A'
                        row[f'Batch_{batch_size}_mbps_std'] = 'N/A'

            table_data.append(row)

    # Create and display the table
    df_table = pd.DataFrame(table_data)

    # Reorder columns for better readability
    column_order = ['Model', 'Direction', 'Batch_1_ms', 'Batch_1_blocks_per_sec',
                    'Batch_10000_ms', 'Batch_10000_blocks_per_sec', 'Batch_10000_mbps', 'Batch_10000_mbps_std']
    df_table = df_table[column_order]

    # Rename columns for better display
    df_table.columns = ['Model', 'Direction', '1 Block (ms)', '1 Block (bl/s)',
                        '10k Blocks (ms)', '10k Blocks (bl/s)', '10k Blocks (Mbps)', '10k Blocks (Mbps StdDev)']

    print(df_table.to_string(index=False, float_format='{:.3f}'.format))
    print("\n" + "=" * 100)
    print("KEY INSIGHTS:")
    print("=" * 100)

    # Protection overhead analysis
    print(f"\nProtection Overhead Analysis:")
    for direction in directions:
        for batch_size in batch_sizes:
            unprotected = next((r for r in all_results
                                if r['model'] == f'TTablesNeuralAES_{direction}' and r['batch_size'] == batch_size),
                               None)
            protected = next((r for r in all_results
                              if r['model'] == f'TTablesNeuralAES_Protected_{direction}' and r[
                                  'batch_size'] == batch_size), None)

            if unprotected and protected:
                overhead_percent = ((protected['mean_time_ms'] - unprotected['mean_time_ms']) / unprotected[
                    'mean_time_ms']) * 100
                print(f"  {direction} (Batch {batch_size}): {overhead_percent:+.1f}% overhead")


if __name__ == '__main__':
    main()
