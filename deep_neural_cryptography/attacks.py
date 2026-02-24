from nn_aes import NeuralAESBase as NeuralAES
from nn_aes import encrypt_array_of_plaintexts
from utils import integer_to_bitvector
import numpy as np
import random


def get_bit_prediction(c0, c1, plain_bit):
    if np.allclose(c0, c1):
        guess = plain_bit
    else:
        guess = 1 - plain_bit
    return guess


def get_byte_prediction(c0, c1):
    # Find indices where c0 equals c1
    collisions_at = np.where(np.isclose(c0, c1).all(axis=1))[0].tolist()
    if len(collisions_at) == 1:  # one single collision, successful recovery
        guess = collisions_at[0]
    else:
        guess = -1
    return guess


def gen_pairs_sym(p, epsilon, neural_aes):
    zeroed_bit_i = (1 - np.eye(128)) * p  # 0 in position i of row i
    oned_bit_i = zeroed_bit_i + np.eye(128)  # 1 in position i of row i

    _0_minus_e = zeroed_bit_i - np.eye(128) * epsilon
    _0_plus_e = zeroed_bit_i + np.eye(128) * epsilon
    _1_minus_e = oned_bit_i - np.eye(128) * epsilon
    _1_plus_e = oned_bit_i + np.eye(128) * epsilon

    all_plaintexts = np.vstack([_0_minus_e, _0_plus_e, _1_minus_e, _1_plus_e])
    all_ciphertexts = encrypt_array_of_plaintexts(all_plaintexts, neural_aes)
    c0, c1, c0_control, c1_control = [all_ciphertexts[128 * i:128 * (i + 1)] for i in range(4)]
    return c0, c1, c0_control, c1_control


def gen_pairs_change(p, epsilon, neural_aes):
    zeroed_bit_i = (1 - np.eye(128)) * p  # 0 in position i of row i
    oned_bit_i = zeroed_bit_i + np.eye(128)  # 1 in position i of row i

    _0_plus_e = zeroed_bit_i + np.eye(128) * epsilon
    _1_minus_e = oned_bit_i - np.eye(128) * epsilon

    all_plaintexts = np.vstack([zeroed_bit_i, _0_plus_e, _1_minus_e, oned_bit_i])
    all_ciphertexts = encrypt_array_of_plaintexts(all_plaintexts, neural_aes)
    c0, c1, c0_control, c1_control = [all_ciphertexts[128 * i:128 * (i + 1)] for i in range(4)]
    return c0, c1, c0_control, c1_control


def gen_pairs_clip(p, epsilon, neural_aes):
    preimage_of_zero = np.unpackbits(np.uint8([82]).reshape(1, 1), axis=1)  # 1, 8
    candidate_key_values = np.unpackbits(np.arange(256, dtype=np.uint8).reshape(256, 1), axis=1)  # 256, 8
    x0 = (preimage_of_zero ^ candidate_key_values) * 1.0
    x1 = (preimage_of_zero ^ candidate_key_values) * 1.0
    x1[x1 == 0] += epsilon
    x1[x1 == 1] -= epsilon
    all_plaintexts = np.zeros((256 * 2 * 16, 128)) + p
    for byte_pos in range(16):
        all_plaintexts[byte_pos * 256: (byte_pos + 1) * 256, byte_pos * 8: (byte_pos + 1) * 8] = x0.copy()
        all_plaintexts[256 * 16 + byte_pos * 256: 256 * 16 + (byte_pos + 1) * 256,
        byte_pos * 8:(byte_pos + 1) * 8] = x1.copy()
    all_ciphertexts = encrypt_array_of_plaintexts(all_plaintexts, neural_aes)
    return all_ciphertexts[:256 * 16], all_ciphertexts[256 * 16:]


def run_attack_on_bytes(c_parameter, epsilon):
    gen_pairs = gen_pairs_clip

    # Instanciate target key and implementation
    key = random.getrandbits(128)
    key_bits = np.uint8(integer_to_bitvector(key)).reshape(1, 128)
    aes = NeuralAES(key, c_parameter=c_parameter)
    key_guess = [-1 for i in range(128)]

    # Prepare results dictionary
    attack_results = {'Key': key, 'Correct bits': 0, 'Detected errors': 0, 'Undetected errors': 0,
                      'Total pairs': 0, '# Base plaintexts': 0}

    # Initially, none of the bytes has been recovered
    bytes_left_to_recover = list(range(16))
    data_budget = 4096
    while len(bytes_left_to_recover) > 0:
        updated_bits_list = []
        base_plaintext = np.random.randint(2, size=128).reshape(1, 128)
        c0, c1 = gen_pairs(base_plaintext, epsilon, aes)
        for byte_pos in bytes_left_to_recover:
            guess = get_byte_prediction(c0[byte_pos * 256: (byte_pos + 1) * 256],
                                        c1[byte_pos * 256: (byte_pos + 1) * 256])
            if guess < 0:  # Unsuccessful recovery
                updated_bits_list.append(byte_pos)
                attack_results['Detected errors'] += 1
            else:
                bits = np.unpackbits(np.uint8(guess))
                key_guess[byte_pos * 8: (byte_pos + 1) * 8] = bits
                if np.all(key_guess[byte_pos * 8: (byte_pos + 1) * 8] == key_bits[0, byte_pos * 8: (byte_pos + 1) * 8]):
                    attack_results["Correct bits"] += 8
                else:
                    attack_results['Undetected errors'] += 1
        attack_results['Total pairs'] += (256 * len(bytes_left_to_recover))
        attack_results['# Base plaintexts'] += 1
        bytes_left_to_recover = updated_bits_list
        if attack_results['Total pairs'] >= data_budget:
            return attack_results
    attack_results["Key guess"] = key_guess
    return attack_results


def run_attack_on_bits(c_parameter, epsilon, verbose=False):
    if c_parameter == 1:  # Back to back ReLUs
        gen_pairs = gen_pairs_sym
    elif c_parameter < 1 and c_parameter > 0:  # Separated ReLUs
        gen_pairs = gen_pairs_change
    else:
        return "Error: c_parameter must be in ]0; 1]"

    # Instanciate target key and implementation
    key = random.getrandbits(128)
    key_bits = np.uint8(integer_to_bitvector(key)).reshape(1, 128)
    aes = NeuralAES(key, c_parameter=c_parameter)
    key_guess = [-1 for i in range(128)]

    # Prepare results dictionary
    attack_results = {'Key': key, 'Correct bits': 0, 'Detected errors': 0, 'Undetected errors': 0,
                      'Total pairs': 0, '# Base plaintexts': 0}

    # Initially, none of the bits has been recovered
    bits_left_to_recover = list(range(128))
    data_budget = 4096
    while len(bits_left_to_recover) > 0:
        updated_bits_list = []
        base_plaintext = np.random.randint(2, size=128).reshape(1, 128)
        c0, c1, c0_control, c1_control = gen_pairs(base_plaintext, epsilon, aes)
        for bit in bits_left_to_recover:
            guess = get_bit_prediction(c0[bit], c1[bit], 0)
            control_guess = get_bit_prediction(c0_control[bit], c1_control[bit], 1)
            if guess != control_guess:  # Contradicting predictions -> the recovery failed
                updated_bits_list.append(bit)
                attack_results['Detected errors'] += 1
                if verbose:
                    print(f"Error at bit {bit}, with {base_plaintext=}")
            else:
                key_guess[bit] = guess
                if guess == key_bits[0, bit]:
                    attack_results["Correct bits"] += 1
                else:
                    attack_results['Undetected errors'] += 1
        attack_results['Total pairs'] += (4 * len(bits_left_to_recover))
        attack_results['# Base plaintexts'] += 1
        bits_left_to_recover = updated_bits_list
        if attack_results['Total pairs'] >= data_budget:
            return attack_results
    attack_results["Key guess"] = key_guess
    return attack_results


def run_attack_on_clipped_implementation(epsilon_value=0.1, number_of_keys=1, verbose=False):
    c_param = 1
    attack_results = {'Successful recoveries': 0, 'Detected errors': 0, 'Undetected errors': 0, 'Total pairs': 0,
                      'Average # Base plaintexts': 0, 'individual_results': []}
    for i in range(number_of_keys):
        res = run_attack_on_bytes(c_param, epsilon_value)
        attack_results['Successful recoveries'] += 1 if res['Correct bits'] == 128 else 0
        attack_results['Average # Base plaintexts'] += res['# Base plaintexts'] / number_of_keys
        attack_results['Detected errors'] += res['Detected errors']
        attack_results['Undetected errors'] += res['Undetected errors']
        attack_results['Total pairs'] += res['Total pairs']
        attack_results['individual_results'].append(res)
        if verbose:
            print(res)
    return attack_results


def run_attack_on_back_to_back_relus(epsilon_value=0.1, number_of_keys=1, verbose=False):
    c_param = 1
    attack_results = {'Successful recoveries': 0, 'Detected errors': 0, 'Undetected errors': 0, 'Total pairs': 0,
                      'Average # Base plaintexts': 0, 'individual_results': []}
    for i in range(number_of_keys):
        res = run_attack_on_bits(c_param, epsilon_value, verbose)
        attack_results['Successful recoveries'] += 1 if res['Correct bits'] == 128 else 0
        attack_results['Average # Base plaintexts'] += res['# Base plaintexts'] / number_of_keys
        attack_results['Detected errors'] += res['Detected errors']
        attack_results['Undetected errors'] += res['Undetected errors']
        attack_results['Total pairs'] += res['Total pairs']
        attack_results['individual_results'].append(res)
        if verbose:
            print(res)
    return attack_results


def run_attack_on_separated_relus(epsilon_value=0.1, c_param = 0.25, number_of_keys=1, verbose=False):
    attack_results = {'Successful recoveries': 0, 'Detected errors': 0, 'Undetected errors': 0, 'Total pairs': 0,
                      'Average # Base plaintexts': 0, 'individual_results': []}
    for i in range(number_of_keys):
        res = run_attack_on_bits(c_param, epsilon_value)
        attack_results['Successful recoveries'] += 1 if res['Correct bits'] == 128 else 0
        attack_results['Average # Base plaintexts'] += res['# Base plaintexts'] / number_of_keys
        attack_results['Detected errors'] += res['Detected errors']
        attack_results['Undetected errors'] += res['Undetected errors']
        attack_results['Total pairs'] += res['Total pairs']
        attack_results['individual_results'].append(res)
        if verbose:
            print(res)
    return attack_results

if __name__ == '__main__':
    print("Back to back case")
    results = run_attack_on_back_to_back_relus(epsilon_value=0.01, number_of_keys=1, verbose=True)
    for key in results:
        if key != 'individual_results':
            print(key, results[key])

    print("Clipped case")
    results = run_attack_on_clipped_implementation(epsilon_value=0.01, number_of_keys=1, verbose=True)
    for key in results:
        if key != 'individual_results':
            print(key, results[key])
            
    print("Separated case")
    results = run_attack_on_separated_relus(epsilon_value=0.4, c_param=0.5, number_of_keys=1, verbose=True)
    for key in results:
        if key != 'individual_results':
            print(key, results[key])
