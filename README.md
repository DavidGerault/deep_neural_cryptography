# Deep Neural Cryptography
This repository contains the companion code for the `Deep Neural Cryptography` paper by David Gérault, Anna Hambitzer, Eyal Ronen and Adi Shamir, published at EUROCRYPT 2026 and available at [https://eprint.iacr.org/2025/288](https://eprint.iacr.org/2025/288). In particular, it contains the implementation of the AES block cipher as a DNN, as well as the attacks and benchmarks described in the paper.

## Setup
The project was implemented using the package versions described in environment.yml; it can be used as follows (tested with miniconda 26.1.1):

```bash
conda env create -f environment.yml
conda activate ec26_dnc
```

The implementations from nn_aes.py are compatible with CPUs, but the benchmark scripts are intended to be ran on a CUDA-compatible GPU. 
The code was tested on an NVIDIA A100 GPU.

## Neural AES Implementations
The file `deep_neural_cryptography/nn_aes.py`contains natural implementations of the AES as a Deep Neural Network, as described in our paper, `Deep Neural Cryptography`.
In particular, it defines the classes `NeuralAESBase`, which is a simple implementation, and `TTablesNeuralAES`, an optimized variant using TTables.

Both classes inherit from `NeuralAESImplementation`; they receive the secret key (as an integer), and parameters `direction` ('Encryption' or 'Decryption') and `protected`, a boolean to activate or deactivate the protection mechanism defined in the paper. They expose a `forward` method to process tensors of n plaintexts, with shape (n, 128).

## Neural AES Attacks
The `deep_neural_cryptography/attacks.py` file demonstrates our attacks on 3 cases: the natural implementations, the sanitized case, and the separated ReLUs scenario.

```
Running it outputs the results of key recovery for a random key as follows:
Back to back case
{'Key': 78884133467633640336330480148916510692, 'Correct bits': 128, 'Detected errors': 0, 'Undetected errors': 0, 'Total pairs': 512, '# Base plaintexts': 1, 'Key guess': [0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0]}
Successful recoveries 1
Detected errors 0
Undetected errors 0
Total pairs 512
Average # Base plaintexts 1.0
Clipped case
{'Key': 269721437486363770801514603065261099071, 'Correct bits': 128, 'Detected errors': 0, 'Undetected errors': 0, 'Total pairs': 4096, '# Base plaintexts': 1}
Successful recoveries 1
Detected errors 0
Undetected errors 0
Total pairs 4096
Average # Base plaintexts 1.0
Separated case
{'Key': 162291372610786584835673567970074470779, 'Correct bits': 128, 'Detected errors': 0, 'Undetected errors': 0, 'Total pairs': 512, '# Base plaintexts': 1, 'Key guess': [0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1]}
Successful recoveries 1
Detected errors 0
Undetected errors 0
Total pairs 512
Average # Base plaintexts 1.0
```

## Neural AES Benchmarking 

### Neural AES Throughput

Running `deep_neural_cryptography/nn_aes.py` will provide the following benchmark of the throughputs: 

```
====================================================================================================
COMPREHENSIVE NEURAL AES BENCHMARK RESULTS
====================================================================================================
           Model  Direction 1 Block (ms) 1 Block (bl/s) 10k Blocks (ms) 10k Blocks (bl/s) 10k Blocks (Mbps) 10k Blocks (Mbps StdDev)
   NeuralAESBase Encryption        0.446           2242           4.109           2433642           311.510                    2.710
   NeuralAESBase Decryption        0.485           2060           5.415           1846835           236.390                    1.130
TTablesNeuralAES Encryption        0.354           2824           3.471           2880630           368.720                    2.160
TTablesNeuralAES Decryption          N/A            N/A             N/A               N/A               N/A                      N/A

====================================================================================================
KEY INSIGHTS:
====================================================================================================

Protection Overhead Analysis:
  Encryption (Batch 1): +2.8% overhead
  Encryption (Batch 10000): +0.3% overhead
```

### Runtimes versus Conventional DNNs

Running `deep_neural_cryptography/benchmark_vs_dnns.py` will provide the runtimes of `TTablesNeuralAES_Protected` in comparison to various DNNs: 

```
| Model         | Category     | Params | Time   | vs AES   | Status    | Compiled |
|---------------|--------------|--------|--------|----------|-----------|----------|
| Protected AES | Cryptography | N/A    | 11ms   | 1x       | Measured  | N/A      |
| GPT-2 Large   | Medium LLM   | 774M   | 539ms  | 49x      | Measured  | Yes      |
```
and resulting impacts on an inference pipeline:
```
Protected AES decryption time: 11.1ms for 10k blocks

| Model         | LLM Time | Total Time | AES Overhead | Impact    |
|---------------|----------|------------|--------------|-----------|
| GPT-2 Large   | 539ms    | 550ms      | 2.0  %       | Negligible |
```

Running `deep_neural_cryptography/benchmark_vs_ddns_2k_tokens.py` will provide a comparison on 2,048 tokens (an equivalent of around 512 AES blocks):

```
==========================================================================================

All models processed 2,048 tokens (≈512 AES blocks)
AES time for equivalent blocks: 6.18ms

| Model         | Params | Batches | Time    | Throughput  | vs AES  | Efficiency |
|---------------|--------|---------|---------|-------------|---------|------------|
| GPT-2 Large   | 774M   |       2 | 30.1ms  | 68K tok/s   | 5x      | 87871      |
| Llama 3.1 8B  | 7.5B   |       1 | 1.79s   | 1K tok/s    | 290x    | 152        |
```
