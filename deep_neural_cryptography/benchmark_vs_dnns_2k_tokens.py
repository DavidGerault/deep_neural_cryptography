#!/usr/bin/env python3
"""
Standardized benchmark of TTableNeuralAES versus LLMs using 2,048 tokens.

This version tests all models on a fixed 2,048 tokens to represent
realistic encrypted document processing (2-3 pages of business documents).
Each model uses appropriate batching to achieve this token count.
"""

import torch
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import gc
from transformers import AutoTokenizer, AutoModel, LlamaConfig, LlamaModel
import warnings
warnings.filterwarnings("ignore")

from transformers.utils import logging
logging.set_verbosity_error()

# Import neural AES implementation
from nn_aes import TTablesNeuralAES

class Benchmark2KTokens:
    def __init__(self, gpu_id=1):
        # Use specific GPU
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            self.device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(gpu_id)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {self.device}")

        # Standard test configuration
        self.target_tokens = 2048  # Fixed for all models
        self.bytes_per_aes_block = 16

        # Test key for AES
        self.aes_key = 0x2b7e151628aed2a6abf7158809cf4f3c

        # Get GPU memory if available
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            gpu_mem = torch.cuda.get_device_properties(current_device).total_memory / 1e9
            print(f"GPU {current_device} Memory: {gpu_mem:.1f} GB")

        # Model-specific batching strategies to achieve 2,048 tokens
        self.model_configs = {
            "distilbert-base-uncased": {
                "max_tokens": 512,
                "batches_needed": 4,  # 4 × 512 = 2,048
                "category": "Small LLM"
            },
            "bert-base-uncased": {
                "max_tokens": 512,
                "batches_needed": 4,  # 4 × 512 = 2,048
                "category": "Small LLM"
            },
            "gpt2": {
                "max_tokens": 1024,
                "batches_needed": 2,  # 2 × 1,024 = 2,048
                "category": "Small LLM"
            },
            "gpt2-medium": {
                "max_tokens": 1024,
                "batches_needed": 2,  # 2 × 1,024 = 2,048
                "category": "Medium LLM"
            },
            "gpt2-large": {
                "max_tokens": 1024,
                "batches_needed": 2,  # 2 × 1,024 = 2,048
                "category": "Medium LLM"
            },
            "llama-8b-random": {
                "max_tokens": 2048,  # Use 2048 directly (well within 4K limit)
                "batches_needed": 1,  # 1 × 2,048 = 2,048
                "category": "Large LLM"
            }
        }

        # Dictionary to store measured AES times
        self.aes_measured_times = {}

    def measure_aes_performance(self, num_blocks: int, warmup_runs: int = 50, num_runs: int = 100) -> Dict:
        """Measure actual TTableNeuralAES performance for a given number of blocks."""
        print(f"\n  Measuring TTableNeuralAES for {num_blocks:,} blocks...")

        # Create model with protection enabled
        model = TTablesNeuralAES(
            secret_key=self.aes_key,
            direction='Encryption',
            protected=True,
            epsilon=1/4
        ).to(self.device).eval()

        compilation_success = False
        try:
                model = torch.compile(model, mode="reduce-overhead", fullgraph=True, dynamic=False)
                compilation_success = True
                print(f"    Model compiled successfully")
        except Exception as e:
                print(f"    Model compilation failed: {str(e)[:50]}...")

        # Create random input - generate random 128-bit integers and convert to bit vectors
        from utils import integer_to_bitvector
        import random

        # Generate random 128-bit integers using Python's random
        plaintext_ints = [random.randint(0, 2**128 - 1) for _ in range(num_blocks)]

        # Convert to bit vectors
        plaintext_bits = [integer_to_bitvector(x) for x in plaintext_ints]

        # Convert to tensor
        plaintext = torch.tensor(plaintext_bits, dtype=torch.float16).to(self.device)

        # Warmup
        print(f"    Warming up with {warmup_runs} runs...")
        with torch.inference_mode():#, torch.autocast(device_type='cuda', dtype=torch.float16):
            for _ in range(warmup_runs):
                _ = model(plaintext)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

        # Benchmark
        print(f"    Running benchmark with {num_runs} runs...")
        times = []

        with torch.inference_mode():#, torch.autocast(device_type='cuda', dtype=torch.float16):
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    start_event.record()
                    _ = model(plaintext)
                    end_event.record()
                    end_event.synchronize()
                    times.append(start_event.elapsed_time(end_event) / 1000.0)
                else:
                    start = time.time()
                    _ = model(plaintext)
                    times.append(time.time() - start)

        avg_time = np.mean(times)
        std_time = np.std(times)
        blocks_per_sec = num_blocks / avg_time

        print(f"    Results: {avg_time*1000:.3f}ms for {num_blocks:,} blocks ({blocks_per_sec:.0f} blocks/sec)")

        result = {
            "num_blocks": num_blocks,
            "time": avg_time,
            "std_dev": std_time,
            "blocks_per_second": blocks_per_sec,
            "compiled": compilation_success
        }

        # Store in dictionary
        self.aes_measured_times[num_blocks] = avg_time

        # Cleanup
        del model, plaintext
        torch.cuda.empty_cache()
        gc.collect()

        return result

    def calculate_aes_equivalent(self, num_tokens: int, tokenizer_type: str = "wordpiece") -> Tuple[int, float]:
        """Calculate how many AES blocks are equivalent to the given number of tokens."""
        # Average chars per token based on tokenizer type
        chars_per_token = {
            "wordpiece": 4.5,  # BERT family
            "bpe": 4.0,        # GPT family
            "sentencepiece": 4.2  # T5, LLaMA
        }

        avg_chars = chars_per_token.get(tokenizer_type, 4.2)
        total_bytes = num_tokens * avg_chars
        aes_blocks = total_bytes / self.bytes_per_aes_block

        return int(aes_blocks), None  # We'll use measured times instead

    def get_tokenizer_type(self, model_name: str) -> str:
        """Determine tokenizer type based on model name."""
        if "bert" in model_name.lower():
            return "wordpiece"
        elif "gpt" in model_name.lower():
            return "bpe"
        else:
            return "sentencepiece"

    def benchmark_model_2k_tokens(self, model_name: str, display_name: str) -> Optional[Dict]:
        """Benchmark a model using 2,048 tokens with appropriate batching."""
        try:
            print(f"\n  Loading {display_name}...")

            # Get model-specific configuration
            config = self.model_configs.get(model_name, {
                "max_tokens": 512,
                "batches_needed": 4,
                "category": "Unknown"
            })

            max_tokens = config["max_tokens"]
            batches_needed = config["batches_needed"]
            actual_total_tokens = max_tokens * batches_needed

            print(f"    Configuration: {batches_needed} batches × {max_tokens} tokens = {actual_total_tokens:,} total")

            # Calculate AES equivalent
            tokenizer_type = self.get_tokenizer_type(model_name)
            aes_blocks, _ = self.calculate_aes_equivalent(actual_total_tokens, tokenizer_type)
            print(f"    Processing equivalent of {aes_blocks:,} AES blocks")

            # Special handling for Llama 8B
            if model_name == "llama-8b-random":
                print("    Creating Llama 8B with random weights...")
                llama_config = LlamaConfig(
                    hidden_size=4096,
                    intermediate_size=14336,
                    num_hidden_layers=32,
                    num_attention_heads=32,
                    num_key_value_heads=8,
                    vocab_size=128256,
                    max_position_embeddings=131072,
                    rope_theta=500000.0,
                )
                model = LlamaModel(llama_config).to(self.device).eval()
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name, dtype=torch.float16).to(self.device).eval()

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Get parameter count
            params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"    Parameters: {params:.0f}M")

            # Try torch.compile
            compilation_success = False
            try:
                model = torch.compile(model, mode="reduce-overhead", fullgraph=True, dynamic=False)
                compilation_success = True
                print(f"    Model compiled successfully")
            except Exception as e:
                print(f"    Model compilation failed: {str(e)[:50]}...")

            # Create test input at model's max length
            test_text = "The artificial intelligence system processes encrypted data through neural networks. " * 1000
            inputs = tokenizer(test_text, max_length=max_tokens, truncation=True,
                             padding="max_length", return_tensors="pt").to(self.device)

            actual_tokens_per_batch = inputs['input_ids'].shape[1]
            print(f"    Actual tokens per batch: {actual_tokens_per_batch:,}")

            # GPU warmup
            warmup_runs = 20
            print(f"    Warming up with {warmup_runs} runs...")
            with torch.inference_mode():
                for _ in range(warmup_runs):
                    _ = model(**inputs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

            # Benchmark - process the required number of batches
            print(f"    Running benchmark with {batches_needed} batches...")
            times = []
            num_runs = 50

            with torch.inference_mode():
                for _ in range(num_runs):
                    if torch.cuda.is_available():
                        start_event.record()
                        # Process required number of batches
                        for _ in range(batches_needed):
                            _ = model(**inputs)
                        end_event.record()
                        end_event.synchronize()
                        times.append(start_event.elapsed_time(end_event) / 1000.0)
                    else:
                        start = time.time()
                        for _ in range(batches_needed):
                            _ = model(**inputs)
                        times.append(time.time() - start)

            avg_time = np.mean(times)
            std_time = np.std(times)
            total_tokens_processed = actual_tokens_per_batch * batches_needed
            tokens_per_sec = total_tokens_processed / avg_time

            print(f"    Results: {avg_time*1000:.3f}ms for {total_tokens_processed:,} tokens ({tokens_per_sec:.0f} tokens/sec)")

            result = {
                "model_name": display_name,
                "category": config["category"],
                "parameters_millions": params,
                "time_for_2k_tokens": avg_time,
                "std_dev": std_time,
                "compiled": compilation_success,
                "batches_processed": batches_needed,
                "tokens_per_batch": actual_tokens_per_batch,
                "total_tokens": total_tokens_processed,
                "aes_blocks_equivalent": aes_blocks,
                "tokens_per_second": tokens_per_sec
            }

            # Cleanup
            del model, tokenizer, inputs
            torch.cuda.empty_cache()
            gc.collect()

            return result

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    ❌ {display_name}: Out of GPU memory")
                torch.cuda.empty_cache()
                gc.collect()
                return None
            else:
                print(f"    ❌ {display_name}: Runtime error - {str(e)[:100]}...")
                torch.cuda.empty_cache()
                gc.collect()
                return None
        except Exception as e:
            print(f"    ❌ {display_name}: Error - {str(e)[:100]}...")
            torch.cuda.empty_cache()
            gc.collect()
            return None

    def run_benchmark(self):
        """Run the complete 2K token benchmark."""
        print("\n" + "=" * 70)
        print("STANDARDIZED BENCHMARKING: TTableNeuralAES vs LLMs (2,048 tokens)")
        print("=" * 70)
        print(f"\nAll models tested on {self.target_tokens:,} tokens")
        print("(representing 2-3 pages of business documents)")

        # Calculate AES equivalent for 2048 tokens
        aes_blocks_2k, _ = self.calculate_aes_equivalent(self.target_tokens, "bpe")  # Use average
        print(f"Equivalent AES blocks: {aes_blocks_2k:,}")

        # First, measure AES performance for the equivalent blocks
        print("\n--- Measuring AES Performance ---")
        aes_result = self.measure_aes_performance(aes_blocks_2k)
        aes_time_2k = aes_result["time"]

        results = []

        # Test LLMs with 2K token configuration
        print("\n--- Testing LLMs with 2,048 Tokens ---")

        llm_models = [
            ("distilbert-base-uncased", "DistilBERT"),
            ("bert-base-uncased", "BERT-base"),
            ("gpt2", "GPT-2"),
            ("gpt2-medium", "GPT-2 Medium"),
            ("gpt2-large", "GPT-2 Large"),
            ("llama-8b-random", "Llama 3.1 8B"),
        ]

        for model_name, display_name in llm_models:
            result = self.benchmark_model_2k_tokens(model_name, display_name)
            if result:
                results.append(result)
            else:
                print(f"  Failed to benchmark {display_name}")
                if "large" in model_name.lower() or "llama" in model_name.lower():
                    print("  Stopping due to likely memory constraints")
                    break

        # Save raw results
        output_file = "benchmark_2k_tokens.json"
        with open(output_file, 'w') as f:
            # Add AES measurements to results
            results_with_aes = {
                "target_tokens": self.target_tokens,
                "aes_blocks_equivalent": aes_blocks_2k,
                "aes_time": aes_time_2k,
                "aes_measurements": {
                    str(blocks): {
                        "time": time,
                        "blocks_per_second": blocks / time
                    } for blocks, time in self.aes_measured_times.items()
                },
                "llm_results": results
            }
            json.dump(results_with_aes, f, indent=2)

        print(f"\nResults saved to: {output_file}")

        # Display results table
        self.display_results_table(results, aes_time_2k, aes_blocks_2k)

        return results

    def display_results_table(self, results: List[Dict], aes_time_2k: float, aes_blocks_2k: int):
        """Display a comprehensive results table for 2K token benchmark."""
        print("\n" + "=" * 90)
        print("2K TOKEN BENCHMARK RESULTS")
        print("=" * 90)

        # Header
        print(f"\nAll models processed {self.target_tokens:,} tokens (≈{aes_blocks_2k:,} AES blocks)")
        print(f"AES time for equivalent blocks: {aes_time_2k*1000:.2f}ms")

        print("\n| Model         | Params | Batches | Time    | Throughput  | vs AES  | Efficiency |")
        print("|---------------|--------|---------|---------|-------------|---------|------------|")

        for r in results:
            model_name = r["model_name"][:13]

            # Format parameters
            params = f"{r['parameters_millions']:.0f}M" if r["parameters_millions"] < 1000 else f"{r['parameters_millions']/1000:.1f}B"

            # Get values
            batches = r["batches_processed"]
            llm_time = r["time_for_2k_tokens"]
            throughput = r["tokens_per_second"]

            # Calculate slowdown vs AES
            vs_aes = llm_time / aes_time_2k

            # Calculate efficiency (tokens per second per billion parameters)
            efficiency = throughput / (r["parameters_millions"] / 1000)  # tokens/sec/B params

            # Format values
            time_str = f"{llm_time*1000:.1f}ms" if llm_time < 1 else f"{llm_time:.2f}s"
            throughput_str = f"{throughput/1000:.0f}K tok/s" if throughput < 1000000 else f"{throughput/1000000:.1f}M tok/s"
            vs_aes_str = f"{vs_aes:.0f}x" if vs_aes < 1000 else f"{vs_aes:,.0f}x"
            efficiency_str = f"{efficiency:.0f}" if efficiency >= 1 else f"{efficiency:.1f}"

            print(f"| {model_name:<13} | {params:<6} | {batches:>7} | {time_str:<7} | {throughput_str:<11} | {vs_aes_str:<7} | {efficiency_str:<10} |")

        # Print explanations
        print("\n" + "=" * 90)
        print("COLUMN EXPLANATIONS:")
        print("=" * 90)
        print(f"\n- All models process exactly {self.target_tokens:,} tokens")
        print("- Batches: Number of forward passes needed (depends on model's context limit)")
        print("- Time: Total time to process 2K tokens")
        print("- Throughput: Tokens processed per second")
        print("- vs AES: Slowdown factor compared to TTableNeuralAES")
        print("- Efficiency: Throughput per billion parameters (higher is better)")

        print("\n" + "=" * 90)
        print("KEY INSIGHTS:")
        print("=" * 90)
        print("\n1. Standardized 2K token test ensures fair comparison across architectures")
        print("2. Batching strategy reveals computational efficiency differences")
        print("3. Parameter efficiency shows performance per model complexity")
        print("4. Results represent realistic encrypted document processing scenarios")

def create_results_table_from_json(json_file: str = "benchmark_2k_tokens.json"):
    """Load results from JSON and create a formatted table."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        results = data["llm_results"]
        aes_time_2k = data["aes_time"]
        aes_blocks_2k = data["aes_blocks_equivalent"]
        target_tokens = data["target_tokens"]

        print("\n" + "=" * 90)
        print(f"RESULTS TABLE FROM: {json_file}")
        print("=" * 90)

        # Display table
        benchmark = Benchmark2KTokens()
        benchmark.target_tokens = target_tokens
        benchmark.display_results_table(results, aes_time_2k, aes_blocks_2k)

    except FileNotFoundError:
        print(f"Error: File {json_file} not found")
    except Exception as e:
        print(f"Error loading results: {str(e)}")

def main():
    """Run the 2K token benchmark."""
    benchmark = Benchmark2KTokens()
    benchmark.run_benchmark()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--table":
        # Just display the table from existing results
        json_file = sys.argv[2] if len(sys.argv) > 2 else "benchmark_2k_tokens.json"
        create_results_table_from_json(json_file)
    else:
        # Run full benchmark
        main()
