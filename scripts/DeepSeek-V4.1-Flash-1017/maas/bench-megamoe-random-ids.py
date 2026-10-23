#!/usr/bin/env python3
"""Run SGLang's serving benchmark with the archived Humming workload matrix.

Run using the deployment venv on the server. Outputs retain every attempt;
cache-hit attempts are replaced with independently randomized token prefixes.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request


def unique_prefix_cli():
    import numpy as np
    from sglang.benchmark.datasets.random import RandomDataset
    from sglang.benchmark.serving import cli_main

    original_load = RandomDataset.load

    def load(self, tokenizer, model_id=None):
        rows = original_load(self, tokenizer, model_id)
        for row in rows:
            assert isinstance(row.prompt, list)
            n = min(256, len(row.prompt))
            row.prompt[:n] = np.random.randint(128, tokenizer.vocab_size, n).tolist()
        print('Independent random 256-token prefixes enabled to prevent cache collisions.')
        return rows

    RandomDataset.load = load
    sys.argv.remove('--unique-prefix-cli')
    cli_main()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:5050')
    parser.add_argument('--short-only', action='store_true')
    parser.add_argument('--gpu-count', type=int, default=8)
    parser.add_argument('--ep-size', type=int, default=None)
    parser.add_argument('--a2a-backend', default='megamoe')
    parser.add_argument('--reference-gpu-count', type=int, default=8)
    parser.add_argument('--per-gpu-matrix', action='store_true')
    parser.add_argument('--speculative', choices=['none', 'DSPARK'], default='none')
    parser.add_argument('--reference-summary', help='Reuse accepted seeds and prefix mode.')
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    base = args.base_url.rstrip('/')

    def snapshot(endpoint, filename):
        with urllib.request.urlopen(base + endpoint, timeout=60) as response:
            data = response.read()
        (out / filename).write_bytes(data)
        return data

    info = json.loads(snapshot('/get_server_info', 'server-info-before.json'))
    assert info['tp_size'] == info['dp_size'] == args.gpu_count
    assert info['ep_size'] == (args.ep_size if args.ep_size is not None else args.gpu_count)
    assert info['moe_a2a_backend'] == args.a2a_backend
    assert info['speculative_algorithm'] == (None if args.speculative == 'none' else args.speculative)
    if args.speculative == 'DSPARK':
        assert info['speculative_dspark_block_size'] == 5
    matrix = [(1024, 256, c, max(8, 2*c)) for c in [1,4,8,16,32,64,128,256]]
    if args.per_gpu_matrix:
        matrix = [(1024, 256, c * args.gpu_count, max(8, 2*c*args.gpu_count)) for c in [1,2,4,8,16,32]]
    if not args.short_only:
        matrix += [(8192, 256, c, max(4, 2*c)) for c in [1,8,32,64]]
        matrix += [(32768, 128, c, max(4, 2*c)) for c in [1,8,16]]
    reference = {}
    if args.reference_summary:
        reference = {(r['input_len'], r['output_len'], r['concurrency']): r
                     for r in json.loads(Path(args.reference_summary).read_text())}
    (out / 'matrix.json').write_text(json.dumps(matrix))
    stop = threading.Event()

    def sample_gpu():
        with (out / 'gpu.csv').open('w') as f:
            f.write('timestamp,index,memory.used_MiB,utilization.gpu_pct,power.draw_W\n')
            while not stop.is_set():
                r = subprocess.run(['/usr/local/nvidia/bin/nvidia-smi',
                    '--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True)
                f.write(r.stdout)
                f.flush()
                stop.wait(2)

    sampler = threading.Thread(target=sample_gpu, daemon=True)
    sampler.start()
    summary = []
    try:
        for case_id, (inp, output, concurrency, count) in enumerate(matrix):
            name = f'in{inp}-out{output}-c{concurrency}'
            ref_c = concurrency * args.reference_gpu_count // args.gpu_count
            ref = reference.get((inp, output, ref_c), {})
            for attempt in range(3):
                label = name if attempt == 0 else f'{name}-unique{attempt}'
                seed = ref.get('seed', 211000 + case_id * 100) + attempt
                unique_prefix = attempt > 0 or 'unique' in ref.get('artifact', '')
                cmd = [sys.executable, '-m', 'sglang.benchmark.serving']
                if unique_prefix:
                    cmd = [sys.executable, str(Path(__file__).resolve()), '--unique-prefix-cli']
                cmd += ['--backend','sglang','--base-url',base,
                    '--dataset-name','random-ids',
                    '--model','/mnt/models2/DeepSeek-V4.1-Flash-1017',
                    '--tokenizer','/mnt/models2/DeepSeek-V4.1-Flash-1017',
                    '--served-model-name','DeepSeek-V4.1-Flash-1017',
                    '--random-input-len',str(inp),'--random-output-len',str(output),
                    '--random-range-ratio','1','--num-prompts',str(count),
                    '--max-concurrency',str(concurrency),'--seed',str(seed),
                    '--tokenize-prompt','--warmup-requests','0','--flush-cache',
                    '--cache-report','--output-details','--disable-tqdm',
                    '--output-file',str(out / f'{label}.jsonl')]
                snapshot('/metrics', f'{label}.metrics-before.txt')
                print(time.strftime('%F %T'), 'START', label, 'requests', count, flush=True)
                (out / f'{label}.command.json').write_text(json.dumps(cmd))
                with (out / f'{label}.log').open('w') as log:
                    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=1800)
                snapshot('/metrics', f'{label}.metrics-after.txt')
                result = json.loads((out / f'{label}.jsonl').read_text().splitlines()[-1])
                assert result['completed'] == count, (label, result['completed'])
                assert result['total_input_tokens'] == inp * count
                assert result['total_output_tokens'] == output * count
                cached = result['cache_report']['total_cached_tokens']
                print(time.strftime('%F %T'), 'DONE', label,
                      'input_tps', round(result['input_throughput'],2),
                      'output_tps', round(result['output_throughput'],2),
                      'cached_tokens', cached, flush=True)
                if cached:
                    continue
                keys = ['completed','duration','total_input_tokens','total_output_tokens',
                        'input_throughput','output_throughput','mean_ttft_ms','p95_ttft_ms',
                        'mean_tpot_ms','p95_tpot_ms','mean_e2e_latency_ms','cache_report',
                        'accept_length']
                summary.append(dict(input_len=inp, output_len=output,
                    concurrency=concurrency, artifact=label, seed=seed,
                    unique_prefix=unique_prefix,
                    **{k:result[k] for k in keys}))
                (out / 'summary.json').write_text(json.dumps(summary,indent=2))
                break
            else:
                raise RuntimeError(f'Could not obtain zero-cache result: {name}')
        snapshot('/get_server_info', 'server-info-after.json')
        print('ALL DONE',len(summary),'cases',flush=True)
    finally:
        stop.set()
        sampler.join(timeout=5)


if __name__ == '__main__':
    if '--unique-prefix-cli' in sys.argv:
        unique_prefix_cli()
    else:
        main()
