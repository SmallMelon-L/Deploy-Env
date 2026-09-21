#!/usr/bin/env python3
"""独立多轮缓存测试；依赖：pip install aiohttp transformers。

所有轮次均发送完整 messages 历史。输入长度是随机采样的 token 数，
解码成文本并由服务器重新分词后，实际长度以 usage.prompt_tokens 为准。
服务需支持 usage.prompt_tokens_details.cached_tokens；SGLang 需开启
--enable-cache-report。兼容零命中时省略 prompt_tokens_details 或返回 null。
默认不清理服务端缓存，不写结果文件。同一 seed 可能命中上次测试缓存。
"""

import argparse
import asyncio
import os
import random

# 只需要 tokenizer，无需加载 PyTorch 或初始化本机 GPU。
os.environ.setdefault("USE_TORCH", "0")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://172.16.33.30:5050/v1",
        help="OpenAI API 根地址，包含 /v1 或服务商的对应前缀",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="API Key；默认读取 OPENAI_API_KEY 环境变量",
    )
    parser.add_argument(
        "--flush-cache-url", help="可选：清缓存接口完整 URL，使用 POST；默认不调用"
    )
    parser.add_argument(
        "--model", default="DeepSeek-V4-Flash-0731", help="API 使用的模型名称"
    )
    parser.add_argument(
        "--tokenizer",
        default="/mnt/models/DeepSeek-V4-Flash-0731",
        help="本地 tokenizer 路径或 Hugging Face tokenizer 名称",
    )
    parser.add_argument(
        "--input-len", type=int, default=4096, help="第一轮随机输入 token 数"
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="每轮允许生成的最大 token 数（max_tokens）",
    )
    parser.add_argument(
        "--append-len", type=int, default=256, help="后续每轮追加的随机用户 token 数"
    )
    parser.add_argument("--num-rounds", type=int, default=5, help="每个会话的轮数")
    parser.add_argument(
        "--num-requests",
        type=int,
        default=128,
        help="独立会话数；总 HTTP 请求数 = 此值 × 轮数",
    )
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--seed", type=int, help="随机种子；默认每次使用新随机输入")
    args = parser.parse_args()
    if (
        min(
            args.input_len,
            args.output_len,
            args.num_rounds,
            args.num_requests,
            args.max_concurrency,
        )
        < 1
        or args.append_len < 0
    ):
        parser.error("长度、轮数、会话数和并发数必须为正数；append-len 可为 0")
    return args


def print_stats(label, count, prompt, cached, theoretical):
    print(
        f"{label:>6} {count:>8,} {prompt:>14,} {cached:>14,} "
        f"{theoretical:>14,} {cached / prompt:>12.2%} "
        f"{theoretical / prompt:>12.2%}",
        flush=True,
    )


async def benchmark(args):
    import aiohttp
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
    )
    special_ids = set(tokenizer.all_special_ids)
    vocabulary = sorted(set(tokenizer.get_vocab().values()) - special_ids)
    rng = random.Random(args.seed)

    def random_text(length):
        return tokenizer.decode(
            rng.choices(vocabulary, k=length), clean_up_tokenization_spaces=False
        )

    histories = [
        [{"role": "user", "content": random_text(args.input_len)}]
        for _ in range(args.num_requests)
    ]
    previous_totals = [0] * args.num_requests
    semaphore = asyncio.Semaphore(args.max_concurrency)
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    url = args.base_url.rstrip("/") + "/chat/completions"
    totals = [0, 0, 0]

    async with aiohttp.ClientSession(headers=headers) as session:
        if args.flush_cache_url:
            async with session.post(args.flush_cache_url) as response:
                response.raise_for_status()

        async def send(messages):
            payload = {
                "model": args.model,
                "messages": messages,
                "max_tokens": args.output_len,
                "temperature": 0,
                "stream": False,
            }
            async with semaphore:
                async with session.post(url, json=payload) as response:
                    if response.status >= 400:
                        body = await response.text()
                        raise RuntimeError(
                            f"HTTP {response.status} {response.reason}: {body[:4000]}"
                        )
                    return await response.json()

        print(
            "轮次     请求数     输入tokens     实际命中tokens   理论命中tokens"
            "   实际命中率   理论最高命中率",
            flush=True,
        )
        for turn in range(args.num_rounds):
            responses = await asyncio.gather(*(send(h) for h in histories))
            prompt = cached = theoretical = 0
            for i, response in enumerate(responses):
                usage = response["usage"]
                prompt += usage["prompt_tokens"]
                details = usage.get("prompt_tokens_details")
                cached += details["cached_tokens"] if details is not None else 0
                theoretical += previous_totals[i]
                previous_totals[i] = usage["total_tokens"]

                message = response["choices"][0]["message"]
                assistant = {"role": "assistant", "content": message["content"] or ""}
                if message.get("reasoning_content"):
                    assistant["reasoning_content"] = message["reasoning_content"]
                histories[i].append(assistant)
                if turn + 1 < args.num_rounds:
                    histories[i].append(
                        {"role": "user", "content": random_text(args.append_len)}
                    )

            print_stats(str(turn + 1), len(responses), prompt, cached, theoretical)
            totals = [a + b for a, b in zip(totals, (prompt, cached, theoretical))]

    print_stats("总计", args.num_requests * args.num_rounds, *totals)


if __name__ == "__main__":
    asyncio.run(benchmark(parse_args()))
