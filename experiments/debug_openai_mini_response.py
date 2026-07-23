"""Inspect gpt-5-mini Chat Completions response shape for the simulator call.

This is a tiny API diagnostic: it prints finish reason, content, and token usage,
but never prints credentials.
"""
import argparse
import json
import os

from dotenv import load_dotenv
from openai import OpenAI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max-completion-tokens", type=int, default=120)
    parser.add_argument("--reasoning-effort", default=None)
    args = parser.parse_args()

    load_dotenv()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    kwargs = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "Output only JSON."},
            {
                "role": "user",
                "content": (
                    'Output JSON {"activity":0.2,"quit":false,'
                    '"next_context":"A"} and no other text.'
                ),
            },
        ],
        "max_completion_tokens": args.max_completion_tokens,
        "response_format": {"type": "json_object"},
    }
    if args.reasoning_effort:
        kwargs["reasoning_effort"] = args.reasoning_effort

    response = client.chat.completions.create(**kwargs)
    payload = response.model_dump()
    choice = payload["choices"][0]
    message = choice.get("message", {})
    print(json.dumps({
        "model": payload.get("model"),
        "finish_reason": choice.get("finish_reason"),
        "content": message.get("content"),
        "refusal": message.get("refusal"),
        "usage": payload.get("usage"),
    }, indent=2))


if __name__ == "__main__":
    main()
