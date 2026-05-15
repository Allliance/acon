"""FastAPI server wrapping LLMLingua-2 prompt compression.

POST /compress  body: {"prompt": "...", "rate": 0.5}
                resp: {"compressed_prompt": "..."}
"""

from __future__ import annotations

import argparse
import os

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from llmlingua import PromptCompressor


parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=9999)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument(
    "--model",
    type=str,
    default="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    help="LLMLingua model. The default is LLMLingua-2 (fast, encoder-only).",
)
parser.add_argument("--use_llmlingua2", action="store_true", default=True)
parser.add_argument("--device_map", type=str, default="cuda")
args = parser.parse_args()


print(f"[llmlingua-server] loading {args.model} ...", flush=True)
compressor = PromptCompressor(
    model_name=args.model,
    use_llmlingua2=args.use_llmlingua2,
    device_map=args.device_map,
)
print(f"[llmlingua-server] ready on {args.host}:{args.port}", flush=True)


class CompressRequest(BaseModel):
    prompt: str
    rate: float = 0.5


app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/compress")
def compress(req: CompressRequest):
    result = compressor.compress_prompt(req.prompt, rate=req.rate)
    return {
        "compressed_prompt": result.get("compressed_prompt", ""),
        "origin_tokens": result.get("origin_tokens"),
        "compressed_tokens": result.get("compressed_tokens"),
        "ratio": result.get("ratio"),
        "rate": result.get("rate"),
    }


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
