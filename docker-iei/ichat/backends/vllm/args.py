
import argparse
from argparse import Namespace
from typing import List

from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser


def parse_vllm_args(backend_argv: List[str]) -> Namespace:
    """
    Parses vLLM-specific arguments, including unified arguments from iChat.
    """
    # 1. Create a parser for vLLM to define all its arguments
    # The make_arg_parser function from vllm now expects an existing parser.
    parser = FlexibleArgumentParser(
        prog="vllm",
        description="vLLM-backed iChat server for OpenAI-compatible API.")
    make_arg_parser(parser)

    # 2. Add iChat's unified arguments to the parser for recognition
    # These arguments might not be in vLLM's default parser, so we add them
    # to ensure they are parsed correctly.
    # Note: This is a placeholder for a more robust mapping if names differ.
    if not any(opt.dest == 'model_path' for opt in parser._actions):
            parser.add_argument("--model-path", type=str, default=None, help="iChat alias for --model.")
    if not any(opt.dest == 'tokenizer_path' for opt in parser._actions):
            parser.add_argument("--tokenizer-path", type=str, default=None, help="iChat alias for --tokenizer.")
    if not any(opt.dest == 'context_length' for opt in parser._actions):
            parser.add_argument("--context-length", type=int, default=None, help="iChat alias for --max-model-len.")

    # 3. Parse the backend-specific arguments
    vllm_args, _ = parser.parse_known_args(backend_argv)

    # 4. Apply mappings from iChat's unified args to vLLM's native args
    # If the user provided an iChat-specific argument, its value is mapped
    # to the corresponding vLLM argument.
    if vllm_args.model_path is not None:
        vllm_args.model = vllm_args.model_path

    if vllm_args.tokenizer_path is not None:
        vllm_args.tokenizer = vllm_args.tokenizer_path

    if vllm_args.context_length is not None:
        vllm_args.max_model_len = vllm_args.context_length
    
    return vllm_args

