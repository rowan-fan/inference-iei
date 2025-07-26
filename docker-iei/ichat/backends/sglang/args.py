
from argparse import ArgumentParser, Namespace
from typing import List

from sglang.srt.server_args import ServerArgs


def parse_sglang_args(framework_args: Namespace,  backend_argv: List[str]) -> ServerArgs:
    """
    Parses SGLang-specific arguments from the command line.
    """
    parser = ArgumentParser()
    ServerArgs.add_cli_args(parser)

    # Add compatibility arguments
    if not any(opt.dest == 'max_model_len' for opt in parser._actions):
        parser.add_argument("--max-model-len", type=int, default=None, help="SGLang alias for --context-length.")

    # 1. Parse the backend-specific arguments passed from the command line
    sglang_cli_args, _ = parser.parse_known_args(backend_argv)

    # 2. Apply mappings from iChat's unified args to SGLang's native args
    if sglang_cli_args.max_model_len is not None:
        sglang_cli_args.context_length = sglang_cli_args.max_model_len
    # 3. Create the final ServerArgs instance from the combined arguments
    server_args = ServerArgs.from_cli_args(sglang_cli_args)
    server_args.check_server_args()
    
    return server_args

