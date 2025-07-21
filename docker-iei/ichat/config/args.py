import argparse
import sys


def parse_worker_args():
    """
    Parses command-line arguments for the iChat Worker.

    This function focuses on parsing only the iChat framework-specific arguments.
    Any unrecognized arguments are collected and returned to be passed through
    to the specific backend engine (e.g., vLLM, SGLang) for its own parsing.

    Returns:
        A tuple containing:
        - An argparse.Namespace object with the parsed iChat framework arguments.
        - A list of strings representing the remaining, unparsed arguments for the backend.
    """
    parser = argparse.ArgumentParser(
        description="iChat Worker - Universal LLM Serving"
    )

    # 1. iChat Framework-Specific Arguments
    ichat_group = parser.add_argument_group("iChat Framework Arguments")
    ichat_group.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=["vllm", "sglang"],
        help="The inference backend to use.",
    )
    ichat_group.add_argument(
        "--gateway-address",
        type=str,
        default=None,
        help="The address of the iChat Gateway for service registration and heartbeat.",
    )
    ichat_group.add_argument(
        "--heartbeat-interval",
        type=int,
        default=30,
        help="The interval in seconds for sending heartbeats to the gateway.",
    )
    ichat_group.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    ichat_group.add_argument(
        "--log-streaming",
        action="store_true",
        help="Enable streaming of logs to the iChat Gateway.",
    )

    # 2. Parse known args to separate framework-specific args from backend-specific ones.
    # We slice sys.argv to exclude the script name itself.
    framework_args, backend_argv = parser.parse_known_args(args=sys.argv[1:])

    return framework_args, backend_argv
