import argparse
import sys

def parse_sentence_server_args():
    """
    Parses command-line arguments for the Sentence Transformer API Server.
    """
    parser = argparse.ArgumentParser(
        description="iChat Sentence Transformer API Server"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for the server to listen on."
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Port for the server to listen on."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the model weights or HuggingFace ID."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="The name for the served model."
    )
    parser.add_argument(
        "--task-type",
        type=str,
        required=True,
        choices=["embedding", "rerank"],
        help="The task to perform with the model."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="The device to use for inference (e.g., 'cpu', 'cuda')."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )

    # Note: We parse the full sys.argv, so the script name is sys.argv[0]
    return parser.parse_args(args=sys.argv[1:]) 