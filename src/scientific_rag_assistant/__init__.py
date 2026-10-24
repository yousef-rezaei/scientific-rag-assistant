
import argparse

from .faq import index_faq, search_faq


def main():
    parser = argparse.ArgumentParser(
        description="Scientific AI Assistant"
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    commands.add_parser(
        "index",
        help="Build the FAQ vector index",
    )

    ask_command = commands.add_parser(
        "ask",
        help="Search the FAQ knowledge base",
    )

    ask_command.add_argument(
        "question",
        type=str,
        help="Natural-language user question",
    )

    args = parser.parse_args()

    if args.command == "index":
        index_faq()

    elif args.command == "ask":
        search_faq(args.question)
