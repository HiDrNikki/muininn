import argparse
from pathlib import Path

def build_parser(cli_templates):
    parser = argparse.ArgumentParser(prog="test")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for tmpl in cli_templates:
        cmd = tmpl["command"]
        desc = tmpl.get("description", None)
        func = tmpl.get("func", None)
        args = tmpl.get("args", [])
        sub = subparsers.add_parser(cmd, help=desc)
        for arg in args:
            flags = arg.get("flags", [])
            kwargs = {k: v for k, v in arg.items() if k not in ("name", "flags")}
            if flags:
                sub.add_argument(*flags, dest=arg["name"], **kwargs)
            else:
                sub.add_argument(arg["name"], **kwargs)
        if func:
            sub.set_defaults(func=func)
    return parser

def helpHandler(args):
    parser = args._parser
    if args.command:
        actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
        if actions:
            subparser = actions[0].choices.get(args.command)
            if subparser:
                subparser.print_help()
            else:
                parser.print_help()
        else:
            parser.print_help()
    else:
        parser.print_help()

cli_templates = [
    {
        "command": "help",
        "description": "Show help for a command.",
        "func": helpHandler,
        "args": [
            {
                "name": "command",
                "flags": ["-c", "--command"],
                "nargs": "?",
                "help": "Show help for this command."
            }
        ]
    }
]

def main():
    parser = build_parser(cli_templates)
    args = parser.parse_args()
    # Attach parser for use in help
    setattr(args, "_parser", parser)
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()