#!/usr/bin/env python
"""Command-line interface."""
import click
from rich import traceback


@click.command()
@click.version_option()
def main() -> None:
    """ncem."""


if __name__ == "__main__":
    traceback.install()
    main(prog_name="ncem")  # pragma: no cover
