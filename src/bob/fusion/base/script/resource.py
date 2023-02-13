"""A script to list the resources.
"""
from __future__ import absolute_import, division, print_function

import logging

import click

import bob.bio.base

from bob.extension.scripts.click_helper import verbosity_option

logger = logging.getLogger(__name__)


@click.command(
    epilog="""\b
Examples:
$ bob fusion resource
$ bob fusion resource -v
"""
)
@click.option(
    "--packages",
    "-p",
    multiple=True,
    help="List only the resources from these packages.",
)
@verbosity_option()
@click.pass_context
def resource(ctx, packages, **kwargs):
    """Lists fusion algorithm resources."""
    click.echo(
        bob.bio.base.list_resources(
            "algorithm",
            strip=["dummy"],
            package_prefix="bob.fusion.",
            verbose=ctx.meta["verbosity"],
            packages=packages or None,
        )
    )
