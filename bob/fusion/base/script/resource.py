"""A script to list the resources.
"""
from __future__ import print_function, absolute_import, division
import logging
import click
from bob.extension.scripts.click_helper import verbosity_option
import bob.bio.base

logger = logging.getLogger(__name__)


@click.command(epilog='''\b
Examples:
$ bob fusion resource
$ bob fusion resource -v
''')
@click.option('--packages', '-p', multiple=True,
              help='List only the resources from these packages.')
@verbosity_option()
@click.pass_context
def resource(ctx, packages, **kwargs):
    """Lists fusion algorithm resources.
    """
    click.echo(bob.bio.base.list_resources(
        'algorithm', strip=['dummy'], package_prefix='bob.fusion.',
        verbose=ctx.meta['verbosity'], packages=packages or None))
