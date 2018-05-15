"""The main entry for bob.fusion (click-based) scripts.
"""
import click
import pkg_resources
from click_plugins import with_plugins


@with_plugins(pkg_resources.iter_entry_points('bob.fusion.cli'))
@click.group()
def fusion():
    """Score fusion related scripts"""
    pass
