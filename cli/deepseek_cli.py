#!/usr/bin/env python3
import click
import json
import os
import sys
from tabulate import tabulate

from deepseek.core import Config, SecurityProvider, StorageProvider
from deepseek.ai import AIEngine

@click.group()
def cli():
    """DeepSeek command-line interface."""
    pass

@cli.command()
@click.option('--config', '-c', help='Path to configuration file')
def init(config):
    """Initialize DeepSeek system."""
    try:
        Config.initialize(config)
        click.echo("DeepSeek initialized successfully.")
    except Exception as e:
        click.echo(f"Error initializing DeepSeek: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('query')
@click.option('--context', '-c', help='Context for the query (JSON string)')
@click.option('--output', '-o', help='Output file for results')
@click.option('--token', '-t', help='Authentication token')
def query(query, context, output, token):
    """Execute a query against the DeepSeek AI engine."""
    try:
        # Initialize
        Config.initialize()
        security = SecurityProvider()
        ai_engine = AIEngine()
        
        # Validate token
        if not token:
            token = os.environ.get("DEEPSEEK_TOKEN")
        
        if not token:
            click.echo("Authentication token required. Use --token or set DEEPSEEK_TOKEN environment variable.", err=True)
            sys.exit(1)
        
        try:
            user = security.validate_token(token)
        except Exception as e:
            click.echo(f"Authentication error: {str(e)}", err=True)
            sys.exit(1)
        
        # Parse context
        ctx = {}
        if context:
            try:
                ctx = json.loads(context)
            except json.JSONDecodeError:
                click.echo("Context must be a valid JSON string", err=True)
                sys.exit(1)
        
        # Execute query
        result = ai_engine.process_query(
            query=query,
            context=ctx,
            user_id=user.get("sub")
        )
        
        # Output results
        if output:
            with open(output, 'w') as f:
                json.dump(result, f, indent=2)
        else:
            if isinstance(result, dict) or isinstance(result, list):
                click.echo(json.dumps(result, indent=2))
            else:
                click.echo(result)
                
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--username', '-u', required=True, help='Username')
@click.option('--role', '-r', default='user', help='User role (admin, user, guest)')
@click.option('--expiry', '-e', type=int, default=1440, help='Token expiry in minutes')
def create_token(username, role, expiry):
    """Create an authentication token."""
    try:
        # Initialize
        Config.initialize()
        security = SecurityProvider()
        
        # Create token
        token = security.create_token(username, role, expiry)
        
        click.echo(f"Token for {username} (role: {role}, expiry: {expiry} minutes):")
        click.echo(token)
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--token', '-t', required=True, help='Token to validate')
def validate_token(token):
    """Validate an authentication token."""
    try:
        # Initialize
        Config.initialize()
        security = SecurityProvider()
        
        # Validate token
        payload = security.validate_token(token)
        
        click.echo("Token is valid:")
        click.echo(tabulate(
            [[k, v] for k, v in payload.items()],
            headers=["Claim", "Value"]
        ))
        
    except Exception as e:
        click.echo(f"Token is invalid: {str(e)}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    cli()
