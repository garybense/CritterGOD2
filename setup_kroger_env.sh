#!/bin/bash
# Setup script for Kroger API environment variables
# 
# Usage:
#   source setup_kroger_env.sh
#
# This will set the required environment variables for the current shell session.
# For permanent setup, add these exports to your ~/.zshrc file.

export KROGER_CLIENT_ID="{{CLIENT_ID}}"
export KROGER_CLIENT_SECRET="{{CLIENT_SECRET}}"
export KROGER_OAUTH2_BASE_URL="https://api.kroger.com/v1/connect/oauth2"
export KROGER_API_BASE_URL="https://api.kroger.com"
export KROGER_REDIRECT_URL="http://localhost:3000/callback"

echo "✓ Kroger API environment variables set!"
echo ""
echo "You can now run:"
echo "  python3 kroger_api_client.py"
