#!/usr/bin/env python3
"""
Kroger API Client with OAuth2 Authentication

This client handles authentication and API requests to the Kroger API.
Supports both client credentials flow and authorization code flow.
"""

import os
import requests
import base64
from typing import Optional, Dict, Any
from urllib.parse import urlencode


class KrogerAPIClient:
    """Client for interacting with Kroger API."""
    
    def __init__(self):
        """Initialize the Kroger API client with credentials from environment."""
        self.client_id = os.getenv('KROGER_CLIENT_ID')
        self.client_secret = os.getenv('KROGER_CLIENT_SECRET')
        self.oauth2_base_url = os.getenv('KROGER_OAUTH2_BASE_URL', 'https://api.kroger.com/v1/connect/oauth2')
        self.api_base_url = os.getenv('KROGER_API_BASE_URL', 'https://api.kroger.com')
        self.redirect_url = os.getenv('KROGER_REDIRECT_URL', 'http://localhost:3000/callback')
        
        if not self.client_id or not self.client_secret:
            raise ValueError("KROGER_CLIENT_ID and KROGER_CLIENT_SECRET must be set in environment")
        
        self.access_token: Optional[str] = None
        self.token_type: Optional[str] = None
        self.expires_in: Optional[int] = None
    
    def _get_auth_header(self) -> str:
        """Generate Basic Auth header for OAuth2."""
        credentials = f"{self.client_id}:{self.client_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"
    
    def get_client_credentials_token(self, scope: str = "product.compact") -> Dict[str, Any]:
        """
        Get access token using client credentials flow (for public data).
        
        Args:
            scope: The scope to request (default: product.compact)
            
        Returns:
            Token response dictionary
        """
        url = f"{self.oauth2_base_url}/token"
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": self._get_auth_header()
        }
        
        data = {
            "grant_type": "client_credentials",
            "scope": scope
        }
        
        response = requests.post(url, headers=headers, data=data)
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
        response.raise_for_status()
        
        token_data = response.json()
        self.access_token = token_data.get('access_token')
        self.token_type = token_data.get('token_type')
        self.expires_in = token_data.get('expires_in')
        
        return token_data
    
    def get_authorization_url(self, scope: str = "product.compact") -> str:
        """
        Get authorization URL for user authorization flow.
        
        Args:
            scope: The scope to request
            
        Returns:
            Authorization URL to redirect user to
        """
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_url,
            "response_type": "code",
            "scope": scope
        }
        
        return f"{self.oauth2_base_url}/authorize?{urlencode(params)}"
    
    def exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """
        Exchange authorization code for access token.
        
        Args:
            code: Authorization code from callback
            
        Returns:
            Token response dictionary
        """
        url = f"{self.oauth2_base_url}/token"
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": self._get_auth_header()
        }
        
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_url
        }
        
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        
        token_data = response.json()
        self.access_token = token_data.get('access_token')
        self.token_type = token_data.get('token_type')
        self.expires_in = token_data.get('expires_in')
        
        return token_data
    
    def _get_api_headers(self) -> Dict[str, str]:
        """Get headers for authenticated API requests."""
        if not self.access_token:
            raise ValueError("No access token. Call get_client_credentials_token() first.")
        
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json"
        }
    
    def get_products(self, filter_term: Optional[str] = None, 
                     location_id: Optional[str] = None,
                     limit: int = 10) -> Dict[str, Any]:
        """
        Search for products.
        
        Args:
            filter_term: Search term for products
            location_id: Store location ID
            limit: Number of results to return
            
        Returns:
            Products response
        """
        url = f"{self.api_base_url}/v1/products"
        
        params = {"filter.limit": limit}
        if filter_term:
            params["filter.term"] = filter_term
        if location_id:
            params["filter.locationId"] = location_id
        
        response = requests.get(url, headers=self._get_api_headers(), params=params)
        response.raise_for_status()
        
        return response.json()
    
    def get_locations(self, zip_code: Optional[str] = None,
                      lat: Optional[float] = None,
                      lon: Optional[float] = None,
                      radius: int = 10,
                      limit: int = 10) -> Dict[str, Any]:
        """
        Search for store locations.
        
        Args:
            zip_code: ZIP code to search near
            lat: Latitude for location search
            lon: Longitude for location search
            radius: Search radius in miles
            limit: Number of results to return
            
        Returns:
            Locations response
        """
        url = f"{self.api_base_url}/v1/locations"
        
        params = {
            "filter.limit": limit,
            "filter.radiusInMiles": radius
        }
        
        if zip_code:
            params["filter.zipCode.near"] = zip_code
        elif lat and lon:
            params["filter.lat.near"] = lat
            params["filter.lon.near"] = lon
        
        response = requests.get(url, headers=self._get_api_headers(), params=params)
        response.raise_for_status()
        
        return response.json()


def main():
    """Example usage of the Kroger API client."""
    
    # Initialize client
    client = KrogerAPIClient()
    
    # Get access token using client credentials flow
    print("Getting access token...")
    token_response = client.get_client_credentials_token()
    print(f"✓ Access token obtained (expires in {token_response['expires_in']} seconds)\n")
    
    # Example: Search for products
    print("Searching for 'milk'...")
    products = client.get_products(filter_term="milk", limit=5)
    print(f"✓ Found {len(products.get('data', []))} products:\n")
    
    for product in products.get('data', [])[:3]:
        print(f"  - {product.get('description', 'N/A')}")
        print(f"    UPC: {product.get('upc', 'N/A')}")
        print()
    
    # Example: Find nearby stores
    print("\nSearching for stores near ZIP 90210...")
    locations = client.get_locations(zip_code="90210", limit=3)
    print(f"✓ Found {len(locations.get('data', []))} locations:\n")
    
    for location in locations.get('data', []):
        print(f"  - {location.get('name', 'N/A')}")
        address = location.get('address', {})
        print(f"    {address.get('addressLine1', 'N/A')}, {address.get('city', 'N/A')}, {address.get('state', 'N/A')}")
        print()


if __name__ == "__main__":
    main()
