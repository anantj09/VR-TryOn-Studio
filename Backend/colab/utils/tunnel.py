import os
import sys
from typing import Optional

def start_tunnel(port: int, authtoken: Optional[str] = None) -> str:
    """
    Starts an Ngrok tunnel on the specified port.
    Returns the public HTTPS URL.
    """
    try:
        from pyngrok import ngrok, conf
    except ImportError:
        print("Error: 'pyngrok' is not installed. Please run pip install pyngrok first.")
        sys.exit(1)

    # Use authtoken if provided, otherwise look in env variables
    token = authtoken or os.environ.get("NGROK_AUTHTOKEN")
    
    if not token:
        print("\n"
              "=======================================================================\n"
              "[ERROR] NGROK AUTHTOKEN NOT PROVIDED\n"
              "To expose the Colab server, please sign up at https://ngrok.com/ and:\n"
              "1. Retrieve your authtoken from your dashboard.\n"
              "2. Pass it as a command line argument or set NGROK_AUTHTOKEN env var.\n"
              "=======================================================================\n")
        sys.exit(1)

    conf.get_default().auth_token = token
    
    # Expose port via HTTPS
    print(f"Opening Ngrok tunnel on local port: {port}...")
    try:
        tunnel = ngrok.connect(port, "http")
        public_url = tunnel.public_url
        
        # Ngrok returns URLs with http:// or https://. Ensure we print the https:// version.
        if public_url.startswith("http://"):
            public_url = public_url.replace("http://", "https://", 1)
            
        print("\n"
              "=======================================================================\n"
              f"  [NGROK TUNNEL ONLINE]\n"
              f"  Public Endpoint: {public_url}\n"
              "  Paste the URL above in your Frontend settings to link with Colab!\n"
              "=======================================================================\n")
              
        return public_url
    except Exception as e:
        print(f"Failed to start Ngrok tunnel: {str(e)}")
        sys.exit(1)

def stop_tunnel():
    """
    Closes all active Ngrok tunnels.
    """
    try:
        from pyngrok import ngrok
        print("Closing all active Ngrok tunnels...")
        ngrok.disconnect()
        ngrok.kill()
    except Exception as e:
        print(f"Error closing tunnels: {str(e)}")
