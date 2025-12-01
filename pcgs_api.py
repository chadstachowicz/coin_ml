"""PCGS API integration module."""
import requests
import os


class PCGSClient:
    """Client for interacting with the PCGS API."""
    
    BASE_URL = "https://api.pcgs.com/publicapi"
    API_TOKEN = "RUH_TtXdHQ9deyf9uEXiuAD4xcD4coLXUcfvkXZCWqbql3otggplwcVYOQLG1P37w_1MlO-in-nOF2FdwSs09exPvK20yhpAIg64WZjJdYbde44GLsIGRmNygqIrJ71Gp4Lo4IrgPJJjrQb-vVcXbPWXRZW7jB5jrlm9Cy7Q5qPNrH9YDaJLyUVckYU2gwf8043YzI9WsPey4i5uOKbhPnalvIPSaLnm0LPFXIBesVC4VLN_XhLdiicIUgTkd_9YUqEcFG8U67zsb0VgFfVGq969ipSKNRUtNXThftvwMthWZPE4"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.API_TOKEN}',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_coin_facts(self, cert_no):
        """
        Get coin facts by certification number.
        
        Args:
            cert_no: PCGS certification number
            
        Returns:
            dict: Coin facts data or None if not found
        """
        url = f"{self.BASE_URL}/coindetail/GetCoinFactsByCertNo/{cert_no}"
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if data and isinstance(data, dict):
                return data
            return None
        except Exception as e:
            print(f"Error fetching coin facts for {cert_no}: {e}")
            return None
    
    def get_images(self, cert_no):
        """
        Get coin images by certification number.
        
        Args:
            cert_no: PCGS certification number
            
        Returns:
            dict: Image data with structure:
            {
                "CertNo": "string",
                "Images": [{"Url": "string", "Resolution": "string", "Description": "string"}],
                "HasObverseImage": bool,
                "HasReverseImage": bool,
                "HasTrueViewImage": bool,
                "ImageReady": bool,
                "IsValidRequest": bool,
                "ServerMessage": "string"
            }
        """
        url = f"{self.BASE_URL}/coindetail/GetImagesByCertNo"
        try:
            # API uses query parameter, not path parameter
            response = self.session.get(url, params={'certNo': cert_no}, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            # Check if request was valid
            if data and isinstance(data, dict) and data.get('IsValidRequest'):
                return data
            return None
        except Exception as e:
            print(f"Error fetching images for {cert_no}: {e}")
            return None
    
    def download_image(self, image_url, save_path):
        """
        Download an image from URL and save it.
        
        Args:
            image_url: URL of the image
            save_path: Path where to save the image
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            response = self.session.get(image_url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
        except Exception as e:
            print(f"Error downloading image from {image_url}: {e}")
            return False

