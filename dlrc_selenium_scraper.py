"""
David Lawrence Selenium Scraper
Uses Selenium to render JavaScript and scrape coin details and images
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import json
import os
import re
import requests
from pathlib import Path
from urllib.parse import urljoin


class DavidLawrenceSeleniumScraper:
    """Scraper for davidlawrence.com using Selenium to handle JavaScript."""
    
    BASE_URL = "https://davidlawrence.com"
    
    def __init__(self, output_dir='davidlawrence_coins', headless=True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.images_dir = self.output_dir / 'images'
        self.data_dir = self.output_dir / 'data'
        self.images_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        # Setup Chrome options
        chrome_options = Options()
        if headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        # Initialize driver
        print("Initializing Chrome WebDriver...")
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        print("✓ WebDriver ready")
        
        # Session for downloading images
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def __del__(self):
        """Clean up driver on exit."""
        if hasattr(self, 'driver'):
            self.driver.quit()
    
    def scrape_coin(self, inventory_id, wait_time=3, url_type='inventory'):
        """
        Scrape a single coin by inventory ID.
        
        Args:
            inventory_id: The inventory ID
            wait_time: Time to wait for page to load (seconds)
            url_type: 'inventory' or 'auction' - determines URL format
            
        Returns:
            dict: Coin data including details and image paths
        """
        # Handle both /inventory/ and /auctions/lot/ URLs
        if url_type == 'auction':
            url = f"{self.BASE_URL}/auctions/lot/{inventory_id}"
        else:
            url = f"{self.BASE_URL}/inventory/{inventory_id}"
        print(f"\n{'='*60}")
        print(f"Scraping: {url}")
        print(f"{'='*60}")
        
        try:
            # Load page
            self.driver.get(url)
            
            # Wait for page to load
            print(f"Waiting for page to load ({wait_time}s)...")
            time.sleep(wait_time)
            
            # Wait for content to appear
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except:
                print("⚠️  Timeout waiting for content")
            
        except Exception as e:
            print(f"❌ Error loading page: {e}")
            return None
        
        # Initialize coin data
        coin_data = {
            'inventory_id': inventory_id,
            'url': url,
            'cert_number': None,
            'grading_service': None,
            'pcgs_number': None,
            'series': None,
            'grade': None,
            'denomination': None,
            'year': None,
            'eye_appeal': None,
            'degree_of_toning': None,
            'mint_mark': None,
            'variety': None,
            'price': None,
            'description': None,
            'images': []
        }
        
        # Get page source for text extraction
        page_text = self.driver.page_source
        
        # Extract title/description
        try:
            # Try h1 first
            h1_elements = self.driver.find_elements(By.TAG_NAME, 'h1')
            if h1_elements:
                coin_data['description'] = h1_elements[0].text.strip()
                print(f"Title: {coin_data['description']}")
            
            # Try h2 if no h1
            if not coin_data['description']:
                h2_elements = self.driver.find_elements(By.TAG_NAME, 'h2')
                if h2_elements:
                    coin_data['description'] = h2_elements[0].text.strip()
                    print(f"Title (H2): {coin_data['description']}")
        except Exception as e:
            print(f"⚠️  Error extracting title: {e}")
        
        # Extract structured data from "About this item" section
        print("\nExtracting coin details...")
        try:
            # First, try to find "About this item" section
            about_section = None
            try:
                # Look for section containing "About this item"
                headings = self.driver.find_elements(By.XPATH, 
                    "//*[contains(text(), 'About this item') or contains(text(), 'about this item')]")
                if headings:
                    # Get the parent container of this heading
                    about_section = headings[0].find_element(By.XPATH, '../..')
                    print("Found 'About this item' section")
            except:
                print("Could not find 'About this item' section, using full page")
            
            # Get text content from the section or full page
            if about_section:
                page_text = about_section.text
            else:
                page_text = self.driver.find_element(By.TAG_NAME, 'body').text
            
            # Define fields to extract with their regex patterns and target keys
            fields = {
                'Grading Service': (r'Grading\s+Service[:\s\t]+(\S+)', 'grading_service'),
                'Certification #': (r'Certification\s*#?[:\s\t]+([\d-]+)', 'cert_number'),
                'PCGS Number': (r'PCGS\s+Number[:\s\t]+(\d+)', 'pcgs_number'),
                'Series': (r'Series[:\s\t]+([^\n\r\t]+?)(?:\n|\r|\t|$)', 'series'),
                'Date': (r'Date[:\s\t]+(\d{4})', 'year'),
                'Denomination': (r'Denomination[:\s\t]+([^\n\r\t]+?)(?:\n|\r|\t|$)', 'denomination'),
                'Grade': (r'Grade[:\s\t]+(\d+(?:\+)?)', 'grade'),
                'Eye Appeal': (r'Eye\s+Appeal[:\s\t]+([^\n\r\t]+?)(?:\n|\r|\t|$)', 'eye_appeal'),
                'Degree of Toning': (r'Degree\s+of\s+Toning[:\s\t]+([^\n\r\t]+?)(?:\n|\r|\t|$)', 'degree_of_toning'),
            }
            
            # Extract each field
            for field_name, (pattern, data_key) in fields.items():
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    print(f"{field_name}: {value}")
                    
                    # Store in coin_data
                    if data_key:
                        coin_data[data_key] = value
            
        except Exception as e:
            print(f"⚠️  Error extracting structured data: {e}")
        
        # Extract price
        try:
            price_elements = self.driver.find_elements(By.XPATH, 
                "//*[contains(@class, 'price') or contains(@class, 'amount')]")
            if price_elements:
                coin_data['price'] = price_elements[0].text.strip()
                print(f"Price: {coin_data['price']}")
            else:
                # Fallback to regex
                price_match = re.search(r'\$[\d,]+(?:\.\d{2})?', page_text)
                if price_match:
                    coin_data['price'] = price_match.group(0)
                    print(f"Price: {coin_data['price']}")
        except Exception as e:
            print(f"⚠️  Error extracting price: {e}")
        
        # Find all images by clicking thumbnails to reveal full-size
        print(f"\nSearching for images...")
        valid_images = []
        seen_srcs = set()
        
        # Strategy 1: Click thumbnails in thumbnailContainer to get full-size images
        try:
            thumbnail_containers = self.driver.find_elements(By.CLASS_NAME, 'thumbnailContainer')
            if thumbnail_containers:
                print(f"Found {len(thumbnail_containers)} thumbnail container(s)")
                
                # Get all thumbnail images
                thumbnails = []
                for container in thumbnail_containers:
                    imgs = container.find_elements(By.TAG_NAME, 'img')
                    thumbnails.extend(imgs)
                
                print(f"Found {len(thumbnails)} thumbnails to click")
                
                # Click each thumbnail and grab the full-size image
                for idx, thumb in enumerate(thumbnails):
                    try:
                        print(f"\n  Clicking thumbnail {idx + 1}/{len(thumbnails)}...")
                        
                        # Scroll to thumbnail
                        self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", thumb)
                        time.sleep(0.3)
                        
                        # Click the thumbnail (or its parent link)
                        try:
                            thumb.click()
                        except:
                            # Try clicking parent element
                            parent = thumb.find_element(By.XPATH, '..')
                            parent.click()
                        
                        # Wait for the larger image to load
                        time.sleep(1.0)
                        
                        # Look for the full-size image in the magnifier section
                        # David Lawrence shows it in: <img class="magnifier-image" src="...">
                        full_size_img = None
                        
                        try:
                            # Look for magnifier-image class (David Lawrence specific)
                            magnifier_img = self.driver.find_element(By.CLASS_NAME, 'magnifier-image')
                            if magnifier_img:
                                src = magnifier_img.get_attribute('src')
                                if src:
                                    full_size_img = src
                                    print(f"    ✓ Found magnifier-image: {src[:80]}...")
                        except:
                            print(f"    No magnifier-image found, trying other selectors...")
                            
                            # Fallback: Try other common selectors
                            for selector in [
                                '.main-coin-image img',
                                '[class*="mainImage"] img',
                                '[class*="zoomImage"] img',
                                '[class*="enlarged"] img',
                            ]:
                                try:
                                    elem = self.driver.find_element(By.CSS_SELECTOR, selector)
                                    src = elem.get_attribute('src')
                                    if src:
                                        full_size_img = src
                                        print(f"    ✓ Found via {selector}: {src[:80]}...")
                                        break
                                except:
                                    continue
                        
                        if full_size_img and full_size_img not in seen_srcs:
                            valid_images.append(full_size_img)
                            seen_srcs.add(full_size_img)
                        else:
                            print(f"    ⚠️  No full-size image found for this thumbnail")
                        
                    except Exception as e:
                        print(f"    ❌ Error clicking thumbnail: {e}")
                        continue
                
        except Exception as e:
            print(f"  Error processing thumbnailContainer: {e}")
        
        # Strategy 2: Look for gallery/carousel
        if not valid_images:
            try:
                gallery_elements = self.driver.find_elements(By.CSS_SELECTOR, 
                    '[class*="gallery"], [class*="carousel"], [class*="slider"]')
                for gallery in gallery_elements:
                    imgs = gallery.find_elements(By.TAG_NAME, 'img')
                    for img in imgs:
                        src = img.get_attribute('src') or img.get_attribute('data-src')
                        if src and src not in seen_srcs:
                            valid_images.append(src)
                            seen_srcs.add(src)
            except:
                pass
        
        # Strategy 3: All images on page (filtered)
        if not valid_images:
            print("Scanning all images on page...")
            image_elements = self.driver.find_elements(By.TAG_NAME, 'img')
            print(f"Found {len(image_elements)} total <img> elements")
            
            for img in image_elements:
                try:
                    src = img.get_attribute('src') or img.get_attribute('data-src')
                    if not src or src in seen_srcs:
                        continue
                    
                    # Skip tiny images, icons, logos
                    if any(skip in src.lower() for skip in ['logo', 'icon', 'button', 'sprite', 'favicon']):
                        continue
                    
                    # Check image dimensions
                    try:
                        width = img.size.get('width', 0)
                        height = img.size.get('height', 0)
                        
                        # Skip small images
                        if width and height and (width < 100 or height < 100):
                            continue
                    except:
                        pass
                    
                    valid_images.append(src)
                    seen_srcs.add(src)
                    
                except Exception as e:
                    continue
        
        print(f"Found {len(valid_images)} valid images total")
        
        # Download images
        downloaded_images = []
        
        for idx, img_url in enumerate(valid_images):
            # Make absolute URL
            if img_url.startswith('//'):
                img_url = 'https:' + img_url
            elif img_url.startswith('/'):
                img_url = self.BASE_URL + img_url
            
            # Try to get full size
            full_size_url = self._get_fullsize_url(img_url)
            
            print(f"\nImage {idx + 1}: {full_size_url}")
            
            # Download
            image_path = self._download_image(full_size_url, inventory_id, idx)
            if image_path:
                downloaded_images.append(str(image_path))
                print(f"  ✓ Downloaded")
        
        coin_data['images'] = downloaded_images
        
        # Save coin data
        self._save_coin_data(coin_data)
        
        print(f"\n{'='*60}")
        print(f"✓ Scraping complete!")
        print(f"  Images downloaded: {len(downloaded_images)}")
        print(f"  Data saved: {self.data_dir / f'{inventory_id}.json'}")
        print(f"{'='*60}")
        
        return coin_data
    
    def scrape_multiple(self, inventory_ids, delay=3, wait_time=3, url_type='inventory'):
        """
        Scrape multiple coins by inventory ID.
        
        Args:
            inventory_ids: List of inventory IDs to scrape
            delay: Delay between requests (seconds)
            wait_time: Wait time for each page load
            url_type: 'inventory' or 'auction' - determines URL format
            
        Returns:
            list: List of scraped coin data
        """
        results = []
        total = len(inventory_ids)
        
        print(f"\n{'='*60}")
        print(f"BULK SCRAPING {total} COINS")
        print(f"{'='*60}\n")
        
        for idx, inv_id in enumerate(inventory_ids, 1):
            print(f"\n{'='*60}")
            print(f"Coin {idx}/{total}: Inventory ID {inv_id}")
            print(f"{'='*60}")
            
            try:
                coin_data = self.scrape_coin(inv_id, wait_time=wait_time, url_type=url_type)
                if coin_data:
                    results.append(coin_data)
                    print(f"✓ Success! ({len(results)}/{idx} successful)")
                else:
                    print(f"⚠️  Failed to scrape coin {inv_id}")
            
            except Exception as e:
                print(f"❌ Error scraping {inv_id}: {e}")
            
            # Progress summary
            success_rate = (len(results) / idx) * 100
            print(f"\nProgress: {idx}/{total} ({idx/total*100:.1f}%) | Success rate: {success_rate:.1f}%")
            
            # Delay before next request (except for last one)
            if idx < total:
                print(f"Waiting {delay}s before next coin...")
                time.sleep(delay)
        
        print(f"\n{'='*60}")
        print(f"✓ BULK SCRAPING COMPLETE")
        print(f"{'='*60}")
        print(f"Successfully scraped: {len(results)}/{total} ({len(results)/total*100:.1f}%)")
        print(f"{'='*60}\n")
        
        return results
    
    def _get_fullsize_url(self, thumbnail_url):
        """Convert thumbnail URL to full-size by replacing dimensions."""
        full_url = thumbnail_url
        
        # David Lawrence specific: Replace width=256&height=256 with 1000x1000
        full_url = re.sub(r'width=\d+', 'width=1000', full_url)
        full_url = re.sub(r'height=\d+', 'height=1000', full_url)
        
        # Also handle w= and h= parameters
        full_url = re.sub(r'\bw=\d+', 'w=1000', full_url)
        full_url = re.sub(r'\bh=\d+', 'h=1000', full_url)
        
        return full_url
    
    def _download_image(self, url, inventory_id, index):
        """Download image and save to disk."""
        try:
            response = self.session.get(url, timeout=15, stream=True)
            response.raise_for_status()
            
            # Get file extension
            content_type = response.headers.get('content-type', '')
            if 'jpeg' in content_type or 'jpg' in content_type:
                ext = '.jpg'
            elif 'png' in content_type:
                ext = '.png'
            else:
                ext = '.jpg'
            
            # Create filename
            filename = f"{inventory_id}_image_{index}{ext}"
            filepath = self.images_dir / filename
            
            # Save image
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return filepath
            
        except Exception as e:
            print(f"  ❌ Error downloading: {e}")
            return None
    
    def _save_coin_data(self, coin_data):
        """Save coin data to JSON file."""
        filepath = self.data_dir / f"{coin_data['inventory_id']}.json"
        with open(filepath, 'w') as f:
            json.dump(coin_data, f, indent=2)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='David Lawrence Selenium Scraper')
    parser.add_argument('--inventory-id', '-i', default='789518', help='Inventory ID')
    parser.add_argument('--wait', '-w', type=int, default=3, help='Wait time for page load (seconds)')
    parser.add_argument('--visible', '-v', action='store_true', help='Show browser (not headless)')
    
    args = parser.parse_args()
    
    scraper = DavidLawrenceSeleniumScraper(headless=not args.visible)
    
    try:
        coin_data = scraper.scrape_coin(args.inventory_id, wait_time=args.wait)
        
        if coin_data:
            print("\n" + "="*60)
            print("Scraped Data:")
            print("="*60)
            for key, value in coin_data.items():
                if key != 'images':
                    print(f"{key}: {value}")
                else:
                    print(f"images: {len(value)} files")
    finally:
        scraper.driver.quit()


if __name__ == '__main__':
    main()

