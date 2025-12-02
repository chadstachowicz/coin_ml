"""
David Lawrence Rare Coins Scraper
Scrapes coin details and downloads full-size images (1000x1000)
"""

import requests
from bs4 import BeautifulSoup
import json
import os
import re
from pathlib import Path
from urllib.parse import urljoin
import time


class DavidLawrenceScraper:
    """Scraper for davidlawrence.com coin listings."""
    
    BASE_URL = "https://davidlawrence.com"
    
    def __init__(self, output_dir='davidlawrence_coins'):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.images_dir = self.output_dir / 'images'
        self.data_dir = self.output_dir / 'data'
        self.images_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
    
    def discover_inventory_ids(self, listing_url, max_pages=None):
        """
        Discover inventory IDs from a listing/auction/category page.
        
        Args:
            listing_url: URL of the listing page (e.g., auction results, category page)
            max_pages: Maximum number of pages to scrape (None = all pages)
            
        Returns:
            list: List of unique inventory IDs found
        """
        inventory_ids = set()
        page = 1
        
        print(f"\n{'='*60}")
        print(f"Discovering inventory IDs from listing page")
        print(f"{'='*60}")
        print(f"URL: {listing_url}")
        print(f"Max pages: {max_pages if max_pages else 'All'}")
        print(f"{'='*60}\n")
        
        while True:
            # Add page parameter if not first page
            if page > 1:
                separator = '&' if '?' in listing_url else '?'
                current_url = f"{listing_url}{separator}page={page}"
            else:
                current_url = listing_url
            
            print(f"Scraping page {page}: {current_url}")
            
            try:
                response = self.session.get(current_url, timeout=15)
                response.raise_for_status()
            except Exception as e:
                print(f"  ❌ Error fetching page: {e}")
                break
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all links to inventory pages
            # Pattern: /inventory/123456 or /inventory/123456?...
            links = soup.find_all('a', href=re.compile(r'/inventory/\d+'))
            
            page_ids = set()
            for link in links:
                href = link.get('href')
                # Extract inventory ID from URL
                match = re.search(r'/inventory/(\d+)', href)
                if match:
                    inv_id = match.group(1)
                    page_ids.add(inv_id)
                    inventory_ids.add(inv_id)
            
            print(f"  Found {len(page_ids)} inventory IDs on this page")
            print(f"  Total unique IDs: {len(inventory_ids)}")
            
            # Check if there's a next page
            next_link = soup.find('a', text=re.compile(r'next|›|»', re.IGNORECASE))
            if not next_link:
                # Also check for pagination links
                pagination = soup.find('ul', class_=re.compile(r'pagination', re.IGNORECASE))
                if pagination:
                    next_link = pagination.find('a', attrs={'rel': 'next'})
            
            # Check if we should continue
            has_next = next_link is not None and len(page_ids) > 0
            reached_max = max_pages and page >= max_pages
            
            if not has_next or reached_max:
                break
            
            page += 1
            time.sleep(1)  # Be nice between pages
        
        inventory_list = sorted(list(inventory_ids))
        
        print(f"\n{'='*60}")
        print(f"✓ Discovery complete!")
        print(f"  Total pages scraped: {page}")
        print(f"  Total inventory IDs found: {len(inventory_list)}")
        print(f"{'='*60}\n")
        
        # Show first 10 IDs as preview
        if inventory_list:
            print("Preview (first 10 IDs):")
            for inv_id in inventory_list[:10]:
                print(f"  - {inv_id}")
            if len(inventory_list) > 10:
                print(f"  ... and {len(inventory_list) - 10} more")
            print()
        
        return inventory_list
    
    def scrape_coin(self, inventory_id, debug=False):
        """
        Scrape a single coin by inventory ID.
        
        Args:
            inventory_id: The inventory ID (e.g., '789518')
            debug: Print debug information
            
        Returns:
            dict: Coin data including details and image paths
        """
        url = f"{self.BASE_URL}/inventory/{inventory_id}"
        print(f"\n{'='*60}")
        print(f"Scraping: {url}")
        print(f"{'='*60}")
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
        except Exception as e:
            print(f"❌ Error fetching page: {e}")
            return None
        
        if debug:
            print(f"\nDEBUG: Response status: {response.status_code}")
            print(f"DEBUG: Content length: {len(response.text)} bytes")
            print(f"DEBUG: Content type: {response.headers.get('content-type')}")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        if debug:
            # Save HTML for inspection
            debug_file = self.output_dir / f'debug_{inventory_id}.html'
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"DEBUG: Saved HTML to {debug_file}")
        
        # Extract coin details
        coin_data = {
            'inventory_id': inventory_id,
            'url': url,
            'cert_number': None,
            'series': None,
            'grade': None,
            'denomination': None,
            'year': None,
            'mint_mark': None,
            'variety': None,
            'price': None,
            'description': None,
            'images': []
        }
        
        # Extract title/description - try multiple strategies
        title = None
        
        # Strategy 1: Look for common product title classes
        for class_name in ['product-title', 'item-title', 'coin-title', 'title', 'lot-title']:
            title = soup.find(['h1', 'h2', 'h3'], class_=re.compile(class_name, re.I))
            if title:
                break
        
        # Strategy 2: Look for any h1
        if not title:
            title = soup.find('h1')
        
        # Strategy 3: Look in meta tags
        if not title:
            meta_title = soup.find('meta', property='og:title') or soup.find('meta', {'name': 'title'})
            if meta_title:
                coin_data['description'] = meta_title.get('content', '').strip()
        
        if title and not coin_data['description']:
            coin_data['description'] = title.get_text(strip=True)
        
        if coin_data['description']:
            print(f"Title: {coin_data['description']}")
        else:
            print(f"⚠️  No title found")
        
        if debug:
            print(f"DEBUG: Description: {coin_data['description']}")
        
        # Look for cert number (PCGS/NGC cert)
        cert_patterns = [
            r'PCGS\s*#?\s*(\d+)',
            r'NGC\s*#?\s*(\d+)',
            r'Cert\s*#?\s*(\d+)',
            r'Certification\s*#?\s*(\d+)'
        ]
        
        text_content = soup.get_text()
        for pattern in cert_patterns:
            match = re.search(pattern, text_content, re.IGNORECASE)
            if match:
                coin_data['cert_number'] = match.group(1)
                print(f"Cert Number: {coin_data['cert_number']}")
                break
        
        # Extract grade (MS, AU, PR, etc.)
        grade_pattern = r'\b(MS|AU|PR|PF|XF|VF|F|VG|G|AG|FR)\s*\d+\+?\b'
        grade_match = re.search(grade_pattern, text_content, re.IGNORECASE)
        if grade_match:
            coin_data['grade'] = grade_match.group(0).upper()
            print(f"Grade: {coin_data['grade']}")
        
        # Extract year (1700-2099)
        year_match = re.search(r'\b(1[7-9]\d{2}|20\d{2})\b', text_content)
        if year_match:
            coin_data['year'] = year_match.group(1)
            print(f"Year: {coin_data['year']}")
        
        # Extract price
        price_elem = soup.find('span', class_=re.compile(r'price|amount', re.I))
        if not price_elem:
            price_match = re.search(r'\$[\d,]+(?:\.\d{2})?', text_content)
            if price_match:
                coin_data['price'] = price_match.group(0)
        else:
            coin_data['price'] = price_elem.get_text(strip=True)
        if coin_data['price']:
            print(f"Price: {coin_data['price']}")
        
        # Find all images - enhanced strategies
        images = []
        seen_srcs = set()
        
        # Strategy 1: Look for image gallery
        for gallery_class in ['gallery', 'images', 'product-images', 'coin-images', 'item-images']:
            gallery = soup.find(['div', 'section'], class_=re.compile(gallery_class, re.I))
            if gallery:
                if debug:
                    print(f"DEBUG: Found gallery with class: {gallery_class}")
                for img in gallery.find_all('img'):
                    src = img.get('src') or img.get('data-src') or img.get('data-zoom-image')
                    if src and src not in seen_srcs:
                        images.append(img)
                        seen_srcs.add(src)
        
        # Strategy 2: Look for main product image
        for img_class in ['product', 'main', 'primary', 'coin', 'item']:
            main_img = soup.find('img', class_=re.compile(img_class, re.I))
            if main_img:
                src = main_img.get('src') or main_img.get('data-src')
                if src and src not in seen_srcs:
                    images.append(main_img)
                    seen_srcs.add(src)
                    if debug:
                        print(f"DEBUG: Found main image with class: {img_class}")
        
        # Strategy 3: Find all images in product area
        for area_class in ['product', 'item', 'coin', 'lot', 'content']:
            product_area = soup.find(['div', 'section', 'article'], class_=re.compile(area_class, re.I))
            if product_area:
                for img in product_area.find_all('img'):
                    src = img.get('src') or img.get('data-src') or img.get('data-zoom-image')
                    if src and src not in seen_srcs:
                        images.append(img)
                        seen_srcs.add(src)
        
        # Strategy 4: Look for data attributes that might contain image URLs
        for elem in soup.find_all(['div', 'a'], attrs={'data-image': True}):
            # Create a pseudo-img element
            img_data = {'src': elem.get('data-image')}
            if img_data['src'] and img_data['src'] not in seen_srcs:
                class FakeImg:
                    def get(self, key):
                        return img_data.get(key)
                images.append(FakeImg())
                seen_srcs.add(img_data['src'])
        
        # Strategy 5: Fallback - all images on page (filtered)
        if not images:
            for img in soup.find_all('img'):
                src = img.get('src') or img.get('data-src')
                if src and src not in seen_srcs:
                    images.append(img)
                    seen_srcs.add(src)
        
        print(f"\nFound {len(images)} potential images")
        
        if debug and images:
            print("\nDEBUG: Image sources found:")
            for idx, img in enumerate(images[:5]):  # Show first 5
                src = img.get('src') or img.get('data-src') or img.get('data-zoom-image')
                print(f"  {idx + 1}. {src}")
        
        # Download images
        downloaded_images = []
        for idx, img in enumerate(images):
            img_url = img.get('src') or img.get('data-src') or img.get('data-zoom-image')
            
            if not img_url:
                continue
            
            # Skip tiny images (icons, logos)
            if any(skip in img_url.lower() for skip in ['logo', 'icon', 'button', 'sprite']):
                continue
            
            # Make absolute URL
            img_url = urljoin(url, img_url)
            
            # Try to get full-size version
            # Many sites use patterns like: thumb_file.jpg -> file.jpg or file_small.jpg -> file_large.jpg
            full_size_url = self._get_fullsize_url(img_url)
            
            print(f"\nImage {idx + 1}:")
            print(f"  Thumbnail: {img_url}")
            print(f"  Full-size: {full_size_url}")
            
            # Download full-size image
            image_path = self._download_image(full_size_url, inventory_id, idx)
            if image_path:
                downloaded_images.append(str(image_path))
                print(f"  ✓ Downloaded: {image_path}")
        
        coin_data['images'] = downloaded_images
        
        # Save coin data
        self._save_coin_data(coin_data)
        
        print(f"\n{'='*60}")
        print(f"✓ Scraping complete!")
        print(f"  Images downloaded: {len(downloaded_images)}")
        print(f"  Data saved: {self.data_dir / f'{inventory_id}.json'}")
        print(f"{'='*60}")
        
        return coin_data
    
    def _get_fullsize_url(self, thumbnail_url):
        """
        Try to convert thumbnail URL to full-size image URL.
        Common patterns:
        - /thumb/ -> /large/ or /full/
        - _thumb -> _large or remove suffix
        - _small -> _large
        - ?size=small -> ?size=large
        """
        full_url = thumbnail_url
        
        # Replace common thumbnail indicators
        replacements = [
            ('/thumb/', '/large/'),
            ('/thumb/', '/full/'),
            ('/thumb/', '/'),
            ('_thumb', '_large'),
            ('_thumb', ''),
            ('_small', '_large'),
            ('_s.', '_l.'),
            ('/s/', '/l/'),
            ('/small/', '/large/'),
            ('/thumbnails/', '/images/'),
        ]
        
        for old, new in replacements:
            if old in full_url.lower():
                full_url = full_url.replace(old, new)
                break
        
        # Try removing size parameters
        if 'size=' in full_url.lower():
            full_url = re.sub(r'[?&]size=\w+', '', full_url, flags=re.IGNORECASE)
        
        # Try adding width parameter for 1000x1000
        if '?' in full_url:
            full_url += '&width=1000&height=1000'
        else:
            full_url += '?width=1000&height=1000'
        
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
            elif 'gif' in content_type:
                ext = '.gif'
            else:
                ext = '.jpg'  # default
            
            # Create filename
            filename = f"{inventory_id}_image_{index}{ext}"
            filepath = self.images_dir / filename
            
            # Save image
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return filepath
            
        except Exception as e:
            print(f"  ❌ Error downloading image: {e}")
            return None
    
    def _save_coin_data(self, coin_data):
        """Save coin data to JSON file."""
        filepath = self.data_dir / f"{coin_data['inventory_id']}.json"
        with open(filepath, 'w') as f:
            json.dump(coin_data, f, indent=2)
    
    def scrape_multiple(self, inventory_ids, delay=2):
        """
        Scrape multiple coins.
        
        Args:
            inventory_ids: List of inventory IDs
            delay: Delay between requests (seconds)
        """
        results = []
        
        print(f"{'='*60}")
        print(f"Scraping {len(inventory_ids)} coins")
        print(f"{'='*60}")
        
        for i, inv_id in enumerate(inventory_ids, 1):
            print(f"\n[{i}/{len(inventory_ids)}] Processing inventory {inv_id}...")
            
            result = self.scrape_coin(inv_id)
            if result:
                results.append(result)
            
            # Be nice to the server
            if i < len(inventory_ids):
                print(f"\nWaiting {delay} seconds...")
                time.sleep(delay)
        
        # Save summary
        summary_file = self.data_dir / 'summary.json'
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✓ All scraping complete!")
        print(f"  Total coins scraped: {len(results)}")
        print(f"  Summary saved: {summary_file}")
        print(f"{'='*60}")
        
        return results


def main():
    """Main function with example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test David Lawrence scraper')
    parser.add_argument('--inventory-id', '-i', default='789518', help='Inventory ID to scrape')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    scraper = DavidLawrenceScraper()
    
    # Example: Scrape single coin
    print("Example: Scraping single coin")
    print(f"Debug mode: {args.debug}")
    coin_data = scraper.scrape_coin(args.inventory_id, debug=args.debug)
    
    if coin_data:
        print("\n" + "="*60)
        print("Scraped Data:")
        print("="*60)
        for key, value in coin_data.items():
            if key != 'images':
                print(f"{key}: {value}")
            else:
                print(f"images: {len(value)} files")
        
        if args.debug and coin_data.get('images'):
            print("\nImage files:")
            for img_path in coin_data['images']:
                print(f"  - {img_path}")
    
    # Example: Scrape multiple coins
    # Uncomment to use:
    # inventory_ids = ['789518', '789519', '789520']
    # results = scraper.scrape_multiple(inventory_ids, delay=2)


if __name__ == '__main__':
    main()

