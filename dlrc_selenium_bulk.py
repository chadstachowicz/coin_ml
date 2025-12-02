"""
David Lawrence Selenium Bulk Scraper
Discovers inventory IDs from listing URLs and scrapes all coins
"""

from davidlawrence_selenium_scraper import DavidLawrenceSeleniumScraper
from selenium.webdriver.common.by import By
import os
import re
import time
import argparse
import json


class DavidLawrenceBulkScraper(DavidLawrenceSeleniumScraper):
    """Extended scraper with bulk discovery and scraping."""
    
    def discover_inventory_ids(self, listing_url, max_pages=None, cache_file=None):
        """
        Discover all inventory IDs from a listing/auction/category page.
        
        Args:
            listing_url: URL of the listing page
            max_pages: Maximum number of pages to scrape (None = all)
            cache_file: File to save/load discovered IDs
            
        Returns:
            list: List of unique inventory IDs
        """
        # Check if we have cached IDs
        if cache_file and os.path.exists(cache_file):
            print(f"\n{'='*60}")
            print(f"FOUND CACHED IDs")
            print(f"{'='*60}")
            print(f"Loading from: {cache_file}")
            
            with open(cache_file, 'r') as f:
                cached_ids = [line.strip() for line in f if line.strip()]
            
            print(f"Loaded {len(cached_ids)} IDs from cache")
            print(f"{'='*60}\n")
            
            response = input("Use cached IDs? (y/n): ")
            if response.lower() == 'y':
                inventory_list = cached_ids
                # Still need to set detected URL type
                self._detected_url_type = 'auction'
                return inventory_list
            else:
                print("Re-discovering IDs...\n")
        
        inventory_ids = set()
        page = 1
        
        print(f"\n{'='*60}")
        print(f"Discovering inventory IDs from listing")
        print(f"{'='*60}")
        print(f"URL: {listing_url}")
        print(f"Max pages: {max_pages if max_pages else 'All'}")
        if cache_file:
            print(f"Cache file: {cache_file}")
        print(f"{'='*60}\n")
        
        while True:
            # Build URL for current page or click pagination
            if page == 1:
                current_url = listing_url
                print(f"Page {page}: {current_url}")
                
                try:
                    # Load first page
                    self.driver.get(current_url)
                    time.sleep(3)  # Wait for page to load
                    time.sleep(2)  # Wait for content to render
                except Exception as e:
                    print(f"  ❌ Error loading page: {e}")
                    break
            else:
                # For pages 2+, try clicking pagination first (faster than URL navigation)
                print(f"Page {page}: Navigating via pagination...")
                
                try:
                    # Try to click the page number
                    page_button = self.driver.find_elements(By.CSS_SELECTOR, 
                        f'li[data-role="jump-to"][data-target="{page}"][data-disabled="false"]')
                    
                    if page_button:
                        print(f"  Clicking page {page} button")
                        page_button[0].click()
                        time.sleep(3)
                        time.sleep(2)
                    else:
                        # Fallback: use URL with page parameter
                        separator = '&' if '?' in listing_url else '?'
                        current_url = f"{listing_url}{separator}page={page}"
                        print(f"  Loading via URL: {current_url}")
                        self.driver.get(current_url)
                        time.sleep(3)
                        time.sleep(2)
                        
                except Exception as e:
                    print(f"  ❌ Error navigating to page {page}: {e}")
                    break
            
            # Get page source and find auction lot links ONLY
            page_source = self.driver.page_source
            
            # Find all /auctions/lot/XXXXX links (ignore /inventory/ links)
            auction_matches = re.findall(r'/auctions/lot/(\d+)', page_source)
            page_ids = set(auction_matches)
            
            # Also try finding links via Selenium (more reliable)
            try:
                # Look for ONLY /auctions/lot/ links
                links = self.driver.find_elements(By.XPATH, "//a[contains(@href, '/auctions/lot/')]")
                for link in links:
                    href = link.get_attribute('href')
                    match = re.search(r'/auctions/lot/(\d+)', href)
                    if match:
                        page_ids.add(match.group(1))
            except Exception as e:
                print(f"  Warning: Error finding links via Selenium: {e}")
            
            print(f"  Found {len(page_ids)} auction lot IDs (expected ~24 per page)")
            
            if len(page_ids) == 0:
                print(f"  ⚠️  No auction lot links found on this page")
            
            inventory_ids.update(page_ids)
            print(f"  Total unique IDs so far: {len(inventory_ids)}")
            
            # Check if there's a next page by looking at React pagination component
            has_next = False
            
            try:
                # Look for the pagination nav with data-testid
                pagination_nav = self.driver.find_elements(By.CSS_SELECTOR, 
                    'nav[data-testid="pagination-selector"]')
                
                if pagination_nav:
                    print(f"  Found pagination component")
                    
                    # Strategy 1: Look for "jump-next" button that is NOT disabled
                    next_button = self.driver.find_elements(By.CSS_SELECTOR, 
                        'li[data-role="jump-next"][data-disabled="false"]')
                    
                    if next_button:
                        print(f"  ✓ Next button is enabled, more pages available")
                        has_next = True
                    else:
                        # Strategy 2: Check if there are any page numbers > current page
                        next_page_num = page + 1
                        next_page_elements = self.driver.find_elements(By.CSS_SELECTOR, 
                            f'li[data-role="jump-to"][data-target="{next_page_num}"][data-disabled="false"]')
                        
                        if next_page_elements:
                            print(f"  ✓ Found clickable page {next_page_num}")
                            has_next = True
                        else:
                            print(f"  ✗ No more pages (next button disabled)")
                            has_next = False
                else:
                    # Fallback: If we found fewer than expected, probably last page
                    if len(page_ids) < 20:
                        print(f"  Found < 20 items, likely the last page")
                        has_next = False
                    else:
                        print(f"  No pagination found, but found {len(page_ids)} items")
                        # Assume there might be more
                        has_next = True
                        
            except Exception as e:
                print(f"  Error checking pagination: {e}")
                # If error and we found a full page, try next page anyway
                has_next = len(page_ids) >= 20
            
            # Check stopping conditions
            reached_max = max_pages and page >= max_pages
            
            if not has_next or reached_max:
                if reached_max:
                    print(f"\n  Reached max pages ({max_pages})")
                else:
                    print(f"\n  No more pages found")
                break
            
            page += 1
        
        inventory_list = sorted(list(inventory_ids))
        
        # We only search for auction lot URLs
        self._detected_url_type = 'auction'
        
        print(f"\n{'='*60}")
        print(f"✓ Discovery complete!")
        print(f"  Total pages scraped: {page}")
        print(f"  Total auction lot IDs found: {len(inventory_list)}")
        print(f"  URL format: /auctions/lot/")
        print(f"{'='*60}\n")
        
        # Save to cache file if specified
        if cache_file:
            with open(cache_file, 'w') as f:
                for inv_id in inventory_list:
                    f.write(f"{inv_id}\n")
            print(f"✓ Saved {len(inventory_list)} IDs to cache: {cache_file}\n")
        
        # Show preview
        if inventory_list:
            print("Preview (first 20 IDs):")
            for inv_id in inventory_list[:20]:
                print(f"  {inv_id}")
            if len(inventory_list) > 20:
                print(f"  ... and {len(inventory_list) - 20} more")
            print()
        
        return inventory_list
    
    def scrape_from_listing(self, listing_url, max_pages=None, delay=3, wait_time=3, cache_file=None):
        """
        Discover and scrape all coins from a listing URL.
        
        Args:
            listing_url: URL of the listing page
            max_pages: Max pages to discover from (None = all)
            delay: Delay between scraping individual coins
            wait_time: Wait time for each coin page to load
            cache_file: File to save/load discovered IDs
            
        Returns:
            list: List of scraped coin data
        """
        # Step 1: Discover all inventory IDs (with caching)
        inventory_ids = self.discover_inventory_ids(listing_url, max_pages=max_pages, cache_file=cache_file)
        
        if not inventory_ids:
            print("❌ No inventory IDs found")
            return []
        
        print(f"\n{'='*60}")
        print(f"Starting bulk scraping of {len(inventory_ids)} coins")
        print(f"{'='*60}")
        print(f"Delay between coins: {delay}s")
        print(f"Wait time per page: {wait_time}s")
        print(f"Estimated time: {len(inventory_ids) * (delay + wait_time + 5) / 60:.1f} minutes")
        print(f"{'='*60}\n")
        
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return []
        
        # Step 2: Scrape all coins using detected URL format
        url_type = getattr(self, '_detected_url_type', 'inventory')
        print(f"Using URL format: {url_type}\n")
        return self.scrape_multiple(inventory_ids, delay=delay, wait_time=wait_time, url_type=url_type)


def main():
    """Main function for bulk scraping."""
    parser = argparse.ArgumentParser(
        description='David Lawrence Selenium Bulk Scraper',
        epilog='''
Examples:
  # Scrape all coins from an auction category
  %(prog)s "https://davidlawrence.com/auctions?coinCategory=25&soldInPastAuction=true"
  
  # Limit to first 3 pages
  %(prog)s "https://davidlawrence.com/auctions?coinCategory=25" --max-pages 3
  
  # Custom output and timing
  %(prog)s "https://davidlawrence.com/auctions?coinCategory=25" -o my_coins --delay 5 --wait 5
  
  # Save discovered IDs without scraping
  %(prog)s "https://davidlawrence.com/auctions?coinCategory=25" --discover-only --save-ids found.txt
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('listing_url', help='Listing/auction/category URL')
    parser.add_argument('--output', '-o', default='davidlawrence_coins', help='Output directory')
    parser.add_argument('--max-pages', '-m', type=int, help='Max pages to discover from')
    parser.add_argument('--delay', '-d', type=float, default=3.0, help='Delay between coins (seconds)')
    parser.add_argument('--wait', '-w', type=int, default=3, help='Wait time for page load (seconds)')
    parser.add_argument('--cache-file', '-c', default='discovered_ids.txt', help='Cache file for discovered IDs (default: discovered_ids.txt)')
    parser.add_argument('--save-ids', '-s', help='DEPRECATED: Use --cache-file instead')
    parser.add_argument('--discover-only', action='store_true', help='Only discover IDs, don\'t scrape')
    parser.add_argument('--visible', '-v', action='store_true', help='Show browser (not headless)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("DAVID LAWRENCE SELENIUM BULK SCRAPER")
    print("="*60)
    
    scraper = DavidLawrenceBulkScraper(output_dir=args.output, headless=not args.visible)
    
    # Handle deprecated --save-ids argument
    cache_file = args.save_ids if args.save_ids else args.cache_file
    
    try:
        if args.discover_only:
            # Only discover IDs
            inventory_ids = scraper.discover_inventory_ids(
                args.listing_url, 
                max_pages=args.max_pages,
                cache_file=cache_file
            )
            
            print(f"\n✓ Discovery complete! Found {len(inventory_ids)} coins")
            print(f"✓ IDs saved to: {cache_file}")
        else:
            # Discover and scrape (with caching)
            results = scraper.scrape_from_listing(
                args.listing_url,
                max_pages=args.max_pages,
                delay=args.delay,
                wait_time=args.wait,
                cache_file=cache_file
            )
            
            print(f"\n{'='*60}")
            print(f"✓ BULK SCRAPING COMPLETE!")
            print(f"{'='*60}")
            print(f"Total coins scraped: {len(results)}")
            print(f"Images: {args.output}/images/")
            print(f"Data: {args.output}/data/")
            print(f"{'='*60}")
    
    finally:
        scraper.driver.quit()


if __name__ == '__main__':
    main()

