#!/usr/bin/env python3
"""
Quick test to verify pagination works - discovers first 2 pages
"""

from davidlawrence_selenium_bulk import DavidLawrenceBulkScraper
import sys

def test_pagination(listing_url):
    """Test pagination by discovering first 2 pages."""
    print("="*60)
    print("TESTING PAGINATION - First 2 Pages")
    print("="*60)
    
    scraper = DavidLawrenceBulkScraper(headless=False)  # Visible browser
    
    try:
        # Discover only first 2 pages
        inventory_ids = scraper.discover_inventory_ids(listing_url, max_pages=2)
        
        print("\n" + "="*60)
        print("PAGINATION TEST RESULTS")
        print("="*60)
        print(f"✓ Successfully discovered {len(inventory_ids)} auction lot IDs")
        print(f"✓ Expected: ~48 IDs (24 per page × 2 pages)")
        
        if len(inventory_ids) >= 40:
            print(f"✓ PAGINATION WORKS! Found enough IDs from 2 pages")
        else:
            print(f"⚠️  Warning: Only found {len(inventory_ids)} IDs")
            print(f"   Expected ~48 (24 per page)")
        
        print("\nFirst 10 IDs found:")
        for i, inv_id in enumerate(inventory_ids[:10], 1):
            print(f"  {i}. {inv_id}")
        
        if len(inventory_ids) > 10:
            print(f"  ... and {len(inventory_ids) - 10} more")
        
        print("\n" + "="*60)
        print("Ready to run full scrape!")
        print("="*60)
        print(f"\nCommand to scrape all:")
        print(f'python davidlawrence_selenium_bulk.py "{listing_url}"')
        print()
        
    finally:
        scraper.driver.quit()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_pagination.py <listing_url>")
        print("\nExample:")
        print('python test_pagination.py "https://davidlawrence.com/auctions?coinCategory=25&soldInPastAuction=true"')
        sys.exit(1)
    
    test_pagination(sys.argv[1])

