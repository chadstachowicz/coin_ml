"""
David Lawrence Bulk Scraper
Read inventory IDs from file OR discover from listing URL and scrape in bulk
"""

from davidlawrence_scraper import DavidLawrenceScraper
import argparse
from pathlib import Path
import re


def read_inventory_ids(filepath):
    """Read inventory IDs from a text file (one per line or comma-separated)."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by newlines, commas, or spaces
    ids = re.split(r'[\n,\s]+', content)
    ids = [id.strip() for id in ids if id.strip()]
    
    return ids


def is_url(text):
    """Check if text is a URL."""
    return text.startswith('http://') or text.startswith('https://')


def main():
    parser = argparse.ArgumentParser(
        description='Bulk scrape David Lawrence Rare Coins',
        epilog='''
Examples:
  # From file with IDs
  %(prog)s inventory_ids.txt
  
  # From listing URL
  %(prog)s "https://davidlawrence.com/auctions?coinCategory=25&soldInPastAuction=true"
  
  # From listing URL with max pages
  %(prog)s "https://davidlawrence.com/auctions?coinCategory=25" --max-pages 5
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('input', help='File with inventory IDs OR listing URL to discover IDs')
    parser.add_argument('--output', '-o', default='davidlawrence_coins', help='Output directory')
    parser.add_argument('--delay', '-d', type=float, default=2.0, help='Delay between requests (seconds)')
    parser.add_argument('--max-pages', '-m', type=int, help='Max pages to scrape from listing URL (default: all)')
    parser.add_argument('--save-ids', '-s', help='Save discovered IDs to file')
    
    args = parser.parse_args()
    
    # Create scraper
    scraper = DavidLawrenceScraper(output_dir=args.output)
    
    # Get inventory IDs - either from file or by discovering from URL
    if is_url(args.input):
        # Discover IDs from listing URL
        print(f"Mode: Discover inventory IDs from listing URL")
        inventory_ids = scraper.discover_inventory_ids(args.input, max_pages=args.max_pages)
        
        if not inventory_ids:
            print("❌ Error: No inventory IDs found on listing page")
            return
        
        # Optionally save discovered IDs
        if args.save_ids:
            with open(args.save_ids, 'w') as f:
                for inv_id in inventory_ids:
                    f.write(f"{inv_id}\n")
            print(f"✓ Saved {len(inventory_ids)} IDs to {args.save_ids}\n")
    else:
        # Read from file
        if not Path(args.input).exists():
            print(f"❌ Error: File not found: {args.input}")
            return
        
        print(f"Mode: Read inventory IDs from file")
        inventory_ids = read_inventory_ids(args.input)
        
        if not inventory_ids:
            print("❌ Error: No inventory IDs found in file")
            return
    
    print(f"\nFound {len(inventory_ids)} inventory IDs")
    print(f"Output directory: {args.output}")
    print(f"Delay between requests: {args.delay}s")
    print()
    
    # Confirm
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Scrape all coins
    results = scraper.scrape_multiple(inventory_ids, delay=args.delay)
    
    print(f"\n✓ Complete! Scraped {len(results)} coins")
    print(f"  Images: {args.output}/images/")
    print(f"  Data: {args.output}/data/")


if __name__ == '__main__':
    main()

