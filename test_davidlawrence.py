#!/usr/bin/env python3
"""
Quick test script for David Lawrence scraper
Helps diagnose issues with specific inventory IDs
"""

import requests
from bs4 import BeautifulSoup
import json

def test_page(inventory_id):
    """Test what we can extract from a David Lawrence page."""
    url = f"https://davidlawrence.com/inventory/{inventory_id}"
    
    print("="*70)
    print(f"Testing: {url}")
    print("="*70)
    
    # Fetch page
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Error fetching page: {e}")
        return
    
    print(f"✓ Page fetched successfully")
    print(f"  Status code: {response.status_code}")
    print(f"  Content length: {len(response.text)} bytes")
    print(f"  Content type: {response.headers.get('content-type')}")
    
    # Save raw HTML
    with open(f'debug_{inventory_id}.html', 'w', encoding='utf-8') as f:
        f.write(response.text)
    print(f"  Saved to: debug_{inventory_id}.html")
    
    # Parse
    soup = BeautifulSoup(response.text, 'html.parser')
    
    print("\n" + "="*70)
    print("EXTRACTED DATA")
    print("="*70)
    
    # Title
    print("\n1. TITLE/DESCRIPTION:")
    title = soup.find('h1')
    if title:
        print(f"   H1: {title.get_text(strip=True)}")
    else:
        print("   ❌ No H1 found")
    
    meta_title = soup.find('meta', property='og:title')
    if meta_title:
        print(f"   Meta title: {meta_title.get('content')}")
    
    # Images
    print("\n2. IMAGES:")
    all_imgs = soup.find_all('img')
    print(f"   Total <img> tags: {len(all_imgs)}")
    
    if all_imgs:
        print("\n   First 10 images:")
        for i, img in enumerate(all_imgs[:10], 1):
            src = img.get('src') or img.get('data-src') or img.get('data-zoom-image')
            alt = img.get('alt', '')[:50]
            print(f"   {i}. {src}")
            if alt:
                print(f"      Alt: {alt}")
    
    # Look for image data in JSON-LD
    print("\n3. STRUCTURED DATA:")
    scripts = soup.find_all('script', type='application/ld+json')
    if scripts:
        print(f"   Found {len(scripts)} JSON-LD scripts")
        for i, script in enumerate(scripts, 1):
            try:
                data = json.loads(script.string)
                if 'image' in data:
                    print(f"   Script {i} has image: {data['image']}")
            except:
                pass
    else:
        print("   No JSON-LD found")
    
    # Text content sample
    print("\n4. TEXT CONTENT (first 500 chars):")
    text = soup.get_text(separator=' ', strip=True)
    print(f"   {text[:500]}...")
    
    # Check for JavaScript frameworks
    print("\n5. JAVASCRIPT DETECTION:")
    if 'react' in response.text.lower():
        print("   ⚠️  React detected - page may be client-side rendered")
    if 'vue' in response.text.lower():
        print("   ⚠️  Vue detected - page may be client-side rendered")
    if 'angular' in response.text.lower():
        print("   ⚠️  Angular detected - page may be client-side rendered")
    if '__NEXT_DATA__' in response.text:
        print("   ⚠️  Next.js detected - page may be client-side rendered")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if len(all_imgs) == 0:
        print("❌ No images found - page likely uses JavaScript to load images")
        print("   Solution: Need to use Selenium or Playwright to render JavaScript")
    elif len(all_imgs) < 5:
        print("⚠️  Few images found - check debug HTML file")
    else:
        print("✓ Images found - scraper should work")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        inventory_id = sys.argv[1]
    else:
        inventory_id = '789518'
    
    test_page(inventory_id)
    
    print("\nNext steps:")
    print("1. Open debug_*.html in browser to see actual page content")
    print("2. Check if images are loaded via JavaScript")
    print("3. If JavaScript-heavy, we need Selenium/Playwright")




