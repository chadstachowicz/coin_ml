# Selenium Scraper Setup

David Lawrence uses JavaScript to render content, so we need Selenium to handle it.

## Installation

```bash
# Install Selenium and Chrome driver manager
pip install selenium webdriver-manager
```

That's it! The scraper will automatically download and manage ChromeDriver for you.

## Usage

### Single Coin

```bash
python davidlawrence_selenium_scraper.py --inventory-id 789518
```

### Show Browser (Debug)

Watch the browser work (not headless mode):

```bash
python davidlawrence_selenium_scraper.py --inventory-id 789518 --visible
```

### Adjust Wait Time

If pages load slowly:

```bash
python davidlawrence_selenium_scraper.py --inventory-id 789518 --wait 10
```

## Options

```bash
python davidlawrence_selenium_scraper.py [OPTIONS]

Options:
  --inventory-id, -i ID    Inventory ID to scrape (default: 789518)
  --wait, -w SECONDS       Wait time for page load (default: 5)
  --visible, -v            Show browser window (not headless)
```

## How It Works

1. **Launches Chrome** (headless by default)
2. **Loads the page** and waits for JavaScript to execute
3. **Extracts content** after page is fully rendered
4. **Downloads images** at full size
5. **Saves data** to JSON

## Troubleshooting

### ChromeDriver issues

The script uses `webdriver-manager` which automatically:
- Downloads correct ChromeDriver version
- Manages driver updates
- Handles compatibility

If you still have issues:

```bash
# Manually install ChromeDriver
# On Mac:
brew install chromedriver

# On Linux:
sudo apt-get install chromium-chromedriver
```

### Page not loading

Increase wait time:

```bash
python davidlawrence_selenium_scraper.py --wait 10
```

### Watch it work

Use visible mode to see what's happening:

```bash
python davidlawrence_selenium_scraper.py --visible
```

### No images found

- Check if page actually has images (some coins might not)
- Try longer wait time
- Use visible mode to see if images load

## Bulk Scraping

For bulk scraping with Selenium (coming soon):

```python
from davidlawrence_selenium_scraper import DavidLawrenceSeleniumScraper

scraper = DavidLawrenceSeleniumScraper()

inventory_ids = ['789518', '789519', '789520']
results = scraper.scrape_multiple(inventory_ids, delay=3, wait_time=5)
```

## Performance

Selenium is slower than regular scraping because it:
- Launches a browser
- Executes JavaScript
- Waits for page to render

Expect:
- **Single coin**: 5-10 seconds
- **10 coins**: 1-2 minutes
- **100 coins**: 10-20 minutes

To speed up:
- Reduce wait time if pages load fast
- Use headless mode (default)
- Scrape during off-peak hours

## Comparison

| Feature | Regular Scraper | Selenium Scraper |
|---------|----------------|------------------|
| **Speed** | Fast (< 1s) | Slow (5-10s) |
| **JavaScript** | ❌ No | ✅ Yes |
| **Works on DL** | ❌ No | ✅ Yes |
| **Setup** | Simple | Needs Chrome |
| **Resource Use** | Low | High |

For David Lawrence, you **must** use Selenium because the site is JavaScript-rendered.

## Next Steps

Once you've verified it works:

1. Test with your inventory IDs
2. Check image quality
3. Run bulk scraping if needed
4. Integrate with training pipeline

---

Now you can scrape JavaScript-heavy websites! 🚀




