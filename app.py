"""Flask application for PCGS coin scraper."""
import json
import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from pcgs_api import PCGSClient

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-secret-key-change-in-production'

# Paths
DATA_DIR = 'data'
IMAGES_DIR = 'images'
CERT_FILE = os.path.join(DATA_DIR, 'cert_numbers.json')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Initialize PCGS client
pcgs_client = PCGSClient()


def load_cert_data():
    """Load cert numbers and their data from JSON file."""
    if os.path.exists(CERT_FILE):
        try:
            with open(CERT_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_cert_data(data):
    """Save cert numbers and their data to JSON file."""
    with open(CERT_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def normalize_grade(grade):
    """Normalize grade string for folder names."""
    if not grade:
        return 'unknown'
    # Remove spaces, convert to lowercase
    return grade.replace(' ', '').lower()


def extract_coin_info(coin_facts):
    """Extract relevant information from coin facts."""
    if not coin_facts:
        return None
    
    # The API response structure may vary, adjust as needed
    info = {
        'grade': coin_facts.get('Grade') or coin_facts.get('grade') or 'Unknown',
        'year': coin_facts.get('Year') or coin_facts.get('year') or 'Unknown',
        'denomination': coin_facts.get('Denomination') or coin_facts.get('denomination') or 'Unknown',
        'variety': coin_facts.get('Variety') or coin_facts.get('variety') or '',
        'designation': coin_facts.get('Designation') or coin_facts.get('designation') or '',
    }
    
    return info


@app.route('/')
def index():
    """Render the main page."""
    cert_data = load_cert_data()
    return render_template('index.html', coins=cert_data)


@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve images from the images directory."""
    return send_from_directory(IMAGES_DIR, filename)


@app.route('/api/add_cert', methods=['POST'])
def add_cert():
    """Add a new cert number and fetch its data."""
    data = request.get_json()
    cert_no = data.get('cert_no', '').strip()
    
    if not cert_no:
        return jsonify({'success': False, 'error': 'Cert number is required'}), 400
    
    # Check if already exists - skip instead of error
    cert_data = load_cert_data()
    if cert_no in cert_data:
        return jsonify({'success': True, 'skipped': True, 'data': cert_data[cert_no]})
    
    # Fetch coin facts
    coin_facts = pcgs_client.get_coin_facts(cert_no)
    if not coin_facts:
        return jsonify({'success': False, 'error': 'Could not fetch coin data from PCGS API'}), 404
    
    # Extract info
    coin_info = extract_coin_info(coin_facts)
    if not coin_info:
        return jsonify({'success': False, 'error': 'Could not parse coin data'}), 500
    
    # Get images
    images_data = pcgs_client.get_images(cert_no)
    image_paths = []
    
    if images_data and images_data.get('ImageReady'):
        # Extract image URLs from the API response
        image_urls = []
        seen_urls = set()  # Track unique URLs to prevent duplicates
        
        # Parse the Images array
        if 'Images' in images_data and isinstance(images_data['Images'], list):
            for idx, img in enumerate(images_data['Images']):
                if isinstance(img, dict) and 'Url' in img:
                    url = img['Url']
                    
                    # Skip if we've already seen this URL
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    
                    description = img.get('Description', '').lower()
                    resolution = img.get('Resolution', '').lower()
                    
                    # Determine side from description
                    if 'obverse' in description or 'obv' in description:
                        side = 'obverse'
                    elif 'reverse' in description or 'rev' in description:
                        side = 'reverse'
                    elif 'trueview' in description.lower():
                        side = 'trueview'
                    else:
                        side = resolution if resolution else f'img{idx}'
                    
                    image_urls.append((side, url, idx))
        
        # Download images
        normalized_grade = normalize_grade(coin_info['grade'])
        grade_dir = os.path.join(IMAGES_DIR, normalized_grade)
        os.makedirs(grade_dir, exist_ok=True)
        
        for side, url, idx in image_urls:
            if url:
                # Determine extension
                ext = '.jpg'
                if '.png' in url.lower():
                    ext = '.png'
                elif '.gif' in url.lower():
                    ext = '.gif'
                
                # Create filename: <grade>-<denomination>-<certNo>-<side>-<idx>.<ext>
                normalized_denom = coin_info['denomination'].replace(' ', '').replace('/', '-').lower()
                filename = f"{normalized_grade}-{normalized_denom}-{cert_no}-{side}-{idx}{ext}"
                
                # Store obverse/reverse in separate subfolders (not shown in UI)
                if side == 'obverse':
                    side_dir = os.path.join(grade_dir, 'obverse')
                    os.makedirs(side_dir, exist_ok=True)
                    save_path = os.path.join(side_dir, filename)
                elif side == 'reverse':
                    side_dir = os.path.join(grade_dir, 'reverse')
                    os.makedirs(side_dir, exist_ok=True)
                    save_path = os.path.join(side_dir, filename)
                else:
                    # Other images organized by resolution (TrueView, high-res)
                    # Store in resolution subfolder (e.g., 5757x2905, 6000x3000)
                    resolution_dir = os.path.join(grade_dir, side)
                    os.makedirs(resolution_dir, exist_ok=True)
                    save_path = os.path.join(resolution_dir, filename)
                
                if pcgs_client.download_image(url, save_path):
                    image_paths.append(save_path)
    
    # Save data
    cert_data[cert_no] = {
        'cert_no': cert_no,
        'info': coin_info,
        'images': image_paths,
        'raw_data': coin_facts
    }
    save_cert_data(cert_data)
    
    return jsonify({
        'success': True,
        'data': cert_data[cert_no]
    })


@app.route('/api/delete_cert', methods=['POST'])
def delete_cert():
    """Delete a cert number and its images."""
    data = request.get_json()
    cert_no = data.get('cert_no', '').strip()
    
    if not cert_no:
        return jsonify({'success': False, 'error': 'Cert number is required'}), 400
    
    cert_data = load_cert_data()
    
    if cert_no not in cert_data:
        return jsonify({'success': False, 'error': 'Cert number not found'}), 404
    
    # Delete images
    coin_data = cert_data[cert_no]
    for image_path in coin_data.get('images', []):
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
        except Exception as e:
            print(f"Error deleting image {image_path}: {e}")
    
    # Remove from data
    del cert_data[cert_no]
    save_cert_data(cert_data)
    
    return jsonify({'success': True})


@app.route('/api/coins')
def get_coins():
    """Get all coins data."""
    cert_data = load_cert_data()
    
    # Filter and select highest resolution images for UI display
    for cert_no, coin in cert_data.items():
        if 'images' in coin:
            # Filter out obverse/reverse images
            ui_images = [
                img for img in coin['images'] 
                if '/obverse/' not in img and '/reverse/' not in img
            ]
            
            # Find highest resolution images
            # Group by base filename and select highest resolution
            resolution_map = {}
            for img_path in ui_images:
                # Extract resolution from path (e.g., images/grade/5757x2905/filename.jpg)
                parts = img_path.split('/')
                if len(parts) >= 3:
                    resolution = parts[2]  # The resolution folder name
                    # Parse resolution to get pixel count (e.g., 5757x2905 -> 16,608,285)
                    if 'x' in resolution:
                        try:
                            w, h = resolution.split('x')
                            pixel_count = int(w) * int(h)
                            
                            # Use cert number as key to group images
                            if cert_no not in resolution_map or pixel_count > resolution_map[cert_no][1]:
                                resolution_map[cert_no] = (img_path, pixel_count)
                        except:
                            # If parsing fails, just add the image
                            if cert_no not in resolution_map:
                                resolution_map[cert_no] = (img_path, 0)
            
            # Use only the highest resolution image
            if resolution_map and cert_no in resolution_map:
                coin['images'] = [resolution_map[cert_no][0]]
            else:
                coin['images'] = ui_images[:1] if ui_images else []  # Fallback to first image
    
    return jsonify(cert_data)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

