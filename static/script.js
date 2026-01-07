// Load coins when page loads
document.addEventListener('DOMContentLoaded', function() {
    loadCoins();
    
    // Allow Enter key to submit
    document.getElementById('certInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            addCert();
        }
    });
});

// Switch between tabs
function switchTab(tab) {
    const singleTab = document.getElementById('singleTab');
    const bulkTab = document.getElementById('bulkTab');
    const singleContent = document.getElementById('singleContent');
    const bulkContent = document.getElementById('bulkContent');
    
    if (tab === 'single') {
        singleTab.className = 'flex-1 py-4 px-6 text-center font-semibold text-lg transition-all duration-200 rounded-tl-2xl border-b-4 border-purple-600 text-purple-600';
        bulkTab.className = 'flex-1 py-4 px-6 text-center font-semibold text-lg transition-all duration-200 rounded-tr-2xl text-gray-500 hover:text-purple-600';
        singleContent.classList.remove('hidden');
        bulkContent.classList.add('hidden');
    } else {
        bulkTab.className = 'flex-1 py-4 px-6 text-center font-semibold text-lg transition-all duration-200 rounded-tr-2xl border-b-4 border-purple-600 text-purple-600';
        singleTab.className = 'flex-1 py-4 px-6 text-center font-semibold text-lg transition-all duration-200 rounded-tl-2xl text-gray-500 hover:text-purple-600';
        bulkContent.classList.remove('hidden');
        singleContent.classList.add('hidden');
    }
}

// Show message
function showMessage(text, type) {
    const messageContainer = document.getElementById('messageContainer');
    const messageId = Date.now();
    
    const bgColor = type === 'success' ? 'bg-green-100 border-green-500 text-green-800' : 
                    type === 'error' ? 'bg-red-100 border-red-500 text-red-800' :
                    'bg-blue-100 border-blue-500 text-blue-800';
    
    const icon = type === 'success' ? 'fa-check-circle' : 
                 type === 'error' ? 'fa-exclamation-circle' : 
                 'fa-info-circle';
    
    const messageHTML = `
        <div id="msg-${messageId}" class="border-l-4 ${bgColor} p-4 rounded-lg shadow-lg mb-4 animate-fade-in">
            <div class="flex items-center">
                <i class="fas ${icon} text-2xl mr-3"></i>
                <p class="font-semibold">${text}</p>
            </div>
        </div>
    `;
    
    messageContainer.insertAdjacentHTML('beforeend', messageHTML);
    
    setTimeout(() => {
        const msg = document.getElementById(`msg-${messageId}`);
        if (msg) {
            msg.style.opacity = '0';
            msg.style.transform = 'translateX(100%)';
            msg.style.transition = 'all 0.3s ease';
            setTimeout(() => msg.remove(), 300);
        }
    }, 5000);
}

// Add cert number
async function addCert() {
    const input = document.getElementById('certInput');
    const certNo = input.value.trim();
    const addBtn = document.getElementById('addBtn');
    
    if (!certNo) {
        showMessage('Please enter a cert number', 'error');
        return;
    }
    
    // Disable button and show loading
    addBtn.disabled = true;
    addBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Adding...';
    
    try {
        const response = await fetch('/api/add_cert', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ cert_no: certNo })
        });
        
        const data = await response.json();
        
        if (data.success) {
            if (data.skipped) {
                showMessage(`Cert ${certNo} already exists - skipped`, 'info');
            } else {
                showMessage(`Successfully added cert ${certNo}!`, 'success');
            }
            input.value = '';
            loadCoins();
        } else {
            showMessage(data.error || 'Failed to add cert number', 'error');
        }
    } catch (error) {
        showMessage('Error: ' + error.message, 'error');
    } finally {
        addBtn.disabled = false;
        addBtn.innerHTML = '<i class="fas fa-plus mr-2"></i>Add Cert';
    }
}

// Process bulk cert numbers
async function processBulk() {
    const bulkInput = document.getElementById('bulkInput');
    const bulkBtn = document.getElementById('bulkBtn');
    const progressDiv = document.getElementById('bulkProgress');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    
    const text = bulkInput.value.trim();
    if (!text) {
        showMessage('Please enter at least one cert number', 'error');
        return;
    }
    
    // Parse cert numbers (split by newlines, commas, or spaces)
    const certNumbers = text.split(/[\n,\s]+/)
        .map(c => c.trim())
        .filter(c => c.length > 0)
        .filter((value, index, self) => self.indexOf(value) === index); // Remove duplicates
    
    if (certNumbers.length === 0) {
        showMessage('No valid cert numbers found', 'error');
        return;
    }
    
    // Disable button and show progress
    bulkBtn.disabled = true;
    bulkBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Processing...';
    progressDiv.classList.remove('hidden');
    
    let processed = 0;
    let added = 0;
    let skipped = 0;
    let failed = 0;
    
    for (const certNo of certNumbers) {
        try {
            const response = await fetch('/api/add_cert', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ cert_no: certNo })
            });
            
            const data = await response.json();
            
            if (data.success) {
                if (data.skipped) {
                    skipped++;
                } else {
                    added++;
                }
            } else {
                failed++;
                console.error(`Failed to add ${certNo}: ${data.error}`);
            }
        } catch (error) {
            failed++;
            console.error(`Error adding ${certNo}: ${error.message}`);
        }
        
        processed++;
        const percent = (processed / certNumbers.length) * 100;
        progressBar.style.width = percent + '%';
        progressText.textContent = `Processing: ${processed}/${certNumbers.length} (Added: ${added}, Skipped: ${skipped}, Failed: ${failed})`;
    }
    
    // Show final results
    showMessage(`Bulk upload complete! Added: ${added}, Skipped: ${skipped}, Failed: ${failed}`, 'success');
    
    // Reset
    bulkBtn.disabled = false;
    bulkBtn.innerHTML = '<i class="fas fa-cloud-upload-alt mr-2"></i>Process All';
    bulkInput.value = '';
    setTimeout(() => {
        progressDiv.classList.add('hidden');
        progressBar.style.width = '0%';
    }, 3000);
    
    loadCoins();
}

// Delete cert number
async function deleteCert(certNo, event) {
    event.stopPropagation();
    
    if (!confirm(`Are you sure you want to delete cert ${certNo}?`)) {
        return;
    }
    
    try {
        const response = await fetch('/api/delete_cert', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ cert_no: certNo })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showMessage(`Successfully deleted cert ${certNo}`, 'success');
            loadCoins();
        } else {
            showMessage(data.error || 'Failed to delete cert number', 'error');
        }
    } catch (error) {
        showMessage('Error: ' + error.message, 'error');
    }
}

// Load all coins
async function loadCoins() {
    const coinsList = document.getElementById('coinsList');
    coinsList.innerHTML = `
        <div class="col-span-full flex flex-col items-center justify-center py-12">
            <div class="animate-spin rounded-full h-16 w-16 border-b-4 border-purple-600 mb-4"></div>
            <p class="text-gray-600 text-lg font-semibold">Loading your collection...</p>
        </div>
    `;
    
    try {
        const response = await fetch('/api/coins');
        const coins = await response.json();
        
        const coinCount = Object.keys(coins).length;
        document.getElementById('coinCount').textContent = coinCount;
        
        if (coinCount === 0) {
            coinsList.innerHTML = `
                <div class="col-span-full text-center py-16">
                    <i class="fas fa-inbox text-6xl text-gray-300 mb-4"></i>
                    <p class="text-gray-500 text-xl font-semibold">No coins yet</p>
                    <p class="text-gray-400 mt-2">Add your first cert number above to get started!</p>
                </div>
            `;
            return;
        }
        
        coinsList.innerHTML = '';
        
        // Sort by cert number
        const sortedCoins = Object.values(coins).sort((a, b) => {
            return parseInt(b.cert_no) - parseInt(a.cert_no);
        });
        
        sortedCoins.forEach(coin => {
            coinsList.insertAdjacentHTML('beforeend', createCoinCard(coin));
        });
    } catch (error) {
        coinsList.innerHTML = `
            <div class="col-span-full text-center py-16">
                <i class="fas fa-exclamation-triangle text-6xl text-red-400 mb-4"></i>
                <p class="text-red-600 text-xl font-semibold">Error loading coins</p>
                <p class="text-gray-500 mt-2">${error.message}</p>
            </div>
        `;
    }
}

// Create coin card element
function createCoinCard(coin) {
    const info = coin.info || {};
    const images = coin.images || [];
    
    const gradeColors = {
        'ms': 'bg-yellow-100 text-yellow-800',
        'pr': 'bg-blue-100 text-blue-800',
        'au': 'bg-orange-100 text-orange-800',
        'xf': 'bg-green-100 text-green-800',
        'vf': 'bg-teal-100 text-teal-800',
        'f': 'bg-indigo-100 text-indigo-800',
        'default': 'bg-gray-100 text-gray-800'
    };
    
    const gradePrefix = info.grade ? info.grade.toLowerCase().substring(0, 2) : '';
    const gradeColor = gradeColors[gradePrefix] || gradeColors['default'];
    
    return `
        <div class="bg-gradient-to-br from-white to-gray-50 rounded-xl shadow-lg hover:shadow-2xl transform hover:-translate-y-2 transition-all duration-300 overflow-hidden border border-gray-200">
            <div class="p-6">
                <div class="flex items-center justify-between mb-4">
                    <div class="flex items-center gap-3">
                        <i class="fas fa-certificate text-3xl text-purple-600"></i>
                        <span class="text-2xl font-bold text-gray-800">#${coin.cert_no}</span>
                    </div>
                    <button onclick="deleteCert('${coin.cert_no}', event)" 
                            class="text-red-500 hover:text-red-700 hover:bg-red-50 p-2 rounded-lg transition-all duration-200">
                        <i class="fas fa-trash-alt text-xl"></i>
                    </button>
                </div>
                
                <div class="space-y-3 mb-4">
                    <div class="flex items-center justify-between py-2 border-b border-gray-200">
                        <span class="text-gray-600 font-semibold flex items-center">
                            <i class="fas fa-award mr-2 text-purple-500"></i>Grade
                        </span>
                        <span class="px-3 py-1 rounded-full font-bold ${gradeColor}">
                            ${info.grade || 'N/A'}
                        </span>
                    </div>
                    <div class="flex items-center justify-between py-2 border-b border-gray-200">
                        <span class="text-gray-600 font-semibold flex items-center">
                            <i class="fas fa-calendar mr-2 text-purple-500"></i>Year
                        </span>
                        <span class="text-gray-800 font-bold">${info.year || 'N/A'}</span>
                    </div>
                    <div class="flex items-center justify-between py-2 border-b border-gray-200">
                        <span class="text-gray-600 font-semibold flex items-center">
                            <i class="fas fa-coins mr-2 text-purple-500"></i>Denomination
                        </span>
                        <span class="text-gray-800 font-bold">${info.denomination || 'N/A'}</span>
                    </div>
                    ${info.variety ? `
                    <div class="flex items-center justify-between py-2 border-b border-gray-200">
                        <span class="text-gray-600 font-semibold flex items-center">
                            <i class="fas fa-star mr-2 text-purple-500"></i>Variety
                        </span>
                        <span class="text-gray-800 font-bold">${info.variety}</span>
                    </div>
                    ` : ''}
                    ${info.designation ? `
                    <div class="flex items-center justify-between py-2 border-b border-gray-200">
                        <span class="text-gray-600 font-semibold flex items-center">
                            <i class="fas fa-tag mr-2 text-purple-500"></i>Designation
                        </span>
                        <span class="text-gray-800 font-bold">${info.designation}</span>
                    </div>
                    ` : ''}
                </div>
                
                ${images.length > 0 ? `
                <div class="mt-4">
                    <h4 class="text-gray-700 font-semibold mb-3 flex items-center">
                        <i class="fas fa-images mr-2 text-purple-500"></i>
                        Images (${images.length})
                    </h4>
                    <div class="grid grid-cols-${Math.min(images.length, 2)} gap-3">
                        ${images.map(img => `
                            <div class="relative group cursor-pointer" onclick="openImageModal('/${img}')">
                                <img src="/${img}" alt="Coin image" 
                                     class="w-full h-32 object-cover rounded-lg border-2 border-gray-200 group-hover:border-purple-500 transition-all duration-200" />
                                <div class="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 rounded-lg transition-all duration-200 flex items-center justify-center">
                                    <i class="fas fa-search-plus text-white text-2xl opacity-0 group-hover:opacity-100 transition-all duration-200"></i>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
                ` : `
                <div class="mt-4 text-center py-6 bg-gray-100 rounded-lg">
                    <i class="fas fa-image text-3xl text-gray-400 mb-2"></i>
                    <p class="text-gray-500 text-sm">No images available</p>
                </div>
                `}
            </div>
        </div>
    `;
}

// Image modal functions
function openImageModal(imagePath) {
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImage');
    
    modal.classList.remove('hidden');
    modalImg.src = imagePath;
    document.body.style.overflow = 'hidden'; // Prevent scrolling
}

function closeImageModal() {
    const modal = document.getElementById('imageModal');
    modal.classList.add('hidden');
    document.body.style.overflow = 'auto'; // Re-enable scrolling
}













