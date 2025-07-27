document.addEventListener('DOMContentLoaded', function () {
    let taskId = null;
    let pollInterval = null;
    
    function startUmapGeneration() {
        const placeholder = document.getElementById('umap-placeholder');
        
        // Start UMAP generation
        fetch('/generate-umap')
            .then(response => response.json())
            .then(data => {
                if (data.success && data.image) {
                    // Immediate result (cached)
                    displayUmapImage(data.image);
                } else if (data.status === 'started' || data.status === 'running') {
                    // Task started or is running, start polling
                    taskId = `umap_${Date.now()}`; // Simple task ID for polling
                    startPolling();
                } else if (data.error) {
                    showError('Error loading UMAP: ' + data.error);
                }
            })
            .catch(err => {
                showError('UMAP generation failed to start.');
                console.error('UMAP fetch error:', err);
            });
    }
    
    function startPolling() {
        const placeholder = document.getElementById('umap-placeholder');
        
        // Update placeholder to show polling status
        placeholder.innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-primary mb-2" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <div class="text-muted">Generating UMAP visualization...</div>
                <small class="text-muted">This may take a few moments</small>
            </div>
        `;
        
        pollInterval = setInterval(checkUmapStatus, 2000); // Poll every 2 seconds
    }
    
    function checkUmapStatus() {
        fetch('/generate-umap')
            .then(response => response.json())
            .then(data => {
                if (data.success && data.image) {
                    // UMAP is ready
                    clearInterval(pollInterval);
                    displayUmapImage(data.image);
                } else if (data.status === 'running') {
                    // Still processing, keep waiting
                    console.log('UMAP still generating...');
                } else if (data.error) {
                    // Error occurred
                    clearInterval(pollInterval);
                    showError('Error generating UMAP: ' + data.error);
                }
            })
            .catch(err => {
                clearInterval(pollInterval);
                showError('Failed to check UMAP status.');
                console.error('UMAP status check error:', err);
            });
    }
    
    function displayUmapImage(imageData) {
        const placeholder = document.getElementById('umap-placeholder');
        const img = document.createElement('img');
        img.src = 'data:image/png;base64,' + imageData;
        img.className = 'img-fluid rounded shadow-sm w-100';
        img.style.height = '100%';
        img.style.objectFit = 'contain';
        img.style.maxHeight = '400px';
        
        // Fade in effect
        img.style.opacity = '0';
        img.style.transition = 'opacity 0.5s ease-in';
        
        placeholder.replaceWith(img);
        
        // Trigger fade in
        setTimeout(() => {
            img.style.opacity = '1';
        }, 50);
    }
    
    function showError(message) {
        const placeholder = document.getElementById('umap-placeholder');
        placeholder.innerHTML = `
            <div class="text-center text-danger">
                <i class="fas fa-exclamation-triangle mb-2"></i>
                <div>${message}</div>
                <button class="btn btn-sm btn-outline-primary mt-2" onclick="location.reload()">
                    <i class="fas fa-redo me-1"></i>Retry
                </button>
            </div>
        `;
    }
    
    // Cleanup polling if user navigates away
    window.addEventListener('beforeunload', function() {
        if (pollInterval) {
            clearInterval(pollInterval);
        }
    });
    
    // Start the process
    startUmapGeneration();
});