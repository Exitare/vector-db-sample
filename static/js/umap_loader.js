document.addEventListener('DOMContentLoaded', function () {
    fetch('/generate-umap')
        .then(response => response.json())
        .then(data => {
            const placeholder = document.getElementById('umap-placeholder');
            if (data.image) {
                const img = document.createElement('img');
                img.src = 'data:image/png;base64,' + data.image;
                img.className = 'img-fluid border shadow-sm';
                placeholder.replaceWith(img);
            } else if (data.error) {
                placeholder.innerText = 'Error loading UMAP: ' + data.error;
            }
        })
        .catch(err => {
            document.getElementById('umap-placeholder').innerText = 'UMAP load failed.';
            console.error('UMAP fetch error:', err);
        });
});