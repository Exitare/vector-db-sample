/**
 * File Upload Handler
 * Handles form validation and file upload functionality for the browse page
 */

(function() {
    'use strict';
    
    document.addEventListener('DOMContentLoaded', function() {
        // Bootstrap form validation
        const forms = document.querySelectorAll('.needs-validation');
        Array.from(forms).forEach(form => {
            form.addEventListener('submit', event => {
                if (!form.checkValidity()) {
                    event.preventDefault();
                    event.stopPropagation();
                } else {
                    // Show upload progress
                    const submitBtn = form.querySelector('button[type="submit"]');
                    const originalText = submitBtn.innerHTML;
                    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Uploading...';
                    submitBtn.disabled = true;
                    
                    // Re-enable button after some time in case of error
                    setTimeout(() => {
                        submitBtn.innerHTML = originalText;
                        submitBtn.disabled = false;
                    }, 30000);
                }
                form.classList.add('was-validated');
            });
        });
        
        // File input change handler
        const fileInput = document.getElementById('data_file');
        if (fileInput) {
            fileInput.addEventListener('change', function() {
                const file = this.files[0];
                if (file) {
                    const fileSize = (file.size / 1024 / 1024).toFixed(2);
                    const fileInfo = document.querySelector('.form-text');
                    if (fileInfo) {
                        fileInfo.innerHTML = `
                            <i class="fas fa-file text-success me-1"></i>
                            Selected: <strong>${file.name}</strong> (${fileSize} MB)
                        `;
                    }
                }
            });
        }
    });
})();
