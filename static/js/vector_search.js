/**
 * Vector Similarity Search
 * Handles vector similarity search functionality for the browse page
 */

document.addEventListener('DOMContentLoaded', function() {
    const searchForm = document.getElementById('searchForm');
    const searchButton = document.getElementById('searchButton');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const searchError = document.getElementById('searchError');
    const searchSuccess = document.getElementById('searchSuccess');
    const errorMessage = document.getElementById('errorMessage');
    const successMessage = document.getElementById('successMessage');
    const searchResultsContainer = document.getElementById('searchResultsContainer');
    const resultsTableBody = document.getElementById('resultsTableBody');
    const queryVector = document.getElementById('query_vector');

    function showError(message) {
        if (searchError && errorMessage) {
            searchError.style.display = 'block';
            if (searchSuccess) {
                searchSuccess.style.display = 'none';
            }
            if (loadingIndicator) {
                loadingIndicator.style.display = 'none';
            }
            errorMessage.textContent = message;
        }
    }

    function showSuccess(message) {
        if (searchSuccess && successMessage) {
            searchSuccess.style.display = 'block';
            if (searchError) {
                searchError.style.display = 'none';
            }
            if (loadingIndicator) {
                loadingIndicator.style.display = 'none';
            }
            successMessage.textContent = message;
        }
    }

    function hideMessages() {
        if (searchError) {
            searchError.style.display = 'none';
        }
        if (searchSuccess) {
            searchSuccess.style.display = 'none';
        }
        if (loadingIndicator) {
            loadingIndicator.style.display = 'none';
        }
    }

    function showLoading() {
        if (loadingIndicator && searchButton) {
            loadingIndicator.style.display = 'block';
            searchButton.disabled = true;
            searchButton.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Searching...';
            hideMessages();
        }
    }

    function hideLoading() {
        if (loadingIndicator && searchButton) {
            loadingIndicator.style.display = 'none';
            searchButton.disabled = false;
            searchButton.innerHTML = '<i class="fas fa-search me-1"></i>Search Similar';
        }
    }

    function renderResults(results) {
        if (!resultsTableBody) return;
        
        resultsTableBody.innerHTML = '';
        
        if (results.length === 0) {
            resultsTableBody.innerHTML = `
                <tr>
                    <td colspan="4" class="text-center py-4">
                        <i class="fas fa-search text-muted me-2"></i>
                        No similar vectors found
                    </td>
                </tr>
            `;
            return;
        }

        results.forEach(result => {
            const similarity = ((1 - result.distance) * 100).toFixed(1);
            const row = document.createElement('tr');
            
            row.innerHTML = `
                <td class="px-4 py-3">
                    <span class="badge bg-primary">${result.id}</span>
                </td>
                <td class="py-3">
                    <div class="text-truncate" style="max-width: 300px;" title="${result.document || ''}">
                        ${result.document || '<span class="text-muted">No document</span>'}
                    </div>
                </td>
                <td class="py-3">
                    ${result.metadata ? 
                        `<div class="glass-effect p-2 rounded">
                            <pre class="mb-0 small text-muted">${JSON.stringify(result.metadata, null, 2)}</pre>
                        </div>` : 
                        '<span class="text-muted">—</span>'
                    }
                </td>
                <td class="py-3">
                    <div class="d-flex align-items-center">
                        <div class="progress me-2" style="width: 100px; height: 8px;">
                            <div class="progress-bar bg-success" style="width: ${similarity}%"></div>
                        </div>
                        <span class="fw-semibold text-success">${result.distance.toFixed(4)}</span>
                    </div>
                </td>
            `;
            
            resultsTableBody.appendChild(row);
        });
    }

    if (searchForm) {
        searchForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const queryValue = queryVector.value.trim();
            
            if (!queryValue) {
                showError('Please enter a query vector');
                return;
            }

            showLoading();
            if (searchResultsContainer) {
                searchResultsContainer.style.display = 'none';
            }

            try {
                const response = await fetch('/api/search-vector', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query_vector: queryValue
                    })
                });

                const data = await response.json();

                if (data.success) {
                    showSuccess(`Found ${data.results.length} similar vectors`);
                    renderResults(data.results);
                    if (searchResultsContainer) {
                        searchResultsContainer.style.display = 'block';
                        
                        // Smooth scroll to results
                        setTimeout(() => {
                            searchResultsContainer.scrollIntoView({ 
                                behavior: 'smooth',
                                block: 'start'
                            });
                        }, 100);
                    }
                } else {
                    showError(data.error);
                    if (searchResultsContainer) {
                        searchResultsContainer.style.display = 'none';
                    }
                }
            } catch (error) {
                showError('Network error: Please check your connection and try again');
                if (searchResultsContainer) {
                    searchResultsContainer.style.display = 'none';
                }
            } finally {
                hideLoading();
            }
        });

        // Clear messages when user starts typing
        if (queryVector) {
            queryVector.addEventListener('input', function() {
                hideMessages();
            });
        }
    }
});
