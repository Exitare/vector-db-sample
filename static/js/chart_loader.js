document.addEventListener('DOMContentLoaded', function () {
    const ctx = document.getElementById('cancerChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: cancerCounts.labels,
            datasets: [{
                label: 'Cancer Type Count',
                data: cancerCounts.data,
                backgroundColor: 'rgba(54, 162, 235, 0.6)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: { display: true, text: 'Count' }
                },
                x: {
                    title: { display: true, text: 'Cancer Type' }
                }
            }
        }
    });
});