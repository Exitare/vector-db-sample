# Grafana Dashboard Setup Instructions

## Default Access
- **URL**: http://localhost:3000
- **Username**: admin
- **Password**: admin123

## Adding Prometheus Data Source
1. Go to Configuration > Data Sources
2. Click "Add data source"
3. Select "Prometheus"
4. Set URL to: `http://prometheus:9090`
5. Click "Save & Test"

## Importing Dashboards
You can import pre-built dashboards for common metrics:

### System Metrics Dashboard
- Dashboard ID: 1860 (Node Exporter Full)
- Dashboard ID: 893 (Docker Prometheus Monitoring)

### Custom Metrics
Create custom panels for your Vector DB application:
- API response times
- Vector search performance
- Memory usage
- Request rates
- Error rates

## Useful Queries
Here are some PromQL queries you can use:

### Application Metrics
```promql
# Request rate
rate(flask_requests_total[5m])

# Response time
histogram_quantile(0.95, rate(flask_request_duration_seconds_bucket[5m]))

# Error rate
rate(flask_requests_total{status=~"5.."}[5m]) / rate(flask_requests_total[5m])
```

### System Metrics
```promql
# CPU Usage
100 - (avg(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# Memory Usage
(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100

# Disk Usage
100 - ((node_filesystem_avail_bytes * 100) / node_filesystem_size_bytes)
```
