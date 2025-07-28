from quart import Quart, render_template, request, redirect, url_for, flash, jsonify, Response
import chromadb
import pandas as pd
import numpy as np
import os
import h5py
import json
from collections import Counter
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import io
import base64
import umap
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import time
import argparse
# Prometheus metrics
from prometheus_client import Counter as PrometheusCounter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Define metrics with proper parameters
REQUEST_COUNT = PrometheusCounter(
    'http_requests_total', 
    'Total HTTP requests', 
    ['method', 'endpoint', 'status']
)
REQUEST_DURATION = Histogram(
    'http_request_duration_seconds', 
    'HTTP request duration in seconds', 
    ['method', 'endpoint']
)
VECTOR_COUNT = Gauge(
    'vector_database_vectors_total', 
    'Total number of vectors in database'
)
SEARCH_DURATION = Histogram(
    'vector_search_duration_seconds', 
    'Vector search duration in seconds'
)
UMAP_GENERATION_DURATION = Histogram(
    'umap_generation_duration_seconds', 
    'UMAP generation duration in seconds'
)
UPLOAD_COUNT = PrometheusCounter(
    'vector_uploads_total', 
    'Total number of vector uploads', 
    ['file_type']
)
COLLECTION_COUNT = Gauge(
    'vector_database_collections_total', 
    'Total number of collections'
)
API_ERRORS = PrometheusCounter(
    'api_errors_total', 
    'Total API errors', 
    ['endpoint', 'error_type']
)

# Page view tracking metric
PAGE_VIEWS = PrometheusCounter(
    'page_views_total',
    'Total number of page views',
    ['page', 'user_agent']
)

app = Quart(__name__)
app.secret_key = "your-secret-key"
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB, adjust as needed

# Thread pool for background tasks
executor = ThreadPoolExecutor(max_workers=2)

# Storage for UMAP generation tasks
umap_cache = {}

# Initialize Chroma DB
chroma_client = chromadb.PersistentClient(path=".chromadb")
collection = chroma_client.get_or_create_collection("heterogeneous_vectors")

# Middleware to track metrics
@app.before_request
async def before_request():
    request.start_time = time.time()

@app.after_request
async def after_request(response):
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        endpoint = request.endpoint or 'unknown'
        method = request.method
        status = str(response.status_code)
        
        # Update metrics
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
    
    return response

# Update collection metrics
def update_collection_metrics():
    try:
        # Update vector count
        results = collection.get()
        VECTOR_COUNT.set(len(results['ids']) if results['ids'] else 0)
        
        # Update collection count
        collections = chroma_client.list_collections()
        COLLECTION_COUNT.set(len(collections))
    except Exception as e:
        print(f"Error updating metrics: {e}")

# Metrics endpoint
@app.route('/metrics')
async def metrics():
    """Prometheus metrics endpoint"""
    try:
        update_collection_metrics()
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
    except Exception as e:
        API_ERRORS.labels(endpoint='metrics', error_type=type(e).__name__).inc()
        return Response(f"Error generating metrics: {str(e)}", status=500)


@app.route('/', methods=['GET'])
async def index():
    # Track page view
    user_agent = request.headers.get('User-Agent', 'Unknown')[:50]  # Truncate to avoid high cardinality
    PAGE_VIEWS.labels(page='index', user_agent=user_agent).inc()
    
    collections = chroma_client.list_collections()
    
    # Create enhanced collection info with metadata
    collections_info = []
    for col in collections:
        collections_info.append({
            'name': col.name,
            'metadata': getattr(col, 'metadata', None)
        })
    
    # Get current collection metadata
    current_collection_metadata = getattr(collection, 'metadata', None)
    
    return await render_template("index.html", 
                                 collections=[col.name for col in collections],  # Keep backward compatibility
                                 collections_info=collections_info,
                                 current_collection=collection.name,
                                 current_collection_metadata=current_collection_metadata)


@app.route('/health', methods=['GET'])
async def health_check():
    """Health check endpoint for Docker and monitoring systems"""
    try:
        # Check if ChromaDB is accessible
        collections = chroma_client.list_collections()
        
        return jsonify({
            'status': 'healthy',
            'service': 'vector-db-app',
            'timestamp': datetime.now().isoformat(),
            'collections_count': len(collections),
            'current_collection': collection.name if collection else None
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'vector-db-app',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }), 503


@app.route('/set-collection', methods=['POST'])
async def set_collection():
    form = await request.form
    selected_name = form.get('collection_name')
    global collection  # so it updates the top-level variable
    collection = chroma_client.get_or_create_collection(name=selected_name)
    await flash(f"Switched to collection: {selected_name}", "success")
    return redirect(url_for('index'))


@app.route('/add-vector', methods=['POST'])
async def add_vector():
    form = await request.form
    vector_id = form['id']
    vector = list(map(float, form['vector'].split(',')))
    metadata = {"label": form.get('label', 'none')}
    document = form.get('document', '')

    collection.add(
        ids=[vector_id],
        embeddings=[vector],
        metadatas=[metadata],
        documents=[document]
    )

    await flash("Vector added successfully!", "success")
    return redirect(url_for('index'))


@app.route('/search-vector', methods=['POST'])
async def search_vector():
    form = await request.form
    try:
        query_vector = list(map(float, form['query_vector'].split(',')))
    except ValueError:
        await flash("Invalid vector format. Please enter comma-separated numbers.", "danger")
        return redirect(url_for('index'))

    try:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=10,
            include=['documents', 'metadatas', 'distances']
        )

        print(results)
    except Exception as e:
        await flash(f"Search failed: {str(e)}", "danger")
        return redirect(url_for('index'))

    hits = [
        {
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]

    return await render_template("index.html", results=hits)


@app.route('/api/search-vector', methods=['POST'])
async def api_search_vector():
    """API endpoint for AJAX vector search"""
    start_time = time.time()
    try:
        data = await request.get_json()
        query_vector_str = data.get('query_vector', '')
        
        if not query_vector_str:
            API_ERRORS.labels(endpoint='search-vector', error_type='ValidationError').inc()
            return jsonify({
                'success': False, 
                'error': 'Query vector is required'
            }), 400
        
        try:
            query_vector = list(map(float, query_vector_str.split(',')))
        except ValueError:
            API_ERRORS.labels(endpoint='search-vector', error_type='ValidationError').inc()
            return jsonify({
                'success': False,
                'error': 'Invalid vector format. Please enter comma-separated numbers (e.g., 0.1, 0.2, 0.3)'
            }), 400

        if len(query_vector) == 0:
            API_ERRORS.labels(endpoint='search-vector', error_type='ValidationError').inc()
            return jsonify({
                'success': False,
                'error': 'Query vector cannot be empty'
            }), 400

        # Perform search with timing
        search_start = time.time()
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=10,
            include=['documents', 'metadatas', 'distances']
        )
        search_duration = time.time() - search_start
        SEARCH_DURATION.observe(search_duration)

        hits = [
            {
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
            for i in range(len(results["ids"][0]))
        ]

        return jsonify({
            'success': True,
            'results': hits
        })

    except Exception as e:
        API_ERRORS.labels(endpoint='search-vector', error_type=type(e).__name__).inc()
        return jsonify({
            'success': False,
            'error': f'Search failed: {str(e)}'
        }), 500


@app.route('/api/vectors-3d', methods=['GET'])
async def api_vectors_3d():
    """API endpoint for 3D visualization data (optimized for performance)"""
    start_time = time.time()
    
    try:
        # Get vectors with pagination limit for performance
        limit = min(int(request.args.get('limit', 1000)), 5000)  # Max 5000 points
        
        # Get all vectors with embeddings and metadata
        results = collection.get(include=["embeddings", "metadatas", "documents"])
        
        if not results['ids']:
            return jsonify({
                'success': True,
                'vectors': []
            })
        
        vectors_3d = []
        # Limit the number of data points to prevent browser overload
        total_points = min(len(results['ids']), limit)
        
        for i in range(total_points):
            # Safely get metadata
            metadata = None
            if results['metadatas'] is not None and i < len(results['metadatas']):
                metadata = results['metadatas'][i]
            
            cancer_type = 'Unknown'
            if metadata and isinstance(metadata, dict):
                cancer_type = metadata.get('y', 'Unknown')
            
            # Safely get embedding - only take first 3 dimensions for 3D visualization
            embedding_3d = [0.0, 0.0, 0.0]  # Default values
            if results['embeddings'] is not None and i < len(results['embeddings']):
                full_embedding = results['embeddings'][i]
                # Extract only first 3 dimensions
                for j in range(min(3, len(full_embedding))):
                    embedding_3d[j] = float(full_embedding[j])
            
            vectors_3d.append({
                'id': results['ids'][i],
                'embedding': embedding_3d,  # Only 3D coordinates
                'cancer_type': cancer_type,
                'document': results['documents'][i] if results['documents'] and i < len(results['documents']) else None
            })
        
        # Record successful 3D data retrieval
        duration = time.time() - start_time
        SEARCH_DURATION.observe(duration)  # Reuse search duration metric
        
        return jsonify({
            'success': True,
            'vectors': vectors_3d,
            'total_available': len(results['ids']),
            'returned': len(vectors_3d)
        })
        
    except Exception as e:
        # Record 3D visualization error
        duration = time.time() - start_time
        SEARCH_DURATION.observe(duration)
        API_ERRORS.labels(endpoint='vectors-3d', error_type=type(e).__name__).inc()
        
        return jsonify({
            'success': False,
            'error': f'Failed to load 3D data: {str(e)}'
        }), 500


@app.route('/api/nearest-neighbors/<vector_id>')
async def api_nearest_neighbors(vector_id):
    """API endpoint to find nearest neighbors for a specific vector"""
    start_time = time.time()
    
    try:
        # Get the specific vector by ID
        results = collection.get(
            ids=[vector_id],
            include=["embeddings", "metadatas", "documents"]
        )
        
        if not results['ids'] or len(results['ids']) == 0:
            API_ERRORS.labels(endpoint='nearest-neighbors', error_type='NotFound').inc()
            return jsonify({
                'success': False,
                'error': f'Vector with ID {vector_id} not found'
            }), 404
        
        # Get the embedding for the query vector
        query_embedding = results['embeddings'][0]
        
        # Query for similar vectors
        search_start = time.time()
        similar_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10,  # Get top 10 similar vectors
            include=['documents', 'metadatas', 'distances']
        )
        search_duration = time.time() - search_start
        SEARCH_DURATION.observe(search_duration)
        
        # Format the results
        hits = []
        for i in range(len(similar_results["ids"][0])):
            hits.append({
                "id": similar_results["ids"][0][i],
                "document": similar_results["documents"][0][i] if similar_results["documents"] and similar_results["documents"][0] else None,
                "metadata": similar_results["metadatas"][0][i] if similar_results["metadatas"] and similar_results["metadatas"][0] else None,
                "distance": similar_results["distances"][0][i],
            })
        
        # Record successful nearest neighbors search
        duration = time.time() - start_time
        SEARCH_DURATION.observe(duration)
        
        return jsonify({
            'success': True,
            'query_vector_id': vector_id,
            'results': hits
        })
        
    except Exception as e:
        # Record nearest neighbors error
        duration = time.time() - start_time
        SEARCH_DURATION.observe(duration)
        API_ERRORS.labels(endpoint='nearest-neighbors', error_type=type(e).__name__).inc()
        
        return jsonify({
            'success': False,
            'error': f'Failed to find nearest neighbors: {str(e)}'
        }), 500


@app.route('/upload-data', methods=['POST'])
async def upload_data():
    files = await request.files
    if 'data_file' not in files:
        await flash("No file uploaded.", "danger")
        return redirect(url_for('browse_vectors'))

    file = files['data_file']

    if file.filename == '':
        await flash("No selected file.", "danger")
        return redirect(url_for('browse_vectors'))

    filename = file.filename.lower()
    file_type = 'unknown'

    try:
        content = file.read()  # This is sync, valid in Quart
        if filename.endswith('.csv'):
            file_type = 'csv'
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            ids = df['submitter_ids'].astype(str).tolist()
            embeddings = df['X'].apply(lambda v: list(map(float, v.split(',')))).tolist()
            y_values = df['y']
            metadatas = [{'y': y.decode('utf-8') if isinstance(y, bytes) else y} for y in y_values.tolist()]

        elif filename.endswith('.h5') or filename.endswith('.hdf5'):
            file_type = 'hdf5'
            # Save and load with h5py
            filepath = f"/tmp/{file.filename}"
            with open(filepath, 'wb') as f:
                f.write(content)

            with h5py.File(filepath, 'r') as f:
                # Assuming datasets are stored as arrays: /submitter_ids, /X, /y
                ids = [str(x.decode('utf-8')) for x in f['submitter_ids'][()]]
                embeddings = f['X'][()].tolist()
                y_values = f['y'][()]
                if hasattr(y_values, "tolist"):  # numpy array
                    y_values = y_values.tolist()
                metadatas = [{'y': y.decode('utf-8') if isinstance(y, bytes) else y} for y in y_values]

            os.remove(filepath)

        else:
            await flash("Unsupported file type. Please upload a .csv or .h5 file.", "danger")
            return redirect(url_for('browse_vectors'))

        # Batch upload to Chroma
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            collection.add(
                ids=ids[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size]
            )

        # Update upload metrics
        UPLOAD_COUNT.labels(file_type=file_type).inc()
        
        await flash(f"Uploaded and stored {len(ids)} vectors.", "success")

    except Exception as e:
        print("Upload error:", e)
        API_ERRORS.labels(endpoint='upload-data', error_type=type(e).__name__).inc()
        await flash(f"Error processing file: {str(e)}", "danger")

    return redirect(url_for('browse_vectors'))


@app.route('/browse-vectors')
async def browse_vectors():
    # Track page view
    user_agent = request.headers.get('User-Agent', 'Unknown')[:50]  # Truncate to avoid high cardinality
    PAGE_VIEWS.labels(page='browse-vectors', user_agent=user_agent).inc()
    
    try:
        results = collection.get(include=["embeddings", "metadatas", "documents"])

        vectors = []
        cancer_types = []

        for i in range(len(results['ids'])):
            metadata = results['metadatas'][i] if results['metadatas'] else None
            cancer_type = metadata.get("y") if metadata else "Unknown"

            vectors.append({
                'id': results['ids'][i],
                'embedding': results['embeddings'][i],
                'metadata': metadata,
                'document': results['documents'][i] if results['documents'] else None
            })

            cancer_types.append(cancer_type)

        cancer_counts = dict(Counter(cancer_types))

        return await render_template(
            'browse.html',
            vectors=vectors,
            cancer_counts=cancer_counts
        )

    except Exception as e:
        await flash(f"Error loading vectors: {str(e)}", "danger")
        return redirect(url_for('index'))


@app.route('/vectors-3d')
async def vectors_3d():
    """Dedicated 3D visualization page"""
    # Track page view
    user_agent = request.headers.get('User-Agent', 'Unknown')[:50]  # Truncate to avoid high cardinality
    PAGE_VIEWS.labels(page='vectors-3d', user_agent=user_agent).inc()
    
    try:
        results = collection.get(include=["embeddings", "metadatas", "documents"])

        vectors = []
        cancer_types = []

        for i in range(len(results['ids'])):
            metadata = results['metadatas'][i] if results['metadatas'] else None
            cancer_type = metadata.get("y") if metadata else "Unknown"

            vectors.append({
                'id': results['ids'][i],
                'embedding': results['embeddings'][i],
                'metadata': metadata,
                'document': results['documents'][i] if results['documents'] else None
            })

            cancer_types.append(cancer_type)

        cancer_counts = dict(Counter(cancer_types))

        return await render_template(
            'vectors_3d.html',
            vectors=vectors,
            cancer_counts=cancer_counts
        )

    except Exception as e:
        await flash(f"Error loading 3D vectors: {str(e)}", "danger")
        return redirect(url_for('browse_vectors'))


def generate_umap_sync(embeddings, metadatas):
    """Synchronous UMAP generation function to run in thread pool"""
    try:
        # Set matplotlib to use non-interactive backend for thread safety
        import matplotlib
        matplotlib.use('Agg')  # Use non-GUI backend
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
        
        # Validate input safely (avoid boolean evaluation on arrays)
        if embeddings is None or len(embeddings) == 0:
            return {"success": False, "error": "No embeddings provided"}
        
        if len(embeddings) < 2:
            return {"success": False, "error": "Need at least 2 vectors for UMAP projection"}

        # Extract cancer types (using your exact working logic)
        cancer_types = []
        for meta in metadatas:
            y = meta.get("y", "Unknown") if meta else "Unknown"
            if isinstance(y, bytes):
                y = y.decode()
            cancer_types.append(str(y))

        unique_labels = sorted(set(cancer_types))
        color_list = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
        label_colors = {label: color_list[i % len(color_list)] for i, label in enumerate(unique_labels)}
        point_colors = [label_colors[label] for label in cancer_types]

        # UMAP projection (using your exact working parameters)
        reducer = umap.UMAP(n_components=2, random_state=42)
        reduced = reducer.fit_transform(embeddings)

        # Plot (using your exact working approach but with larger size)
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=point_colors, alpha=0.7)
        ax.set_title("UMAP Projection of Embeddings", fontsize=14, fontweight='bold')
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

        # Legend (using your exact working legend code)
        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       label=label, markersize=7,
                       markerfacecolor=label_colors[label])
            for label in unique_labels
        ]
        ax.legend(handles=handles, title="Cancer Types", bbox_to_anchor=(1.05, 1), loc='upper left')

        # Save plot to base64 (with higher DPI for better quality)
        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format="png", dpi=200, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        
        # Clean up
        plt.close(fig)

        return {"success": True, "image": img_base64}

    except Exception as e:
        # Make sure to clean up matplotlib resources even on error
        try:
            plt.close('all')
            plt.clf()
        except:
            pass
        return {"success": False, "error": f"UMAP generation failed: {str(e)}"}


@app.route('/generate-umap')
async def generate_umap():
    """Start UMAP generation asynchronously"""
    try:
        # Generate a unique task ID based on current collection
        task_id = f"umap_{collection.name}_{hash(str(collection._client))}"
        
        # Check if we already have a cached result
        if task_id in umap_cache and umap_cache[task_id].get('status') == 'completed':
            return jsonify(umap_cache[task_id]['result'])
        
        # Check if task is already running
        if task_id in umap_cache and umap_cache[task_id].get('status') == 'running':
            return jsonify({"success": False, "status": "running", "message": "UMAP generation in progress"})

        # Get data for UMAP generation
        results = collection.get(include=["embeddings", "metadatas"])
        embeddings = results["embeddings"]
        metadatas = results["metadatas"]

        if len(embeddings) == 0:
            return jsonify({"success": False, "error": "No vectors found in collection"})

        # Mark task as running
        umap_cache[task_id] = {"status": "running", "start_time": asyncio.get_event_loop().time()}

        # Start background task with timeout
        loop = asyncio.get_event_loop()
        
        # Wrap the sync function with timeout handling
        async def run_with_timeout():
            try:
                # Run with a 5-minute timeout
                future = loop.run_in_executor(executor, generate_umap_sync, embeddings, metadatas)
                result = await asyncio.wait_for(future, timeout=300.0)  # 5 minutes
                return result
            except asyncio.TimeoutError:
                return {"success": False, "error": "UMAP generation timed out after 5 minutes"}
            except Exception as e:
                return {"success": False, "error": f"UMAP generation failed: {str(e)}"}

        # Create task but don't await it
        task = asyncio.create_task(run_with_timeout())
        
        def on_complete(task_future):
            try:
                result = task_future.result()
                umap_cache[task_id] = {"status": "completed", "result": result}
            except Exception as e:
                umap_cache[task_id] = {
                    "status": "completed", 
                    "result": {"success": False, "error": str(e)}
                }

        task.add_done_callback(on_complete)

        return jsonify({"success": False, "status": "started", "message": "UMAP generation started"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/umap-status/<task_id>')
async def umap_status(task_id):
    """Check the status of UMAP generation"""
    if task_id not in umap_cache:
        return jsonify({"success": False, "error": "Task not found"})
    
    task_data = umap_cache[task_id]
    
    if task_data["status"] == "running":
        return jsonify({"success": False, "status": "running", "message": "UMAP generation in progress"})
    elif task_data["status"] == "completed":
        return jsonify(task_data["result"])
    else:
        return jsonify({"success": False, "error": "Unknown task status"})


@app.route('/create-collection', methods=['POST'])
async def create_collection():
    form = await request.form
    name = form.get('new_collection_name', '').strip()
    
    # Get standard metadata fields
    source = form.get('source', '').strip()
    description = form.get('description', '').strip()
    version = form.get('version', '').strip()
    created_by = form.get('created_by', '').strip()
    additional_metadata = form.get('additional_metadata', '').strip()

    if not name:
        await flash("Collection name is required.", "danger")
        return redirect(url_for('create_collection_page'))

    try:
        # Start with standard metadata fields
        metadata = {}
        
        # Add standard fields if provided
        if source:
            metadata['source'] = source
        if description:
            metadata['description'] = description
        if version:
            metadata['version'] = version
        if created_by:
            metadata['created_by'] = created_by
        
        # Add creation timestamp
        from datetime import datetime
        metadata['created_at'] = datetime.now().isoformat()
        
        # Parse and merge additional JSON metadata if provided
        if additional_metadata:
            try:
                additional_json = json.loads(additional_metadata)
                if isinstance(additional_json, dict):
                    # Merge additional metadata, with additional_json taking precedence for duplicates
                    metadata.update(additional_json)
                else:
                    await flash("Additional metadata must be a valid JSON object.", "danger")
                    return redirect(url_for('create_collection_page'))
            except json.JSONDecodeError as e:
                await flash(f"Invalid JSON format in additional metadata: {str(e)}", "danger")
                return redirect(url_for('create_collection_page'))
        
        # Create collection with merged metadata (None if empty)
        final_metadata = metadata if metadata else None
        chroma_client.create_collection(name=name, metadata=final_metadata)
        
        # Create success message with metadata summary
        metadata_summary = []
        if source:
            metadata_summary.append(f"Source: {source}")
        if version:
            metadata_summary.append(f"Version: {version}")
        if created_by:
            metadata_summary.append(f"Created by: {created_by}")
        
        success_msg = f"Collection '{name}' created successfully."
        if metadata_summary:
            success_msg += f" ({', '.join(metadata_summary)})"
        
        await flash(success_msg, "success")
        
    except Exception as e:
        error_msg = str(e)
        if "already exists" in error_msg.lower():
            await flash(f"Collection '{name}' already exists. Please choose a different name.", "danger")
        else:
            await flash(f"Error creating collection: {error_msg}", "danger")

    return redirect(url_for('index'))


def get_all_collections():
    return [col.name for col in chroma_client.list_collections()]


@app.route('/create-collection', methods=['GET'])
async def create_collection_page():
    # Track page view
    user_agent = request.headers.get('User-Agent', 'Unknown')[:50]  # Truncate to avoid high cardinality
    PAGE_VIEWS.labels(page='create-collection', user_agent=user_agent).inc()
    
    return await render_template(
        'create.html',
        current_collection=collection,  # Safe fallback
        collections=get_all_collections()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Vector DB Application")
    parser.add_argument('--port',"-p", type=int, default=5050, help='Port to run the application on')
    
    args = parser.parse_args()
    port = args.port

    
    try:
        # Check if we're in production mode
        flask_env = os.getenv('FLASK_ENV', 'development')
        flask_debug = os.getenv('FLASK_DEBUG', '1') == '1'
        
        if flask_env == 'production' and not flask_debug:
            # Production mode - use Hypercorn
            from hypercorn.config import Config
            from hypercorn.asyncio import serve
            import asyncio
            
            config = Config()
            config.bind = ["0.0.0.0:5000"]  # Bind to all interfaces
            config.workers = 1  # Adjust based on your needs
            config.access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
            config.accesslog = "-"  # Log to stdout
            config.errorlog = "-"   # Log to stdout
            
            print(f"🚀 Starting Vector DB Application in PRODUCTION mode on {config.bind[0]}")
            print("📊 Monitoring endpoints:")
            print("   - Application: http://localhost:5000")
            print("   - Health check: http://localhost:5000/")
            
            # Run with Hypercorn
            asyncio.run(serve(app, config))
        else:
            # Development mode - use Quart's development server
            print(f"🔧 Starting Vector DB Application in DEVELOPMENT mode")
            print(f"   - Application: http://localhost:{port}")
            print("   - Debug mode: enabled")
            app.run(debug=True, host='0.0.0.0', port=port)
            
    finally:
        executor.shutdown(wait=True)
