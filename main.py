from quart import Quart, render_template, request, redirect, url_for, flash, jsonify
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


@app.route('/', methods=['GET'])
async def index():
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
    try:
        data = await request.get_json()
        query_vector_str = data.get('query_vector', '')
        
        if not query_vector_str:
            return jsonify({
                'success': False, 
                'error': 'Query vector is required'
            }), 400
        
        try:
            query_vector = list(map(float, query_vector_str.split(',')))
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Invalid vector format. Please enter comma-separated numbers (e.g., 0.1, 0.2, 0.3)'
            }), 400

        if len(query_vector) == 0:
            return jsonify({
                'success': False,
                'error': 'Query vector cannot be empty'
            }), 400

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=10,
            include=['documents', 'metadatas', 'distances']
        )

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
        return jsonify({
            'success': False,
            'error': f'Search failed: {str(e)}'
        }), 500


@app.route('/api/vectors-3d', methods=['GET'])
async def api_vectors_3d():
    """API endpoint for 3D visualization data (optimized for performance)"""
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
        
        return jsonify({
            'success': True,
            'vectors': vectors_3d,
            'total_available': len(results['ids']),
            'returned': len(vectors_3d)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to load 3D data: {str(e)}'
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

    try:
        content = file.read()  # This is sync, valid in Quart
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            ids = df['submitter_ids'].astype(str).tolist()
            embeddings = df['X'].apply(lambda v: list(map(float, v.split(',')))).tolist()
            metadatas = [{'y': y.decode('utf-8') if isinstance(y, bytes) else y} for y in y_values.tolist()]

        elif filename.endswith('.h5') or filename.endswith('.hdf5'):
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

        await flash(f"Uploaded and stored {len(ids)} vectors.", "success")

    except Exception as e:
        print("Upload error:", e)
        await flash(f"Error processing file: {str(e)}", "danger")

    return redirect(url_for('browse_vectors'))


@app.route('/browse-vectors')
async def browse_vectors():
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
    return await render_template(
        'create.html',
        current_collection=collection,  # Safe fallback
        collections=get_all_collections()
    )


if __name__ == "__main__":
    try:
        app.run(debug=True)
    finally:
        executor.shutdown(wait=True)
