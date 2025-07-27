from quart import Quart, render_template, request, redirect, url_for, flash, jsonify
import chromadb
import pandas as pd
import os
import h5py
import json
from collections import Counter
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
    collection_names = [col.name for col in collections]
    return await render_template("index.html", collections=collection_names,
                                 current_collection=collection.name)


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


@app.route('/upload-data', methods=['POST'])
async def upload_data():
    files = await request.files
    if 'data_file' not in files:
        await flash("No file uploaded.", "danger")
        return redirect(url_for('index'))

    file = files['data_file']

    if file.filename == '':
        await flash("No selected file.", "danger")
        return redirect(url_for('index'))

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
            return redirect(url_for('index'))

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

    return redirect(url_for('index'))


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
        
        # Validate input
        if not embeddings or len(embeddings) == 0:
            return {"success": False, "error": "No embeddings provided"}
        
        if len(embeddings) < 2:
            return {"success": False, "error": "Need at least 2 vectors for UMAP projection"}
        
        # Extract cancer types
        cancer_types = []
        for meta in metadatas:
            y = meta.get("y", "Unknown") if meta else "Unknown"
            if isinstance(y, bytes):
                y = y.decode()
            cancer_types.append(str(y))

        unique_labels = sorted(set(cancer_types))
        
        # Limit colors to available ones
        color_list = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
        label_colors = {label: color_list[i % len(color_list)] for i, label in enumerate(unique_labels)}
        point_colors = [label_colors[label] for label in cancer_types]

        # UMAP projection with optimized parameters
        reducer = umap.UMAP(
            n_components=2, 
            random_state=42,
            n_neighbors=min(15, len(embeddings) - 1),  # Adjust for small datasets
            min_dist=0.1,
            metric='cosine'
        )
        reduced = reducer.fit_transform(embeddings)

        # Create figure with explicit figsize and DPI
        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=point_colors, alpha=0.7, s=30)
        ax.set_title("UMAP Projection of Embeddings", fontsize=14, fontweight='bold')
        ax.set_xlabel("UMAP 1", fontsize=12)
        ax.set_ylabel("UMAP 2", fontsize=12)
        ax.grid(True, alpha=0.3)

        # Legend - limit to reasonable number of labels
        if len(unique_labels) <= 20:  # Only show legend if not too many categories
            handles = [
                plt.Line2D([0], [0], marker='o', color='w',
                           label=label, markersize=7,
                           markerfacecolor=label_colors[label])
                for label in unique_labels
            ]
            ax.legend(handles=handles, title="Categories", bbox_to_anchor=(1.05, 1), loc='upper left')

        # Save plot to base64
        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format="png", dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        
        # Explicitly close figure and clear memory
        plt.close(fig)
        plt.clf()

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
    metadata = form.get('collection_metadata', '').strip()

    if not name:
        await flash("Collection name is required.", "danger")
        return redirect(url_for('create_collection_page'))

    try:
        meta = json.loads(metadata) if metadata else None
        chroma_client.create_collection(name=name, metadata=meta)
        await flash(f"Collection '{name}' created successfully.", "success")
    except Exception as e:
        await flash(f"Error creating collection: {e}", "danger")

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
