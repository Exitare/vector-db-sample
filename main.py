from quart import Quart, render_template, request, redirect, url_for, flash
import chromadb
import pandas as pd
import io
import os
import h5py

app = Quart(__name__)
app.secret_key = "your-secret-key"
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB, adjust as needed

# Initialize Chroma DB
chroma_client = chromadb.PersistentClient(path=".chromadb")
collection = chroma_client.get_or_create_collection("heterogeneous_vectors")


@app.route('/', methods=['GET'])
async def index():
    return await render_template("index.html", results=None)


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
            n_results=5,
            include=['documents', 'metadatas', 'distances']
        )
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
                ids = [str(x) for x in f['submitter_ids'][()]]
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


if __name__ == "__main__":
    app.run(debug=True)
