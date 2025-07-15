from quart import Quart, render_template, request, redirect, url_for, flash
import chromadb
from chromadb.config import Settings

app = Quart(__name__)
app.secret_key = "your-secret-key"

# Initialize Chroma DB
chroma_client = chromadb.PersistentClient(path=".chromadb")
collection = chroma_client.get_or_create_collection("my_collection")


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


if __name__ == "__main__":
    app.run(debug=True)
