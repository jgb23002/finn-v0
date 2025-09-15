from app.rag import VectorStore

def test_rag_search():
    vs = VectorStore()
    # Try loading prebuilt, else build on the fly for test
    loaded = vs.load()
    if not loaded:
        vs.build(persist=False)
    results = vs.search("sleep routine tips", k=3)
    assert len(results) >= 1
    # each result is (path, chunk)
    path, chunk = results[0]
    assert isinstance(path, str) and isinstance(chunk, str)
