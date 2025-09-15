from app.rag import VectorStore

if __name__ == "__main__":
    VS = VectorStore()
    VS.build(persist=True)
    print("Built FAISS index and saved metadata to ./data")
