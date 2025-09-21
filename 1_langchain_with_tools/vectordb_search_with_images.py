import os
from typing import List
from sentence_transformers import SentenceTransformer
import chromadb

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

DB_PATH = "img_db"
IMG_DIR = "images"

def load_images(img_dir: str) -> List[str]:
    os.makedirs(img_dir, exist_ok=True)
    paths = []
    for name in os.listdir(img_dir):
        if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            paths.append(os.path.join(img_dir, name))
    return sorted(paths)

def encode_images(model: SentenceTransformer, image_paths: List[str]) -> List[List[float]]:
    # CLIP model from sentence-transformers can encode images directly
    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    embs = model.encode(imgs, convert_to_numpy=True, normalize_embeddings=True).tolist()
    return embs

def encode_text(model: SentenceTransformer, text: str) -> List[float]:
    return model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()

def build_index(image_paths: List[str], model_name: str = "clip-ViT-B-32"):
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Encoding {len(image_paths)} images...")
    img_embs = encode_images(model, image_paths)

    print(f"Connecting to Chroma at {DB_PATH} ...")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name="images",
        metadata={"hnsw:space": "cosine"}  # cosine similarity
    )

    # Upsert images
    ids = [f"img_{i}" for i in range(len(image_paths))]
    metadatas = [{"path": p, "filename": os.path.basename(p)} for p in image_paths]

    # Clean old entries with same ids (optional)
    try:
        collection.delete(ids=ids)
    except Exception:
        pass

    print("Adding embeddings to Chroma...")
    collection.add(ids=ids, embeddings=img_embs, metadatas=metadatas)
    print("Done indexing.")
    return model, collection

from PIL import Image

from PIL import Image

def search_by_text(collection, model, query: str, k: int = 5):
    q_emb = encode_text(model, query)
    res = collection.query(query_embeddings=[q_emb], n_results=k)

    print(f"\n🔎 Text Query: {query}\n")
    best = res.get("metadatas", [[]])[0][0]  # take top-1
    print(f"Best Match: {best.get('filename')} — {best.get('path')}")
    Image.open(best["path"]).show()  # open only the best match

def search_by_image(collection, model: SentenceTransformer, image_path: str, k: int = 5):
    q_emb = encode_images(model, [image_path])[0]
    res = collection.query(query_embeddings=[q_emb], n_results=k)

    print(f"\n🖼️ Image Query: {image_path}\n")
    best = res.get("metadatas", [[]])[0][0]  # take top-1
    print(f"Best Match: {best.get('filename')} — {best.get('path')}")
    Image.open(best["path"]).show()  # open only the best match

def main():
    # 1) Collect images
    image_paths = load_images(IMG_DIR)
    if not image_paths:
        print(f"No images found in '{IMG_DIR}'. Please add some JPG/PNG files and rerun.")
        return

    # 2) Build / load index
    model, collection = build_index(image_paths)

    # 3) Example searches
    search_by_text(collection, model, "I need goo running shoes red in color", k=3)

    # 4) Optional: query-by-image (pick the first image as a demo)
    search_by_image(collection, model, image_paths[5], k=3)

if __name__ == "__main__":
    main()
