from gensim.models import KeyedVectors
emb = "./deepwalk_embeddings.bin"
embs = KeyedVectors.load(emb,mmap='r')

print(embs)