import unittest

from vectorq.config import VectorQConfig
from vectorq.vectorq_core.cache.vector_db.embedding_metadata_storage import EmbeddingMetadataStorage, EmbeddingMetadataStorageType
from vectorq.vectorq_core.cache.vector_db.embedding_metadata_storage import EmbeddingMetadataStorageType
from vectorq.vectorq_core.cache.vector_db.embedding_metadata_storage.embedding_metadata_obj import EmbeddingMetadataObj

class TestVectorDBStrategy(unittest.TestCase):
    
    def test_hnsw_lib_strategy_returns_embedding(self):
        config = VectorQConfig(
            embedding_metadata_storage_type=EmbeddingMetadataStorageType.IN_MEMORY
        )
        
        embedding_metadata_storage: EmbeddingMetadataStorage = EmbeddingMetadataStorage(vectorq_config=config)
        initial_obj = EmbeddingMetadataObj(embedding_id=0, response="test")
        embedding_id = embedding_metadata_storage.add_metadata(embedding_id=0, metadata=initial_obj)
        assert embedding_id == 0
        assert embedding_metadata_storage.get_metadata(embedding_id=0) == initial_obj
        
        updated_obj = EmbeddingMetadataObj(embedding_id=0, response="test2")
        embedding_metadata_storage.update(embedding_id=0, metadata=updated_obj)
        assert embedding_metadata_storage.get_metadata(embedding_id=0) == updated_obj
        
        embedding_metadata_storage.flush()
        with self.assertRaises(ValueError):
            embedding_metadata_storage.get_metadata(embedding_id=0)
        
if __name__ == "__main__":
    unittest.main()
