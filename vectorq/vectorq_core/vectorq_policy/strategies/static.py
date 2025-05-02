from typing_extensions import override

from vectorq.vectorq_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vectorq.vectorq_core.vectorq_policy.action import Action
from vectorq.vectorq_core.vectorq_policy.vectorq_policy import VectorQPolicy


class StaticThresholdPolicy(VectorQPolicy):
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    @override
    def select_action(
        self, similarity_score: float, metadata: EmbeddingMetadataObj
    ) -> Action:
        """
        Select action based on a static threshold.

        Args:
            similarity_score: The similarity score between the query and the embedding
            metadata: The metadata of the embedding

        Returns:
            Action.EXPLOIT if similarity_score >= threshold, Action.EXPLORE otherwise
        """
        if similarity_score >= self.threshold:
            return Action.EXPLOIT
        else:
            return Action.REJECT

    @override
    def update_policy(
        self, similarity_score: float, is_correct: bool, metadata: EmbeddingMetadataObj
    ) -> None:
        """
        No updates needed for static threshold policy.

        Args:
            similarity_score: The similarity score between the query and the embedding
            is_correct: Whether the query was correct
            metadata: The metadata of the embedding
        """
        # Static policy doesn't need updates
        pass
