import logging
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Optional

import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from typing_extensions import override

from vcache.config import VCacheConfig
from vcache.inference_engine import InferenceEngine
from vcache.vcache_core.cache.cache import Cache
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.embedding_store.embedding_store import EmbeddingStore
from vcache.vcache_core.similarity_evaluator import (
    SimilarityEvaluator,
)
from vcache.vcache_policy.vcache_policy import VCachePolicy

# Disable Hugging Face tokenizer parallelism to prevent deadlocks when using
# vCache in multi-threaded applications. This is a library-level fix.
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class CallbackQueue(queue.Queue):
    """
    A queue that processes items with a callback function in a worker thread.
    """

    def __init__(self, callback_function):
        """
        Initializes the CallbackQueue.

        Args:
            callback_function: The function to call for each item in the queue.
                               It will be executed by the worker thread.
        """
        super().__init__()
        self.callback_function = callback_function
        self._stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)

    def _worker(self):
        """
        The main loop for the worker thread.

        It continuously fetches items from the queue and processes them using the
        callback function. The loop includes a timeout to allow for graceful
        shutdown checks.
        """
        while True:
            should_stop = self._stop_event.is_set()
            if should_stop:
                break

            try:
                item = self.get(timeout=1)
                if item is None:  # Sentinel value to stop
                    break
                self.callback_function(item)
                self.task_done()
            except queue.Empty:
                continue

    def start(self):
        """Starts the worker thread."""
        self.worker_thread.start()

    def stop(self):
        """Stops the worker thread gracefully."""
        if self.worker_thread.is_alive():
            self.put(None)
            self.worker_thread.join()


class SigmoidOnlyDecisionPolicy(VCachePolicy):
    """
    Dynamic local threshold policy that computes optimal thresholds for each embedding.

    Computes the cross-entropy-loss-based optimal threshold per embedding. Does not take
    the user-defined accuracy target into account.
    """

    def __init__(self, delta: float = 0.01):
        """
        Initialize dynamic local threshold policy.

        Initializes the core algorithm and sets up placeholders for the thread
        pool executor and callback queue which will be created in `setup`.

        Args:
            delta: The delta value to use for threshold computation.
        """
        self.bayesian = _Algorithm(delta=delta)
        self.similarity_evaluator: Optional[SimilarityEvaluator] = None
        self.inference_engine: Optional[InferenceEngine] = None
        self.cache: Optional[Cache] = None
        self.logger: logging.Logger = logging.getLogger(__name__)

        self.executor: Optional[ThreadPoolExecutor] = None
        self.callback_queue: Optional[CallbackQueue] = None

    def __enter__(self):
        """
        Allows the policy to be used as a context manager.

        Returns:
            The policy instance itself.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Ensures graceful shutdown of background threads when exiting the context.

        This method is called automatically when exiting a `with` block,
        triggering the shutdown of the ThreadPoolExecutor and CallbackQueue.

        Args:
            exc_type: The exception type if an exception was raised in the `with` block.
            exc_val: The exception value if an exception was raised.
            exc_tb: The traceback if an exception was raised.
        """
        self.shutdown()

    @override
    def setup(self, config: VCacheConfig):
        """
        Setup the policy with the given configuration.

        This method initializes the cache, similarity evaluator, and inference
        engine. It also sets up and starts the background processing components:
        a ThreadPoolExecutor for concurrent tasks and a CallbackQueue for
        serialized cache updates.

        Args:
            config: The VCache configuration to use.
        """
        self.similarity_evaluator = config.similarity_evaluator
        self.inference_engine = config.inference_engine
        self.cache = Cache(
            embedding_engine=config.embedding_engine,
            embedding_store=EmbeddingStore(
                embedding_metadata_storage=config.embedding_metadata_storage,
                vector_db=config.vector_db,
            ),
            eviction_policy=config.eviction_policy,
        )

        self.callback_queue = CallbackQueue(
            callback_function=self.__perform_cache_update
        )
        self.callback_queue.start()
        self.executor = ThreadPoolExecutor(max_workers=64)

    def shutdown(self):
        """
        Shuts down the thread pool and callback queue gracefully.
        """
        if self.executor:
            self.executor.shutdown(wait=True)

        if self.callback_queue:
            self.callback_queue.stop()

    @override
    def process_request(
        self, prompt: str, system_prompt: Optional[str], id_set: int
    ) -> tuple[bool, str, EmbeddingMetadataObj]:
        """
        Process a request using dynamic local threshold policy.

        It determines whether to serve a cached response or generate a new one.
        If the policy decides to 'explore', it generates a new response and
        triggers an asynchronous background task to evaluate the decision and
        update the cache, without blocking the current request. The functions returns
        the actual response and some metadata information—whether the response is a
        cache hit and the nearest neighbor response—to enable further analysis.

        Args:
            prompt: The prompt to check for cache hit.
            system_prompt: The optional system prompt to use for the response. It will override the system prompt in the VCacheConfig if provided.
            id_set: The set identifier for the embedding. This is used in the
                benchmark to identify if the nearest neighbor is from the same set
                (if the cached response is correct or incorrect).

        Returns:
            Tuple containing [is_cache_hit, actual_response, nn_metadata_object].
        """
        if self.inference_engine is None or self.cache is None:
            raise ValueError("Policy has not been setup")

        knn = self.cache.get_knn(prompt=prompt, k=1)
        if not knn:
            response = self.inference_engine.create(
                prompt=prompt, system_prompt=system_prompt
            )
            self.cache.add(prompt=prompt, response=response, id_set=id_set)
            return False, response, EmbeddingMetadataObj(embedding_id=-1, response="")

        similarity_score, embedding_id = knn[0]

        try:
            nn_metadata: EmbeddingMetadataObj = self.cache.get_metadata(
                embedding_id=embedding_id
            )
        except Exception:
            # Cache eviction fallback
            new_response: str = self.inference_engine.create(
                prompt=prompt, system_prompt=system_prompt
            )
            self.cache.add(prompt=prompt, response=new_response, id_set=id_set)
            return (
                False,
                new_response,
                EmbeddingMetadataObj(embedding_id=-1, response=""),
            )

        action = self.bayesian.select_action(
            similarity_score=similarity_score, metadata=nn_metadata
        )

        match action:
            case _Action.EXPLOIT:
                return True, nn_metadata.response, nn_metadata
            case _Action.EXPLORE:
                response = self.inference_engine.create(
                    prompt=prompt, system_prompt=system_prompt
                )

                self.__update_cache(
                    response=response,
                    nn_metadata=nn_metadata,
                    similarity_score=similarity_score,
                    embedding_id=embedding_id,
                    prompt=prompt,
                    label_id_set=id_set,
                )

                return False, response, nn_metadata

    def __update_cache(
        self,
        response: str,
        nn_metadata: EmbeddingMetadataObj,
        similarity_score: float,
        embedding_id: int,
        prompt: str,
        label_id_set: int,
    ) -> None:
        """
        Asynchronously validates the correctness of the cached response and updates the cache.

        The validation whether the response is correct can involve a latency expensive LLM-judge call.
        Because this evaluation does not impact the returned response, we process it in the background.
        The LLM-judge call (or any other strategy like an embedding or string-based similarity check) in its own thread
        and returns a label (True/False) whether the response is correct.
        vCache maintains a global queue that waits for the labels. When a label gets available,
        vCache updates the metadata and the vector database accordingly.

        Args:
            response: The response to check for correctness.
            nn_metadata: The metadata of the nearest neighbor embedding.
            similarity_score: The similarity score between the query and the embedding.
            embedding_id: The id of the embedding.
            prompt: The prompt that was used to generate the response.
            label_id_set: The set identifier for the embedding. This is used in the
                benchmark to identify if the nearest neighbor is from the same set
                (if the cached response is correct or incorrect).
        """
        if self.executor is None:
            raise ValueError("Executor not initialized. Call setup() first.")

        self.executor.submit(
            self.__submit_for_background_update,
            response,
            similarity_score,
            embedding_id,
            prompt,
            nn_metadata.response,
            label_id_set,
            nn_metadata.id_set,
        )

    def __submit_for_background_update(
        self,
        new_response: str,
        similarity_score: float,
        embedding_id: int,
        prompt: str,
        cached_response: str,
        label_id_set: int,
        nn_id_set: int,
    ):
        """
        Submits a task to check answer similarity and queue a cache update.

        This method is executed by the ThreadPoolExecutor. It performs the
        potentially slow `answers_similar` check and then puts the result
        and context onto the `callback_queue` for sequential processing.

        Args:
            new_response: The newly generated response.
            similarity_score: The similarity between the prompt and the nearest neighbor.
            embedding_id: The ID of the nearest neighbor embedding.
            prompt: The original user prompt.
            cached_response: The response from the cached nearest neighbor.
            label_id_set: The id_set of the label.
            nn_id_set: The id_set of the nearest neighbor.
        """
        should_have_exploited = self.similarity_evaluator.answers_similar(
            a=new_response, b=cached_response, id_set_a=label_id_set, id_set_b=nn_id_set
        )
        self.callback_queue.put(
            (
                should_have_exploited,
                new_response,
                similarity_score,
                embedding_id,
                prompt,
                label_id_set,
            )
        )

    def __perform_cache_update(self, update_args: tuple) -> None:
        """
        Performs the actual cache update based on the background check.

        This method is executed sequentially by the CallbackQueue's worker
        thread, ensuring thread-safe updates to the cache metadata and
        vector database. It fetches the latest metadata before updating to
        prevent race conditions with evictions or other updates.

        Args:
            update_args: A tuple containing the context required for the update,
                         as passed from `__submit_for_background_update`. It
                         contains the following elements in order:
                         - should_have_exploited (bool): Whether the cache hit
                           should have been exploited.
                         - new_response (str): The newly generated response.
                         - similarity_score (float): The similarity score.
                         - embedding_id (int): The ID of the nearest neighbor.
                         - prompt (str): The original user prompt.
                         - id_set (int): The set identifier for the embedding. This is used in the
                            benchmark to identify if the nearest neighbor is from the same set
                            (if the cached response is correct or incorrect).
        """
        (
            should_have_exploited,
            new_response,
            similarity_score,
            embedding_id,
            prompt,
            id_set,
        ) = update_args

        try:
            latest_metdata_object = self.cache.get_metadata(embedding_id=embedding_id)
        except (ValueError, KeyError):
            logging.warning(
                f"Embedding {embedding_id} was evicted between the time the request was made and the time the update was processed. We can safely ignore this update."
            )
            return

        item_was_evicted = latest_metdata_object is None
        if item_was_evicted:
            return

        try:
            self.bayesian.add_observation_to_metadata(
                similarity_score=similarity_score,
                is_correct=should_have_exploited,
                metadata=latest_metdata_object,
            )
        except (ValueError, KeyError):
            self.logger.warning(
                f"Embedding {embedding_id} was evicted between the time the request was made and the time the update was processed. We can safely ignore this update."
            )
            return

        if not should_have_exploited:
            self.cache.add(prompt=prompt, response=new_response, id_set=id_set)

        try:
            self.cache.update_metadata(
                embedding_id=embedding_id, embedding_metadata=latest_metdata_object
            )
        except (ValueError, KeyError):
            self.logger.warning(
                f"Embedding {embedding_id} was evicted between the time the request was made and the time the update was processed. We can safely ignore this update."
            )
            return


class _Action(Enum):
    """
    Enumeration of possible actions for the algorithm.
    """

    EXPLORE = "explore"
    EXPLOIT = "exploit"


class _Algorithm:
    """
    Internal algorithm implementation for dynamic threshold computation.
    """

    def __init__(self, delta: float):
        """
        Initialize the algorithm with the given delta value.

        Args:
            delta: The delta value for confidence computation.
        """
        self.delta: float = delta
        self.P_c: float = 1.0 - self.delta
        self.epsilon_grid: np.ndarray = np.linspace(1e-6, 1 - 1e-6, 50)
        self.logistic_regression: LogisticRegression = LogisticRegression(
            penalty=None, solver="lbfgs", tol=1e-8, max_iter=1000, fit_intercept=False
        )

    def add_observation_to_metadata(
        self, similarity_score: float, is_correct: bool, metadata: EmbeddingMetadataObj
    ) -> None:
        """
        Update the metadata with the new observation

        Args
            similarity_score: float - The similarity score between the query and the embedding
            is_correct: bool - Whether the query was correct
            metadata: EmbeddingMetadataObj - The metadata of the embedding
        """
        if is_correct:
            metadata.observations.append((round(similarity_score, 3), 1))
        else:
            metadata.observations.append((round(similarity_score, 3), 0))

    def select_action(
        self, similarity_score: float, metadata: EmbeddingMetadataObj
    ) -> _Action:
        """
        Select the action to take based on the similarity score, observations, and accuracy target

        Args
            similarity_score: float - The similarity score between the query and the embedding
            metadata: EmbeddingMetadataObj - The metadata of the embedding
        Returns
            Action - Explore or Exploit
        """
        similarity_score = round(similarity_score, 3)
        similarities: np.ndarray = np.array([obs[0] for obs in metadata.observations])
        labels: np.ndarray = np.array([obs[1] for obs in metadata.observations])

        if len(similarities) < 6 or len(labels) < 6:
            return _Action.EXPLORE

        t_hat: float = self._estimate_parameters(
            similarities=similarities, labels=labels
        )

        if similarity_score >= t_hat:
            return _Action.EXPLOIT
        else:
            return _Action.EXPLORE

    def _estimate_parameters(
        self, similarities: np.ndarray, labels: np.ndarray
    ) -> _Action:
        """
        Optimize parameters with logistic regression

        Args
            labels: np.ndarray - The labels of the embeddings
            metadata: EmbeddingMetadataObj - The metadata of the embedding

        Returns
            t_hat: float - The estimated threshold
        """

        similarities = sm.add_constant(similarities)

        try:
            if len(similarities) != len(labels):
                print(f"len does not match: {len(similarities)} != {len(labels)}")
            self.logistic_regression.fit(similarities, labels)
            intercept, gamma = self.logistic_regression.coef_[0]

            gamma = max(gamma, 1e-6)
            t_hat = -intercept / gamma
            t_hat = float(np.clip(t_hat, 0.0, 1.0))

            return t_hat

        except Exception as e:
            print(f"Logistic regression failed: {e}")
            return -1.0, -1.0, -1.0
