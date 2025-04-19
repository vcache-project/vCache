import asyncio
from typing import List, Tuple

from vectorq.config import VectorQConfig
from vectorq.vectorq_core.core import VectorQCore
from vectorq.inference_engine.inference_engine import InferenceEngine

class VectorQBenchmark:
    
    def __init__(
        self,
        candidate_embedding: List[float],
        candidate_response: str,
    ):
        self.candidate_embedding = candidate_embedding
        self.candidate_response = candidate_response
        
        
class VectorQ:
    '''
    VectorQ is a main class that contains the VectorQ semantic prompt caching system.
    '''
    def __init__(self, vectorq_config = VectorQConfig()):
        self.vectorq_config = vectorq_config
        self.request_queue = asyncio.Queue()
        self.queue_processor_task = None
        self.running = False
        
        try:
            self.inference_engine = self.vectorq_config.inference_engine
            self.core = VectorQCore(vectorq_config=vectorq_config)
            self.start_queue_processor()
        except Exception as e:
            print(f"Error initializing VectorQ: {e}")
            raise Exception(f"Error initializing VectorQ: {e}")
    
    def start_queue_processor(self):
        self.running = True
        self.queue_processor_task = asyncio.create_task(self._process_request_queue())
    
    async def _process_request_queue(self):
        while self.running:
            try:
                request = await self.request_queue.get()
                
                prompt = request['prompt']
                output_format = request['output_format']
                benchmark = request['benchmark']
                future = request['future']
                
                try:
                    if self.vectorq_config.enable_cache:
                        cache_hit, response = self.core.process_request(prompt, benchmark, output_format)
                        future.set_result((response, cache_hit))
                    else:
                        response = self.inference_engine.create(prompt, output_format)
                        future.set_result((response, False))
                except Exception as e:
                    future.set_result((f"[ERROR] Failed to process: {prompt}", False))
                
                self.request_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in queue processor: {e}")
                continue

    async def create(self, prompt: str, output_format: str = None, benchmark: VectorQBenchmark = None) -> Tuple[str, bool]:
        '''
            prompt: str - The prompt to create a response for.
            benchmark: VectorQBenchmark - The optional benchmark object containing the pre-computed embedding and response.
            Returns: Tuple[str, bool] - The response to the prompt and whether the response was cached.
        '''
        future = asyncio.Future()
        request = {
            'prompt': prompt,
            'output_format': output_format,
            'benchmark': benchmark,
            'future': future
        }
        await self.request_queue.put(request)
        
        return await future

    def import_data(self, data: List[str]) -> bool:
        # TODO
        return True

    def flush(self) -> bool:
        # TODO
        return True
    
    def get_statistics(self) -> str:
        # TODO
        return "No statistics available"
    
    def get_inference_engine(self) -> InferenceEngine:
        # TODO
        return self.inference_engine
    
    async def shutdown(self):
        if self.queue_processor_task:
            self.running = False
            self.queue_processor_task.cancel()
            try:
                await self.queue_processor_task
            except asyncio.CancelledError:
                pass
        