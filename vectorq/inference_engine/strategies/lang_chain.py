from typing import TYPE_CHECKING
from langchain.schema import HumanMessage, SystemMessage

if TYPE_CHECKING:
    from vectorq.config import VectorQConfig
    
from vectorq.inference_engine.strategy import InferenceEngineStrategy

class LangChain(InferenceEngineStrategy):
    
    def __init__(self, vectorq_config: "VectorQConfig"):
        super().__init__(vectorq_config=vectorq_config)

        # Parse model name to determine provider and model
        # Expected format: "provider/model_name" (e.g., "openai/gpt-4", "anthropic/claude-2")
        model_parts = self.vectorq_config._inference_engine_model_name.split('/', 1)
        
        if len(model_parts) != 2:
            raise ValueError(f"Invalid model name format: {self.vectorq_config._inference_engine_model_name}. " 
                            "Expected format: 'provider/model_name'")
            
        provider, model_name = model_parts
        try:
            if provider == "openai":
                from langchain_openai import ChatOpenAI
                self.chat_model = ChatOpenAI(
                    model_name=model_name,
                    temperature=self.vectorq_config._inference_engine_temperature,
                )
            elif provider == "anthropic":
                from langchain_anthropic import ChatAnthropic
                self.chat_model = ChatAnthropic(
                    model_name=model_name,
                    temperature=self.vectorq_config._inference_engine_temperature,
                )
            elif provider == "huggingface":
                from langchain_huggingface import HuggingFaceEndpoint
                self.chat_model = HuggingFaceEndpoint(
                    repo_id=model_name,
                    temperature=self.vectorq_config._inference_engine_temperature,
                )
            elif provider == "google":
                from langchain_google_genai import ChatGoogleGenerativeAI
                self.chat_model = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=self.vectorq_config._inference_engine_temperature,
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        except ImportError:
            raise ImportError(f"Could not import LangChain module for provider: {provider}. "
                            f"Make sure to install langchain_{provider}")
        except Exception as e:
            raise Exception(f"Error initializing LangChain model: {e}")
    
    def create(self, prompt: str, output_format: str = None) -> str:
        try:
            messages = []
            if output_format:
                messages.append(SystemMessage(content=output_format))
            messages.append(HumanMessage(content=prompt))
            
            response = self.chat_model(messages)
            return response.content
        except Exception as e:
            raise Exception(f"Error creating completion from LangChain: {e}")
