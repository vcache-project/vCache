from langchain.schema import HumanMessage, SystemMessage

from vectorq.inference_engine.inference_engine import InferenceEngine


class LangChainInferenceEngine(InferenceEngine):
    def __init__(self, provider: str, model_name: str, temperature: float = 1):
        super().__init__()
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature

        try:
            if provider == "openai":
                from langchain_openai import ChatOpenAI

                self.chat_model = ChatOpenAI(
                    model_name=model_name,
                    temperature=self.temperature,
                )
            elif provider == "anthropic":
                from langchain_anthropic import ChatAnthropic

                self.chat_model = ChatAnthropic(
                    model_name=model_name,
                    temperature=self.temperature,
                )
            elif provider == "huggingface":
                from langchain_huggingface import HuggingFaceEndpoint

                self.chat_model = HuggingFaceEndpoint(
                    repo_id=model_name,
                    temperature=self.temperature,
                )
            elif provider == "google":
                from langchain_google_genai import ChatGoogleGenerativeAI

                self.chat_model = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=self.temperature,
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        except ImportError:
            raise ImportError(
                f"Could not import LangChain module for provider: {provider}. "
                f"Make sure to install langchain_{provider}"
            )
        except Exception as e:
            raise Exception(f"Error initializing LangChain model: {e}")

    def create(self, prompt: str, system_prompt: str = None) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))

            response = self.chat_model(messages)
            return response.content
        except Exception as e:
            raise Exception(f"Error creating completion from LangChain: {e}")
