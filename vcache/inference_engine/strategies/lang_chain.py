from langchain.schema import HumanMessage, SystemMessage

from vcache.inference_engine.inference_engine import InferenceEngine


class LangChainInferenceEngine(InferenceEngine):
    """
    LangChain-based inference engine supporting multiple providers.
    """

    def __init__(self, provider: str, model_name: str, temperature: float = 1):
        """
        Initialize LangChain inference engine.

        Args:
            provider: The provider to use (openai, anthropic, huggingface, google).
            model_name: The name of the model to use.
            temperature: The temperature parameter for generation.

        Raises:
            ValueError: If the provider is unsupported.
            ImportError: If the required LangChain module is not installed.
            Exception: If there's an error initializing the model.
        """
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
        """
        Create a response using the LangChain model.

        Args:
            prompt: The user prompt to process.
            system_prompt: Optional system prompt to set context.

        Returns:
            The generated response from the model.

        Raises:
            Exception: If there's an error creating the completion.
        """
        try:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))

            response = self.chat_model(messages)
            return response.content
        except Exception as e:
            raise Exception(f"Error creating completion from LangChain: {e}")
