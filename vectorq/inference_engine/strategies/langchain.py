from typing import Any, Optional

from langchain.schema import HumanMessage, SystemMessage
from typing_extensions import override

from vectorq.inference_engine.inference_engine import InferenceEngine


class LangChainInferenceEngine(InferenceEngine):
    def __init__(
        self,
        provider: str,
        model_name: str,
        temperature: float = 1,
        system_prompt: Optional[str] = None,
        **client_kwargs: Any,
    ):
        """
        Initialize the LangChain inference engine.
        Args
            provider: str - The provider to use. Supported providers are: openai, anthropic, huggingface, google.
            model_name: str - The model to use by default.
            temperature: float - The temperature to use by default.
            **client_kwargs: Any - Additional keyword arguments to pass to the LangChain client. Such as api_key, base_url, etc.
        """
        super().__init__()
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.client_kwargs = client_kwargs
        self._chat_model = None

    @property
    def chat_model(self):
        """
        Lazily initialize the LangChain chat model only when needed.
        Returns
            The LangChain chat model for the specified provider.
        """
        if self._chat_model is None:
            try:
                if self.provider == "openai":
                    from langchain_openai import ChatOpenAI

                    self._chat_model = ChatOpenAI(
                        model_name=self.model_name,
                        temperature=self.temperature,
                        **self.client_kwargs,
                    )
                elif self.provider == "anthropic":
                    from langchain_anthropic import ChatAnthropic

                    self._chat_model = ChatAnthropic(
                        model_name=self.model_name,
                        temperature=self.temperature,
                        **self.client_kwargs,
                    )
                elif self.provider == "huggingface":
                    from langchain_huggingface import HuggingFaceEndpoint

                    self._chat_model = HuggingFaceEndpoint(
                        repo_id=self.model_name,
                        temperature=self.temperature,
                        **self.client_kwargs,
                    )
                elif self.provider == "google":
                    from langchain_google_genai import ChatGoogleGenerativeAI

                    self._chat_model = ChatGoogleGenerativeAI(
                        model=self.model_name,
                        temperature=self.temperature,
                        **self.client_kwargs,
                    )
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")
            except ImportError:
                raise ImportError(
                    f"Could not import LangChain module for provider: {self.provider}. "
                    f"Make sure to install langchain_{self.provider}"
                )
            except Exception as e:
                raise Exception(f"Error initializing LangChain model: {e}")
        return self._chat_model

    @override
    def infer(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Infer a response from the LangChain inference engine.
        Args
            prompt: str - The prompt to create a response for.
            system_prompt: Optional[str] - The optional system prompt to use for the response.
            **kwargs: Any - Additional keyword arguments to pass to the LangChain client.
        Returns
            str - The response of the prompt.
        """
        try:
            # Build messages array with optional system prompt
            messages = []
            system_prompt = system_prompt or self.system_prompt
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))

            # Call the model with messages and additional kwargs
            response = self.chat_model.invoke(messages, **kwargs)
            return response.content
        except Exception as e:
            raise Exception(f"Error creating completion from LangChain: {e}")
