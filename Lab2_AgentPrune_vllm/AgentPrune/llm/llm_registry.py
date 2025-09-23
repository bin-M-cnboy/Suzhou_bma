import os
from typing import List, Optional, Union
from class_registry import ClassRegistry
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

from AgentPrune.llm.llm import LLM
from AgentPrune.llm.format import Message


class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        if model_name is None or model_name=="":
            model_name = "/data0/bma/models/Qwen3-8B"

        if model_name == '/data0/bma/models/Qwen3-8B':
            model = cls.registry.get(model_name)
        elif model_name.startswith('qwen'): 
            model = VLLMChat(model_name)
        else: # any version of GPTChat like "gpt-4o"
            model = cls.registry.get('GPTChat', model_name)

        return model


class VLLMChat(LLM):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=os.environ.get("base_url", "http://localhost:8001/v1"),
            api_key=os.environ.get("api_key", "vllm-Qwen3-8B-Instruct-123456"),
        )

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        # vLLM does not support async calls directly with the current setup
        # For simplicity, we'll call the sync method. In a real async app,
        # you'd use an async HTTP client or run in a thread pool.
        return self.gen(messages, max_tokens, temperature, num_comps)

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        # Convert internal Message format to OpenAI chat completion format
        openai_messages = [{
            "role": msg.role,
            "content": msg.content
        } for msg in messages]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                max_tokens=max_tokens if max_tokens is not None else self.DEFAULT_MAX_TOKENS,
                temperature=temperature if temperature is not None else self.DEFAULT_TEMPERATURE,
                n=num_comps if num_comps is not None else self.DEFUALT_NUM_COMPLETIONS,
            )
            if num_comps and num_comps > 1:
                return [choice.message.content for choice in response.choices if choice.message.content is not None]
            else:
                return response.choices[0].message.content if response.choices[0].message.content is not None else ""
        except Exception as e:
            print(f"Error calling vLLM API: {e}")
            return ""

LLMRegistry.register("/data0/bma/models/Qwen3-8B")(VLLMChat)
