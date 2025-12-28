"""Integration with Ollama for LLM inference"""

import logging
import aiohttp
import asyncio
from typing import AsyncGenerator, Optional
from config import OLLAMA_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaLLM:
    """Interface for Ollama LLM"""
    
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session = None
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def generate(
        self,
        prompt: str,
        stream: bool = True,
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Generate text using Ollama"""
        
        session = await self._ensure_session()
        
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": stream,
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
        }
        
        url = f"{self.base_url}/api/generate"
        
        try:
            logger.info(f"Generating response with {self.model}...")
            
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Ollama error {response.status}: {error_text}")
                    raise RuntimeError(f"Ollama API error: {error_text}")
                
                if stream:
                    async for line in response.content:
                        if line:
                            try:
                                data = line.decode('utf-8').strip()
                                if data:
                                    import json
                                    chunk_data = json.loads(data)
                                    if 'response' in chunk_data:
                                        yield chunk_data['response']
                            except Exception as e:
                                logger.debug(f"Error parsing stream: {e}")
                                continue
                else:
                    response_text = await response.text()
                    import json
                    data = json.loads(response_text)
                    yield data.get('response', '')
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            yield f"❌ Lỗi: {str(e)}"
    
    async def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text and return complete response"""
        
        response_text = ""
        async for chunk in self.generate(prompt, stream=True, system_prompt=system_prompt):
            response_text += chunk
        
        return response_text
    
    async def chat(self, messages: list, system_prompt: Optional[str] = None) -> str:
        """Chat interface using message history"""
        
        prompt_parts = []
        if system_prompt:
            prompt_parts.append(system_prompt)
        
        for msg in messages:
            role = msg.get('role', 'user').upper()
            content = msg.get('content', '')
            prompt_parts.append(f"{role}: {content}")
        
        prompt_parts.append("ASSISTANT:")
        
        full_prompt = "\n\n".join(prompt_parts)
        
        return await self.generate_text(full_prompt)
    
    async def check_connection(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            session = await self._ensure_session()
            url = f"{self.base_url}/api/tags"
            
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [m['name'] for m in data.get('models', [])]
                    
                    if self.model in models or any(self.model in m for m in models):
                        logger.info(f"✅ Ollama is running with model {self.model}")
                        return True
                    else:
                        logger.warning(f"Model {self.model} not found. Available: {models}")
                        return False
                else:
                    logger.error(f"Ollama connection failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            return False
    
    async def pull_model(self, model_name: str = None) -> bool:
        """Pull model from Ollama library"""
        if model_name is None:
            model_name = self.model
        
        try:
            session = await self._ensure_session()
            url = f"{self.base_url}/api/pull"
            
            payload = {"name": model_name}
            
            logger.info(f"Pulling model {model_name}...")
            
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    logger.info(f"✅ Model {model_name} pulled successfully")
                    return True
                else:
                    logger.error(f"Failed to pull model: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            return False
    
    async def get_model_info(self) -> dict:
        """Get information about the current model"""
        try:
            session = await self._ensure_session()
            url = f"{self.base_url}/api/show"
            
            payload = {"name": self.model}
            
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Error getting model info: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}
    
    def __del__(self):
        """Clean up session"""
        if self.session and not self.session.closed:
            try:
                asyncio.run(self.session.close())
            except:
                pass


class OllamaLLMSync:
    """Synchronous wrapper around async OllamaLLM"""
    
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = LLM_MODEL, **kwargs):
        self.async_llm = OllamaLLM(base_url, model, **kwargs)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text synchronously"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.async_llm.generate_text(prompt, system_prompt))
    
    def check_connection(self) -> bool:
        """Check connection synchronously"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.async_llm.check_connection())


async def test_ollama():
    """Test Ollama integration"""
    llm = OllamaLLM(model="qwen2.5:3b")
    
    connected = await llm.check_connection()
    if not connected:
        print("❌ Ollama not available")
        return
    
    print("✅ Ollama connected")
    
    prompt = "Hãy liệt kê 3 loại thuốc hạ sốt phổ biến:"
    print(f"\nPrompt: {prompt}")
    print("\nResponse:")
    
    async for chunk in llm.generate(prompt, stream=True):
        print(chunk, end='', flush=True)
    
    print("\n")


if __name__ == "__main__":
    asyncio.run(test_ollama())
