# src/llm_connector.py
import os
import json
import requests
from typing import Dict, List, Any, Optional
import logging
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
import yaml

logger = logging.getLogger("bias_detection.llm_connector")

class LLMConnector:
    """Handles connections to various LLM APIs."""
    
    def __init__(self, config_path: str = "configs/llm_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.clients = {}
        self._initialize_clients()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load LLM configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("Loaded LLM configuration")
            return config
        except Exception as e:
            logger.error(f"Error loading LLM config: {e}")
            return {}
    
    def _initialize_clients(self) -> None:
        """Initialize API clients for supported providers."""
        providers = self.config.get('providers', {})
        
        # OpenAI client
        if providers.get('openai', {}).get('enabled', False):
            openai_config = providers['openai']
            api_key = openai_config.get('api_key_env_var')
            base_url = openai_config.get('base_url', 'https://api.openai.com/v1')
            
            if api_key:
                self.clients['openai'] = OpenAI(api_key=api_key, base_url=base_url)
                logger.info("OpenAI client initialized successfully")
            else:
                logger.warning("OpenAI API key not found in config")
        
        # Google Gemini client
        if providers.get('gemini', {}).get('enabled', False):
            gemini_config = providers['gemini']
            api_key = gemini_config.get('api_key_env_var')
            
            if api_key:
                genai.configure(api_key=api_key)
                self.clients['google'] = genai
                logger.info("Google Gemini client initialized successfully")
            else:
                logger.warning("Google Gemini API key not found in config")
        
        # Anthropic client (if you add it later)
        anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic_api_key:
            self.clients['anthropic'] = Anthropic(api_key=anthropic_api_key)
            logger.info("Anthropic client initialized successfully")
        
        logger.info(f"Initialized clients for: {list(self.clients.keys())}")
    
    def _get_provider_for_model(self, model_name: str) -> Optional[str]:
        """Determine provider for a given model name."""
        providers = self.config.get('providers', {})
        
        for provider_name, provider_config in providers.items():
            if provider_config.get('enabled', False):
                models = provider_config.get('models', [])
                if model_name in models:
                    return provider_name
        
        # Map common model names to providers
        model_provider_map = {
            'gpt-4o': 'openai',
            'gpt-4o-mini': 'openai',
            'gemini-1.5-pro': 'gemini',
            'grok-1': 'grok',
            'mistral-tiny': 'mistral',
            'qwen-7b': 'qwen',
            'deepseek-1': 'deepseek'
        }
        
        return model_provider_map.get(model_name)
    
    def _call_openai(self, model: str, prompt: str, **kwargs) -> str:
        """Call OpenAI API."""
        client = self.clients.get('openai')
        if not client:
            raise ValueError("OpenAI client not initialized")
        
        # Get provider-specific parameters
        provider_config = self.config.get('providers', {}).get('openai', {})
        default_params = {
            'temperature': provider_config.get('temperature', 0.7),
            'max_tokens': provider_config.get('max_tokens', 500),
            **kwargs
        }
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=default_params.get('temperature'),
                max_tokens=default_params.get('max_tokens'),
                top_p=default_params.get('top_p', 1.0)
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error for {model}: {e}")
            raise
    
    def _call_google(self, model: str, prompt: str, **kwargs) -> str:
        """Call Google Gemini API."""
        client = self.clients.get('google')
        if not client:
            raise ValueError("Google Gemini client not initialized")
        
        # Get provider-specific parameters
        provider_config = self.config.get('providers', {}).get('gemini', {})
        default_params = {
            'temperature': provider_config.get('temperature', 0.7),
            'max_tokens': provider_config.get('max_tokens', 500),
            **kwargs
        }
        
        try:
            model_obj = client.GenerativeModel(model)
            response = model_obj.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=default_params.get('max_tokens'),
                    temperature=default_params.get('temperature'),
                    top_p=default_params.get('top_p', 1.0)
                )
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Google Gemini API error for {model}: {e}")
            raise
    
    def _call_anthropic(self, model: str, prompt: str, **kwargs) -> str:
        """Call Anthropic API."""
        client = self.clients.get('anthropic')
        if not client:
            raise ValueError("Anthropic client not initialized")
        
        default_params = self.config.get('default_parameters', {})
        params = {**default_params, **kwargs}
        
        try:
            response = client.messages.create(
                model=model,
                max_tokens=params.get('max_tokens', 1000),
                temperature=params.get('temperature', 0.1),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error for {model}: {e}")
            raise
    
    def _call_generic_api(self, provider: str, model: str, prompt: str, **kwargs) -> str:
        """Call generic REST API for providers like Grok, Mistral, Qwen, DeepSeek."""
        provider_config = self.config.get('providers', {}).get(provider, {})
        api_key = provider_config.get('api_key_env_var')
        base_url = provider_config.get('base_url')
        
        if not api_key:
            raise ValueError(f"{provider} API key not found in config")
        if not base_url:
            raise ValueError(f"{provider} base URL not found in config")
        
        # Get provider-specific parameters
        default_params = {
            'temperature': provider_config.get('temperature', 0.7),
            'max_tokens': provider_config.get('max_tokens', 500),
            **kwargs
        }
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': default_params.get('temperature'),
            'max_tokens': default_params.get('max_tokens'),
            'top_p': default_params.get('top_p', 1.0)
        }
        
        try:
            # Different endpoints for different providers
            if provider in ['grok', 'qwen', 'deepseek']:
                endpoint = f"{base_url}/chat/completions"
            elif provider == 'mistral':
                endpoint = f"{base_url}/chat/completions"
            else:
                endpoint = f"{base_url}/chat/completions"
            
            response = requests.post(endpoint, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Handle different response formats
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content'].strip()
            elif 'output' in result and 'choices' in result['output']:
                return result['output']['choices'][0]['message']['content'].strip()
            else:
                logger.warning(f"Unexpected response format from {provider}: {result}")
                return str(result)
                
        except Exception as e:
            logger.error(f"{provider} API error for {model}: {e}")
            raise
    
    def generate_response(self, model_name: str, prompt: str, **kwargs) -> str:
        """
        Generate response from specified LLM.
        
        Args:
            model_name: Name of the LLM model
            prompt: Input prompt
            **kwargs: Additional parameters
        
        Returns:
            Model response text
        """
        provider = self._get_provider_for_model(model_name)
        
        if not provider:
            raise ValueError(f"Unknown model or provider not configured: {model_name}")
        
        logger.info(f"Generating response using {model_name} ({provider})")
        
        try:
            if provider == 'openai':
                return self._call_openai(model_name, prompt, **kwargs)
            elif provider == 'gemini':
                return self._call_google(model_name, prompt, **kwargs)
            elif provider == 'anthropic':
                return self._call_anthropic(model_name, prompt, **kwargs)
            elif provider in ['grok', 'mistral', 'qwen', 'deepseek']:
                return self._call_generic_api(provider, model_name, prompt, **kwargs)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        
        except Exception as e:
            logger.error(f"Error generating response from {model_name}: {e}")
            raise
    
    def batch_generate_responses(self, 
                               model_name: str, 
                               prompts: List[str], 
                               **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        Args:
            model_name: Name of the LLM model
            prompts: List of input prompts
            **kwargs: Additional parameters
        
        Returns:
            List of model responses
        """
        responses = []
        
        for i, prompt in enumerate(prompts):
            try:
                logger.debug(f"Processing prompt {i+1}/{len(prompts)}")
                response = self.generate_response(model_name, prompt, **kwargs)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error processing prompt {i+1}: {e}")
                responses.append("")  # Empty response on error
        
        logger.info(f"Generated {len(responses)} responses from {model_name}")
        return responses
    
    def test_connection(self, model_name: str) -> bool:
        """
        Test connection to LLM API.
        
        Args:
            model_name: Name of the model to test
        
        Returns:
            True if connection successful
        """
        try:
            test_prompt = "Hello, please respond with just 'OK'."
            response = self.generate_response(model_name, test_prompt, max_tokens=10)
            return response.strip().upper() == "OK"
        except Exception as e:
            logger.error(f"Connection test failed for {model_name}: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models from enabled providers.
        
        Returns:
            List of model names
        """
        available_models = []
        providers = self.config.get('providers', {})
        
        for provider_name, provider_config in providers.items():
            if provider_config.get('enabled', False):
                models = provider_config.get('models', [])
                available_models.extend(models)
        
        return available_models