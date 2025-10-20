import base64
import openai
from openai import OpenAI
import anthropic
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from tqdm import tqdm
from typing import Any, List, Optional, Sequence, Union
import google.generativeai as genai
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from mistralai import Mistral, UserMessage

import re

def api_models_map(model_name_or_path=None, token=None, base_url=None, **kwargs):
    model_name = model_name_or_path or ""
    if 'gpt-' in model_name:
        if 'vision' in model_name:
            return GPTV(model_name, token, base_url)
        else:
            return GPT(model_name, token, base_url)
    elif 'claude-' in model_name:
        return Claude(model_name, token)
    elif 'gemini-' in model_name:
        return Gemini(model_name, token)
    elif re.match(r'mistral-(tiny|small|medium|large)$', model_name):
        return MistralAI(model_name, token)
    return None

class GPT():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 60
    API_BACKOFF_FACTOR = 2
    API_MAX_SLEEP = 60
    API_MAX_CONCURRENCY = 5
    API_CONCURRENCY_ENV = "OPENAI_MAX_CONCURRENCY"

    def __init__(self, model_name, api_key, base_url=None):
        self.model_name = model_name
        # 如果没有提供 base_url，检查环境变量
        base_url = os.getenv('OPENAI_BASE_URL')
        api_key = os.getenv('OPENAI_API_KEY') 

        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=self.API_TIMEOUT)
        else:
            self.client = OpenAI(api_key=api_key, timeout=self.API_TIMEOUT)
        self._max_concurrency = self._resolve_max_concurrency()

    def _resolve_max_concurrency(self) -> int:
        env_value = os.getenv(self.API_CONCURRENCY_ENV)
        if env_value is not None:
            try:
                concurrency = int(env_value)
            except ValueError:
                print(f"Invalid {self.API_CONCURRENCY_ENV} value '{env_value}', using default {self.API_MAX_CONCURRENCY}.")
                concurrency = self.API_MAX_CONCURRENCY
        else:
            concurrency = self.API_MAX_CONCURRENCY
        return max(1, concurrency)

    def _sleep_backoff(self, attempt: int, base: Optional[float] = None) -> None:
        wait_base = base if base is not None else self.API_RETRY_SLEEP
        wait_time = min(wait_base * (self.API_BACKOFF_FACTOR ** max(0, attempt - 1)), self.API_MAX_SLEEP)
        if wait_time > 0:
            time.sleep(wait_time)

    def _normalize_prompt(self, prompt: Union[str, Sequence[Any], bytes, None]) -> str:
        if prompt is None:
            return ""

        if isinstance(prompt, str):
            return prompt

        if isinstance(prompt, bytes):
            return prompt.decode("utf-8", errors="ignore")

        if isinstance(prompt, Sequence):
            flattened_parts = []
            for part in prompt:
                if part is None:
                    continue
                part_str = part.decode("utf-8", errors="ignore") if isinstance(part, bytes) else str(part)
                part_str = part_str.strip()
                if part_str:
                    flattened_parts.append(part_str)
            if flattened_parts:
                return "\n\n".join(flattened_parts)
            return ""

        return str(prompt)

    def _generate(self, prompt: Union[str, Sequence[Any], bytes, None], 
                max_new_tokens: int, 
                temperature: float,
                top_p: float):
        output = self.API_ERROR_OUTPUT
        last_error: Optional[BaseException] = None
        normalized_prompt = self._normalize_prompt(prompt)
        for attempt in range(1, self.API_MAX_RETRY + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": normalized_prompt}],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                message_content = response.choices[0].message.content
                output = message_content if message_content is not None else ""
                break
            except getattr(openai, "APITimeoutError", openai.OpenAIError) as e:  # type: ignore[arg-type]
                last_error = e
                print(f"OpenAI timeout (attempt {attempt}/{self.API_MAX_RETRY}): {e}")
                self._sleep_backoff(attempt)
                continue
            except (getattr(openai, "RateLimitError", openai.OpenAIError),
                    getattr(openai, "APIConnectionError", openai.OpenAIError)) as e:  # type: ignore[arg-type]
                last_error = e
                print(f"OpenAI transient error (attempt {attempt}/{self.API_MAX_RETRY}): {e}")
                self._sleep_backoff(attempt)
                continue
            except openai.OpenAIError as e:
                last_error = e
                print(f"OpenAI API error (attempt {attempt}/{self.API_MAX_RETRY}): {e}")
                self._sleep_backoff(attempt, base=self.API_QUERY_SLEEP)
                continue
            except Exception as e:
                last_error = e
                print(f"Unexpected error during OpenAI generate (attempt {attempt}/{self.API_MAX_RETRY}): {e}")
                self._sleep_backoff(attempt, base=self.API_QUERY_SLEEP)
                continue

        if output == self.API_ERROR_OUTPUT and last_error is not None:
            print(f"Exhausted OpenAI retries after {self.API_MAX_RETRY} attempts: {last_error}")
        return output 
    
    def generate(self, 
                prompts: List[str],
                max_new_tokens: int, 
                temperature: float,
                top_p: float = 1.0,
                use_tqdm: bool=False,
                **kwargs):
        prompt_list = list(prompts)
        if not prompt_list:
            return []

        configured_concurrency = kwargs.get('concurrency')
        try:
            concurrency_hint = int(configured_concurrency) if configured_concurrency is not None else self._max_concurrency
        except (TypeError, ValueError):
            print(f"Invalid concurrency hint '{configured_concurrency}', using default {self._max_concurrency}.")
            concurrency_hint = self._max_concurrency

        max_workers = max(1, min(concurrency_hint, len(prompt_list)))

        progress_bar = tqdm(total=len(prompt_list)) if use_tqdm else None
        outputs: List[str] = [self.API_ERROR_OUTPUT] * len(prompt_list)

        def _generate_with_postprocess(idx: int, prompt: str) -> str:
            result = self._generate(prompt, max_new_tokens, temperature, top_p)
            if result == self.API_ERROR_OUTPUT:
                time.sleep(self.API_QUERY_SLEEP)
            return result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_generate_with_postprocess, idx, prompt): idx
                for idx, prompt in enumerate(prompt_list)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    outputs[idx] = future.result()
                except Exception as exc:
                    print(f"Exception during GPT.generate for prompt index {idx}: {exc}")
                    outputs[idx] = self.API_ERROR_OUTPUT
                finally:
                    if progress_bar is not None:
                        progress_bar.update(1)

        if progress_bar is not None:
            progress_bar.close()

        return outputs

class GPTV():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20

    def __init__(self, model_name, api_key, base_url=None):
        self.model_name = model_name
        # 如果没有提供 base_url，检查环境变量
        if base_url is None:
            base_url = os.getenv('OPENAI_BASE_URL')
        
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=self.API_TIMEOUT)
        else:
            self.client = OpenAI(api_key=api_key, timeout=self.API_TIMEOUT)


    def _generate(self, prompt: str,
                image_path: str,
                max_new_tokens: int, 
                temperature: float,
                top_p: float):
        output = self.API_ERROR_OUTPUT


        with open(image_path, "rb") as image_file:
            image_s = base64.b64encode(image_file.read()).decode('utf-8')
            image_url = {"url": f"data:image/jpeg;base64,{image_s}"}


        for _ in range(self.API_MAX_RETRY):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": image_url},
                            ],
                        }
                    ],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                output = response.choices[0].message.content
                break
            except openai.InvalidRequestError as e:
                if "Your input image may contain content that is not allowed by our safety system" in str(e):
                    output = "I'm sorry, I can't assist with that request."
                    break 
            except openai.OpenAIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
        
            time.sleep(self.API_QUERY_SLEEP)
        return output 
    
    def generate(self, 
                prompts: List[str],
                images: List[str],
                max_new_tokens: int, 
                temperature: float,
                top_p: float = 1.0,
                use_tqdm: bool=False,
                **kwargs):
        if use_tqdm:
            prompts = tqdm(prompts)

        return [self._generate(prompt, img, max_new_tokens, temperature, top_p) for prompt, img in zip(prompts, images)]

class Claude():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    default_output = "I'm sorry, but I cannot assist with that request."

    def __init__(self, model_name, api_key) -> None:
        self.model_name = model_name
        self.API_KEY = api_key

        self.model= Anthropic(
            api_key=self.API_KEY,
            )

    def _generate(self, prompt: str, 
                max_new_tokens: int, 
                temperature: float,
                top_p: float):

        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                completion = self.model.completions.create(
                    model=self.model_name,
                    max_tokens_to_sample=max_new_tokens,
                    prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}",
                    temperature=temperature,
                    top_p=top_p
                )
                output = completion.completion
                break
            except anthropic.BadRequestError as e:
                # as of Jan 2023, this show the output has been blocked
                if "Output blocked by content filtering policy" in str(e):
                    output = self.default_output
                    break
            except anthropic.APIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return output
    
    def generate(self, 
                prompts: List[str],
                max_new_tokens: int, 
                temperature: float,
                top_p: float = 1.0,  
                use_tqdm: bool=False, 
                **kwargs):

        if use_tqdm:
            prompts = tqdm(prompts)
        return [self._generate(prompt, max_new_tokens, temperature, top_p) for prompt in prompts]
        
class Gemini():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    default_output = "I'm sorry, but I cannot assist with that request."

    def __init__(self, model_name, token) -> None:
        self.model_name = model_name
        genai.configure(api_key=token)

        self.model = genai.GenerativeModel(model_name)

    def _generate(self, prompt: str, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):

        output = self.API_ERROR_OUTPUT
        generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_n_tokens,
                temperature=temperature,
                top_p=top_p)
        chat = self.model.start_chat(history=[])
        
        for _ in range(self.API_MAX_RETRY):
            try:
                completion = chat.send_message(prompt, generation_config=generation_config)
                output = completion.text
                break
            except (genai.types.BlockedPromptException, genai.types.StopCandidateException):
                # Prompt was blocked for safety reasons
                output = self.default_output
                break
            except Exception as e:
                print(type(e), e)
                # as of Jan 2023, this show the output has been filtering by the API
                if "contents.parts must not be empty." in str(e):
                    output = self.default_output
                    break
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(1)
        return output
    
    def generate(self, 
                prompts: List[str],
                max_new_tokens: int, 
                temperature: float,
                top_p: float = 1.0,  
                use_tqdm: bool=False, 
                **kwargs):
        if use_tqdm:
            prompts = tqdm(prompts)
        return [self._generate(prompt, max_new_tokens, temperature, top_p) for prompt in prompts]
    

class MistralAI():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5

    def __init__(self, model_name, token):    
        self.model_name = model_name
        self.client = Mistral(api_key=token)

    def _generate(self, prompt: str, 
                max_new_tokens: int, 
                temperature: float,
                top_p: float):
        
        output = self.API_ERROR_OUTPUT
        messages = [
            UserMessage(role="user", content=prompt),
        ]
        for _ in range(self.API_MAX_RETRY):
            try:
                chat_response = self.client.chat(
                    model=self.model,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    messages=messages,
                )
                output = chat_response.choices[0].message.content
                break
            except Exception as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return output 
    
    def generate(self, 
                prompts: List[str],
                max_new_tokens: int, 
                temperature: float,
                top_p: float = 1.0,
                use_tqdm: bool=False,
                **kwargs):
        
        if use_tqdm:
            prompts = tqdm(prompts)
        return [self._generate(prompt, max_new_tokens, temperature, top_p) for prompt in prompts]