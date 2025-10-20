import json
import gc
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
# https://huggingface.co/docs/accelerate/v0.11.0/en/memory#accelerate.find_executable_batch_size
from accelerate.utils import find_executable_batch_size
from ..baseline import SingleBehaviorRedTeamingMethod
from ..model_utils import load_model_and_tokenizer, get_template
from ..check_refusal_utils import check_refusal_completions
from .utils import (
    PromptGenerator,
    load_judge_model,
    GenerationWorker,
    OpenSourceResponder,
    ClosedSourceResponder,
)
import time

def running_time_record(start_time):
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))
    return elapsed_time, elapsed_seconds

class SRA(SingleBehaviorRedTeamingMethod):
    def __init__(self, 
                 judge_model,
                 target_model,
                 surrogate_model=None,
                 target_model_type="open_source",
                 batch_size=128,
                 top_p=0.99,
                 top_k=100,
                 threshold=1000000,
                 num_workers=0,
                 response_length=128,
                 prompt_length_threshold=1000,
                 use_chat_template=False,
                 **kwargs):
        # ===== Init Judge Model =====
        # self.judge_model, allocated_gpus = load_judge_model(**judge_model)
        self.judge_modle_config = judge_model
        # ===== Init Target Model =====
        # self.model_name_or_path = target_model['model_name_or_path']
        # # manually allocate target model into unallocated gpus
        # num_gpus = target_model.get('num_gpus', 1)
        # max_memory = {i: torch.cuda.mem_get_info(i)[0] if i >= allocated_gpus else 0 for i in range(0, allocated_gpus + num_gpus)}
        # self.model, self.tokenizer = load_model_and_tokenizer(**target_model, max_memory=max_memory)
        self.target_model_config = target_model
        self.target_model_config['num_gpus'] = kwargs.get('num_gpus', target_model.get('num_gpus', 1))
        self.target_model_type = target_model_type

        self._surrogate_shares_target = surrogate_model is None
        self.surrogate_model_config = surrogate_model.copy() if surrogate_model else dict(target_model)
        self.surrogate_model_config.setdefault('num_gpus', self.surrogate_model_config.get('num_gpus', kwargs.get('num_gpus', 1)))

        try:
            self.chat_template = get_template(target_model.get('model_name_or_path'), chat_template=target_model.get('chat_template'))
        except Exception:
            # Fallback to identity template for providers without tokenizer
            self.chat_template = {"description": "Plain prompt", "prompt": "{instruction}"}
        self.batch_size = batch_size
        self.top_p = top_p
        self.top_k = top_k
        self.threshold = threshold
        self.num_workers = num_workers
        self.response_length = response_length
        self.prompt_length_threshold = prompt_length_threshold
        self.use_chat_template = use_chat_template

        # Cached heavy resources to avoid reloading per behavior
        self._judge_model = None
        self._judge_num_gpus = None
        self._surrogate_worker = None
        self._target_responder = None

    def _ensure_shared_resources(self):
        """Lazily initialize judge and target model handles once."""
        if self._judge_model is None or self._judge_num_gpus is None:
            judge_kwargs = dict(self.judge_modle_config)
            model_name = judge_kwargs.pop("model_name_or_path", None)
            if model_name is None:
                raise ValueError("Judge model configuration must include 'model_name_or_path'")
            self._judge_model, self._judge_num_gpus = load_judge_model(
                model_name_or_path=model_name,
                **judge_kwargs,
            )

        if self._surrogate_worker is None:
            num_surrogate_gpus = self.surrogate_model_config.get('num_gpus', 1)
            max_memory = {
                i: torch.cuda.mem_get_info(i)[0] if i >= self._judge_num_gpus else 0
                for i in range(0, self._judge_num_gpus + num_surrogate_gpus)
            }
            self._surrogate_worker = GenerationWorker(
                target_model_config=self.surrogate_model_config,
                response_length=self.response_length,
                max_memory=max_memory,
                use_chat_template=self.use_chat_template
            )

        if self._target_responder is None:
            if self.target_model_type == "closed_source":
                self._target_responder = ClosedSourceResponder(
                    target_model_config=self.target_model_config,
                    response_length=self.response_length,
                    use_chat_template=self.use_chat_template,
                )
            else:
                # For open-source targets, reuse surrogate worker when possible
                if self._surrogate_shares_target or self.surrogate_model_config == self.target_model_config:
                    target_worker = self._surrogate_worker
                else:
                    num_target_gpus = self.target_model_config.get('num_gpus', 1)
                    max_memory = {
                        i: torch.cuda.mem_get_info(i)[0] if i >= self._judge_num_gpus else 0
                        for i in range(0, self._judge_num_gpus + num_target_gpus)
                    }
                    target_worker = GenerationWorker(
                        target_model_config=self.target_model_config,
                        response_length=self.response_length,
                        max_memory=max_memory,
                        use_chat_template=self.use_chat_template
                    )
                self._target_responder = OpenSourceResponder(target_worker)

    def generate_test_cases_single_behavior(self, behavior_dict, num_generate=1, verbose=False, defense_type=None, **kwargs):
        """
        Generates test cases for a single behavior

        :param behavior: a dictionary specifying the behavior to generate test cases for
        :param verbose: whether to print progress
        :return: a test case and logs
        """
        # Initializationd
        self._ensure_shared_resources()
        prompt_generator = PromptGenerator(
            self.judge_modle_config,
            self.target_model_config,
            self.response_length,
            self.prompt_length_threshold,
            self.threshold,
            self.chat_template,
            behavior_dict,
            self.use_chat_template,
            defense_type,
            preloaded_judge_model=self._judge_model,
            preloaded_judge_num_gpus=self._judge_num_gpus,
            surrogate_worker=self._surrogate_worker,
            target_responder=self._target_responder,
        )

        batch_size = self.batch_size

        # ========== Behavior meta data ==========
        behavior = behavior_dict['Behavior']
        context_str = behavior_dict['ContextString']
        behavior_id = behavior_dict['BehaviorID']

        if context_str:
            behavior = f"{context_str}\n\n---\n\n{behavior}"

        print(f'Behavior: {behavior_id}')

        logs = []
        test_case = behavior

        # ==== Generate prompt =====
        start_time = time.time()

        test_case, response = prompt_generator.generate_adv_prompt(test_case, batch_size, self.top_p, self.top_k)

        running_time, running_seconds = running_time_record(start_time)

        logs.append(dict(test_case=test_case, number_of_tried_prompts=prompt_generator.number_of_tried_prompts, successful_attack=prompt_generator.successful_attack, running_time=running_time, running_seconds=running_seconds, prompt_length=prompt_generator.prompt_length, response=response))

        del prompt_generator
        torch.cuda.empty_cache()
        gc.collect()

        return test_case, logs
