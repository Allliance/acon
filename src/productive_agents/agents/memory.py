# Managing all memory features
import os
import sys
import json
import logging
from copy import deepcopy
from typing import List, Dict, Any, Optional, Union
from productive_agents.ctxopt.obs_optimizer import ObservationOptimizer
from productive_agents.ctxopt.history_optimizer import HistoryOptimizer

class MemoryManager:
    """Manages the history of interactions for the LLM agent."""
    
    def __init__(self, config: Dict[str, Any], preinit_model: Optional[Any] = None):
        # Initialize conversation history - list of conversation sessions
        # Each session is a list of messages [system, user, assistant, user, assistant, ...]
        self.llm_history: List[List[Dict[str, str]]] = [[]]
        self.entire_session: List[Dict[str, str]] = [] # Flattened entire session history
        
        # Track alignment between conversation steps and environment steps for visualization
        self.step_alignment: List[List[int]] = []

        self.step = 0 # Track total steps taken in whole sessions

        # Initialize context optimization if configured
        co_config = getattr(config, "co_config", None)
        self.obs_optimizer = None
        self.do_observation_optimization = False
        self.add_history_summary_obs_opt = co_config.get("add_history_summary_obs_opt", False) \
            if co_config else False  # Whether to add history summary to the conversation

        self.history_optimizer = None
        self.do_history_optimization = False
        self.prev_history_summary = None
        # One entry per compression event: the exact session that replaces the
        # old history (system + rebuilt user prompt + preserved/selected turns),
        # with per-message and total token counts so the budget is verifiable.
        self.post_compression_snapshots: List[Dict[str, Any]] = []
        self.preserve_last_k_turns = co_config.get("preserve_last_k_turns", 1) \
            if co_config else 1  # Number of turns to preserve without summarization

        self.first_user_prompt = None # Store the first user prompt for reference

        # history summary strategy:
        # - accumulate: accumulate the history summaries
        # - reset: reset the history summary after each optimization
        self.history_summary_rule = co_config.get("history_summary_rule", "accumulate") if co_config else "accumulate"  # Rule for history summarization
        # baseline strategy:
        # - none: just use history optimizer
        # - discard: discard the previous turns and only keep the last k turns
        # - retrieve: retrieve n turns from the history and only keep the last k turns
        self.baseline_strategy = co_config.get("baseline_strategy", "none") if co_config else "none"  # Strategy for baseline optimization

        self.history_summary_interval = co_config.get("history_summary_interval", -1) if co_config else -1  # Interval for history summarization
        self.retrieve_turns = co_config.get("retrieve_turns", 5) if co_config else 5  # Number of turns to retrieve in "retrieve" baseline strategy
        # Token budget used by selection-based baselines (fifo / mask_obs / mask_action / random).
        self.compression_budget = co_config.get("compression_budget", 2048) if co_config else 2048
        self.random_seed = co_config.get("random_seed", 0) if co_config else 0

        hist_version = co_config.get("history_version", 1) if co_config else 1
        hist_cls = HistoryOptimizer
        if self.baseline_strategy == "retrieve":
            from productive_agents.ctxopt.history_optimizer import HistoryRetriever
            hist_cls = HistoryRetriever

        obs_version = co_config.get("obs_version", 1) if co_config else 1
        obs_cls = ObservationOptimizer
        
        if co_config:
            # load config
            # debug_mode = getattr(config, "debug_mode", False)
            debug_mode = True

            co_type = co_config.get("type", None)
            if co_type == "obs":
                self.obs_optimizer = obs_cls(
                    co_config,
                    debug_mode=debug_mode,
                    llm=preinit_model,  # Use pre-initialized model if available
                )
                self.history_optimizer = None
                self.do_observation_optimization = True
            elif co_type == "history":
                self.history_optimizer = hist_cls(
                    co_config, 
                    debug_mode=debug_mode,
                    llm=preinit_model  # Use pre-initialized model if available
                )
                self.obs_optimizer = None
                self.do_history_optimization = True
            elif co_type == "unified":
                self.obs_optimizer = obs_cls(
                    co_config, 
                    debug_mode=debug_mode,
                    llm=preinit_model  # Use pre-initialized model if available
                )
                self.history_optimizer = hist_cls(
                    co_config, 
                    debug_mode=debug_mode,
                    llm=preinit_model  # Use pre-initialized model if available
                )
                self.do_observation_optimization = True
                self.do_history_optimization = True
            else:
                raise ValueError(f"Unknown context optimization type: {co_type}")

        # ---- Best-of-N compression selection (online divergence / rubric) ----
        # When co_config has `compression_selection.enabled`, each LLM-history
        # compression generates N candidate summaries and installs the one that
        # least perturbs the agent's near-future plan (or scores highest on a
        # rubric). Only meaningful for the LLM-summary path (baseline_strategy
        # == "none"); selection baselines (fifo/random/...) are untouched.
        self.compression_selection_cfg = (
            co_config.get("compression_selection") if co_config else None
        )
        self.compression_selector = None
        self.compression_selection_log: List[Dict[str, Any]] = []
        self.n_candidates = 0
        self.candidate_temperature = 0.6
        self.candidate_seed = 0
        if (
            self.compression_selection_cfg
            and self.compression_selection_cfg.get("enabled")
            and self.do_history_optimization
            and self.history_optimizer is not None
            and self.baseline_strategy == "none"
        ):
            self._init_compression_selector(config, co_config)

    def _init_compression_selector(self, config: Any, co_config: Dict[str, Any]) -> None:
        """Build a CompressionSelector from `co_config['compression_selection']`.

        Resolves the agent endpoint/model (for divergence plan forecasting) from
        the selection config, then the agent config, then the environment. Any
        failure (missing judge key, unreachable endpoint at import time, etc.)
        disables selection with a warning rather than crashing the run — the
        optimizer then falls back to a single compression.
        """
        cfg = self.compression_selection_cfg
        logger = logging.getLogger(__name__)
        try:
            # The divergence package lives under experiments/<bench>/ which is
            # the agent's cwd at run time; make sure it is importable.
            cwd = os.getcwd()
            if cwd not in sys.path:
                sys.path.insert(0, cwd)
            from divergence.online_selection import CompressionSelector

            scorer = cfg.get("scorer", "divergence")
            agent_model = (
                cfg.get("agent_model")
                or getattr(config, "model_name", None)
                or os.environ.get("MODEL_NAME")
            )
            agent_base_url = (
                cfg.get("agent_base_url")
                or os.environ.get("VLLM_BASE_URL")
            )
            self.n_candidates = int(cfg.get("n_candidates", 5))
            self.candidate_temperature = float(cfg.get("candidate_temperature", 0.6))
            self.candidate_seed = int(cfg.get("candidate_seed", 0))

            self.compression_selector = CompressionSelector(
                agent_base_url=agent_base_url,
                agent_model=agent_model,
                scorer=scorer,
                n_actions=int(cfg.get("n_actions", 5)),
                max_gen_tokens=int(cfg.get("max_gen_tokens", 2048)),
                agent_temperature=float(cfg.get("agent_temperature", 0.0)),
                agent_enable_thinking=bool(cfg.get("agent_enable_thinking", False)),
                judge_model=cfg.get("judge_model", "gemini-3.5-flash"),
                judge_thinking_budget=cfg.get("judge_thinking_budget", None),
                seed=int(cfg.get("seed", 42)),
            )
            logger.info(
                "Compression selection ENABLED (scorer=%s, n_candidates=%d, "
                "n_actions=%d, agent=%s @ %s)",
                scorer, self.n_candidates, int(cfg.get("n_actions", 5)),
                agent_model, agent_base_url,
            )
        except Exception as e:  # noqa: BLE001
            self.compression_selector = None
            logger.warning("Compression selection disabled (init failed): %r", e)

    def _summary_to_user_prompt(self, summary: str, current_session: List[Dict[str, str]]) -> str:
        """Rebuild the post-compression user prompt from a summary, honoring
        `history_summary_rule` (identical text to the inline build below)."""
        if self.history_summary_rule == "reset":
            base = self.first_user_prompt
        elif self.history_summary_rule == "accumulate":
            base = current_session[1]["content"]
        else:
            raise NotImplementedError(f"Unknown history summary rule: {self.history_summary_rule}")
        return base + "\n\n<HISTORY_SUMMARY>\n" + summary + "\n</HISTORY_SUMMARY>"

    def _run_compression_selection(
        self,
        task: str,
        history_text: str,
        history_for_summarization: List[Dict[str, str]],
        current_session: List[Dict[str, str]],
        preserved_turns: List[Dict[str, str]],
    ) -> str:
        """Generate N candidate summaries and pick the least-divergent one.

        Returns the chosen summary string. On any failure, falls back to a
        single ordinary compression so the run keeps making progress. Records a
        per-event entry in `self.compression_selection_log` and logs the chosen
        candidate into the history optimizer's history (for parity with the
        single-compression path).
        """
        selector = self.compression_selector
        n = self.n_candidates

        # 1. Generate N candidate summaries from the compressor.
        candidates = self.history_optimizer.generate_candidates(
            task=task,
            history=history_text,
            prev_history_summary=self.prev_history_summary,
            n=n,
            temperature=self.candidate_temperature,
            base_seed=self.candidate_seed,
        )
        candidates = [c for c in candidates if (c or "").strip()]
        if not candidates:
            raise RuntimeError("compressor returned no usable candidates")

        # 2. Build the candidate sessions (what each summary would install) and
        #    the uncompressed reference session.
        cand_inputs = []
        for s in candidates:
            up = self._summary_to_user_prompt(s, current_session)
            msgs = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": up},
            ] + [dict(m) for m in preserved_turns]
            cand_inputs.append({"summary": s, "messages": msgs})
        reference_messages = deepcopy(current_session)

        # 3. Score and select.
        result = selector.select(
            task=task,
            reference_messages=reference_messages,
            history_text=history_text,
            candidates=cand_inputs,
        )
        best = result["best_index"]
        chosen = candidates[best]

        # 4. Logging — per-event selection record + parity entry in the
        #    history-optimizer history (so history_optimizer_history.json still
        #    shows the compression that was actually installed).
        cand_log = []
        for rec in result.get("candidates", []):
            entry = {
                "index": rec.get("index"),
                "score": rec.get("score"),
                "summary_tokens": self._count_tokens(rec.get("summary", "")),
                "judge_reasoning": rec.get("judge_reasoning"),
            }
            if "predicted" in rec:
                entry["predicted_plan"] = rec["predicted"]
            if "dimensions" in rec:
                entry["dimensions"] = rec["dimensions"]
            if "error" in rec:
                entry["error"] = rec["error"]
            cand_log.append(entry)
        self.compression_selection_log.append({
            "compression_index": len(self.compression_selection_log),
            "step": self.step,
            "scorer": result.get("scorer"),
            "n_candidates": len(candidates),
            "best_index": best,
            "scores": result.get("scores"),
            "reference_plan": result.get("reference_plan"),
            "candidates": cand_log,
        })
        self.history_optimizer.add_to_history(
            "",
            self.history_optimizer.convert_llm_history_to_text(history_for_summarization),
            chosen,
            {
                "strategy": "compression_selection",
                "scorer": result.get("scorer"),
                "n_candidates": len(candidates),
                "best_index": best,
                "scores": result.get("scores"),
                "compression_budget": self.compression_budget,
            },
        )
        return chosen

    def current_history_index(self) -> int:
        """Get the index of the current conversation session."""
        return len(self.llm_history) - 1
    
    def current_session_length(self) -> int:
        """Get the number of messages in the current conversation session."""
        if not self.llm_history:
            return 0
        return len(self.llm_history[self.current_history_index()])

    def start_new_session(self) -> None:
        """Start a new conversation session (e.g., after history summarization)."""
        if self.llm_history and len(self.llm_history[-1]) > 0:
            self.llm_history.append([])

    def add_system_prompt(self, system_prompt: str, new_session: bool = False) -> None:
        self.system_prompt = system_prompt
        if len(self.llm_history) == 0 or len(self.llm_history[-1]) == 0:
            self.llm_history[-1].append({
                "role": "system",
                "content": system_prompt
            })
            if not new_session:
                self.entire_session.append({
                    "role": "system",
                    "content": system_prompt
                })

    def add_user_prompt(self, user_prompt: str, new_session: bool = False) -> None:
        """
        Add a user prompt to the current conversation session.
        
        Args:
            user_prompt: The user input to add
        """
        current_session = self.llm_history[self.current_history_index()]
        if len(current_session) == 1 and len(self.llm_history) == 1:
            # first user prompt is important as it includes the important instruction and task
            self.first_user_prompt = user_prompt
        current_session.append({
            "role": "user",
            "content": user_prompt
        })
        if not new_session:
            self.entire_session.append({    
                "role": "user",
                "content": user_prompt
            })

    def add_assistant_response(self, assistant_response: str, new_session: bool = False) -> None:
        """
        Add an assistant response to the current conversation session.
        
        Args:
            assistant_response: The response generated by the LLM
        """
        current_session = self.llm_history[self.current_history_index()]
        current_session.append({
            "role": "assistant",
            "content": assistant_response
        })
        self.step_alignment.append([
            self.current_history_index(), 
            len(current_session) - 1  # Index of the assistant's response
        ])
        self.step += 1  # Increment the step count
        if not new_session:
            self.entire_session.append({    
                "role": "assistant",
                "content": assistant_response
            })

    def get_current_session(self) -> List[Dict[str, str]]:
        """Get the current conversation session for LLM API calls."""
        if not self.llm_history:
            return []
        return self.llm_history[self.current_history_index()]

    def get_conversation_history(self, exclude_system: bool = True) -> List[Dict[str, str]]:
        """
        Get the conversation history for LLM forwarding.
        
        Args:
            exclude_system: Whether to exclude the system message from the history
            
        Returns:
            List of messages in the current session
        """
        current_session = self.get_current_session()
        if exclude_system and current_session and current_session[0].get("role") == "system":
            return current_session[1:]
        return current_session

    def get_all_sessions(self) -> List[List[Dict[str, str]]]:
        """Get all conversation sessions."""
        return self.llm_history

    def clear_current_session(self) -> None:
        """Clear the current conversation session."""
        if self.llm_history:
            self.llm_history[self.current_history_index()] = []

    def _count_tokens(self, text: str) -> int:
        """Token count using the history optimizer's tokenizer when available."""
        if self.history_optimizer is not None:
            try:
                return self.history_optimizer.count_tokens(text or "")
            except Exception:
                pass
        return max(1, len(text) // 4) if text else 0

    def _snapshot_post_compression(self, strategy: str) -> None:
        """Record the exact session installed right after a compression event.

        Captures the full rebuilt session (system + user prompt + preserved /
        selected turns) with per-message and total token counts so you can see
        what the history was set to and whether it fits the budget.
        """
        new_session = self.get_current_session()
        messages = []
        total_tokens = 0
        for msg in new_session:
            content = msg.get("content", "")
            n_tok = self._count_tokens(content)
            total_tokens += n_tok
            messages.append({
                "role": msg.get("role"),
                "tokens": n_tok,
                "content": content,
            })
        self.post_compression_snapshots.append({
            "compression_index": len(self.post_compression_snapshots),
            "strategy": strategy,
            "step": self.step,
            "compression_budget": self.compression_budget,
            "session_index": self.current_history_index(),
            "num_messages": len(messages),
            "total_tokens": total_tokens,
            "messages": messages,
        })

    def _dump_post_compression_text(self, path: str) -> None:
        """Human-readable dump of every post-compression session."""
        lines = []
        for snap in self.post_compression_snapshots:
            lines.append("=" * 80)
            lines.append(
                f"COMPRESSION #{snap['compression_index']} "
                f"(strategy={snap['strategy']} step={snap['step']} "
                f"budget={snap['compression_budget']} "
                f"total_tokens={snap['total_tokens']} "
                f"messages={snap['num_messages']})"
            )
            lines.append("=" * 80)
            for m in snap["messages"]:
                lines.append(f"── {m['role'].upper()} ── [{m['tokens']} tok]")
                lines.append(m["content"])
                lines.append("")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def dump_history(self, output_dir: str) -> None:
        """
        Save conversation history and alignment data to files.

        Args:
            output_dir: Directory to save the history files
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save LLM conversation history
        with open(f'{output_dir}/llm_history.json', 'w') as f:
            json.dump(self.llm_history, f, indent=2)

        # Save step alignment data
        with open(f'{output_dir}/step_alignment.json', 'w') as f:
            json.dump(self.step_alignment, f, indent=2)

        # Save post-compression session snapshots (what the history was set to
        # immediately after each compression event).
        with open(f'{output_dir}/post_compression_history.json', 'w') as f:
            json.dump(self.post_compression_snapshots, f, indent=2)
        self._dump_post_compression_text(f'{output_dir}/post_compression_history.txt')

        # Best-of-N compression selection log (candidates, plans, judge scores,
        # chosen index) — one entry per compression event when selection is on.
        if self.compression_selection_log:
            with open(f'{output_dir}/compression_selection.json', 'w') as f:
                json.dump(self.compression_selection_log, f, indent=2)

        print(f"History dumped to {output_dir}/")
        print(f"  - llm_history.json: {len(self.llm_history)} sessions")
        print(f"  - step_alignment.json: {len(self.step_alignment)} alignments")
        print(f"  - post_compression_history.json: {len(self.post_compression_snapshots)} compression events")

        if self.obs_optimizer:
            self.obs_optimizer.dump_history(output_dir)
            print(f"  - obs_optimizer_history.json: {len(self.obs_optimizer.history)} sessions")

        if self.history_optimizer:
            self.history_optimizer.dump_history(output_dir)
            print(f"  - history_optimizer_history.json: {len(self.history_optimizer.history)} sessions")

    # Optimizer related functions
    def convert_env_history_to_text(self, history: List[Dict[str, str]]) -> str:
        """
        Convert a list of action-observation pairs into a formatted text string.
        
        Args:
            history: List of (action, observation) tuples
        Returns:
            Formatted history text
        """
        history_text = ''
        for i, item in enumerate(history):
            action = item[0]
            observation = item[1]
            if observation:
                history_text += f' - Step {i}: {json.dumps(action)} -> [{observation}]\n'
        return history_text.strip()
    
    def convert_llm_history_to_text(self, history: List[Dict[str, str]]) -> str:
        """
        Convert a list of LLM messages into a formatted text string.
        
        Args:
            history: List of messages in the format [{'role': 'user', 'content': '...'}, ...]
        Returns:
            Formatted history text
        """
        history_text = ''
        for i, message in enumerate(history):
            # Exclude the first user message
            if i == 1 and message['role'] == 'user':
                continue
            role = message['role']
            content = message['content']
            # add as USER: or ASSISTANT: prefix
            if role == 'user':
                history_text += f'USER:\n{content}\n\n'
            elif role == 'assistant':
                history_text += f'ASSISTANT:\n{content}\n\n'
        return history_text.strip()
    
    def extract_last_k_turns(self, session: List[Dict[str, str]], k: int) -> List[Dict[str, str]]:
        """
        Extract the last k turns from a conversation session.
        A turn is defined as an assistant-user pair, where the last message is user (observation).
        
        Args:
            session: List of messages in the conversation session
            k: Number of turns to extract
            
        Returns:
            List of messages representing the last k turns
        """
        if not session or k <= 0:
            return []
        
        # Find assistant-user pairs starting from the end
        turns = []
        i = len(session) - 1
        turn_count = 0
        
        # Start from the end and work backwards
        while i >= 0 and turn_count < k:
            if session[i]['role'] == 'user':
                # Found a user message (observation), now look for the preceding assistant message
                user_msg = session[i]
                assistant_msg = None
                
                # Look backwards for the assistant message
                j = i - 1
                while j >= 0 and session[j]['role'] != 'assistant':
                    j -= 1
                
                if j >= 0:
                    assistant_msg = session[j]
                    # Insert assistant message first, then user message to maintain order
                    turns.insert(0, assistant_msg)
                    turns.insert(1, user_msg)
                    turn_count += 1
                    
                    # Move to before the assistant message for next iteration
                    i = j - 1
                else:
                    # No corresponding assistant message, just add the user message
                    turns.insert(0, user_msg)
                    turn_count += 1
                    i -= 1
            else:
                i -= 1
        
        return turns

    def optimize_observation(self, task, observation, opt_args: Dict[str, Any]) -> str:
        """
        Optimize an observation to reduce context length.
        
        Args:
            opt_args: Dictionary containing different args for tasks:
            e.g., Officebench:
                - task: The task given to the agent
                - observation: The current observation to optimize
                - history: The history of optimizer's inputs and outputs
                - current_app: The current app being used (optional)
                - available_apps: Dictionary of available apps
            
        Returns:
            Optimized observation string
        """
        if self.obs_optimizer:
            if not self.obs_optimizer.check_summarization_needed(observation):
                return observation
            current_session = self.get_current_session()
            history_text = self.convert_llm_history_to_text(current_session)
            if self.prev_history_summary:
                #### 08/17: addition of previous history summary rather makes observation optimization worse.
                if self.add_history_summary_obs_opt:
                    history_text = "PREVIOUS HISTORY SUMMARY:\n" + self.prev_history_summary + "\n\nLATEST HISTORY:\n" + history_text
            # ? how about summarized history?
            # env_history_text = self.convert_env_history_to_text(env_history)
            return self.obs_optimizer.process(
                task=task,
                observation=observation,
                history=history_text,
                raw_history=current_session,
                opt_args=opt_args,
            )
        else:
            raise ValueError("Observation optimizer is not configured.")

    def optimize_history(self, task, opt_args: Dict[str, Any]) -> str:
        """
        Optimize the history of interactions to reduce context length.
        
        Args:
            opt_args: Dictionary containing different args for tasks:
            e.g., Officebench:
                - task: The task given to the agent
                - observation: The current observation to optimize
                - current_app: The current app being used (optional)
                - available_apps: Dictionary of available apps
            
        Returns:
            Optimized history string
        """
        if self.history_optimizer:
            current_session = self.get_current_session()
            
            # system
            # user (first user prompt, important)
            # assistant1
            # obs1
            # assistant2
            # obs2
            # assistant3 (keep if k=2)
            # obs3 (keep if k=2)
            # assistant4 (keep if k=2)
            # obs4 (keep if k=2)

            # Extract the last k turns + the latest assistant turn to preserve
            preserved_turns = self.extract_last_k_turns(current_session, self.preserve_last_k_turns)
            
            # Remove system message and first user message from preserved turns if they exist
            filtered_preserved_turns = []
            for msg in preserved_turns:
                # Skip system message and first user message (task instructions)
                if msg['role'] == 'system':
                    continue
                if msg['role'] == 'user' and msg['content'] == current_session[1]['content']:
                    continue
                filtered_preserved_turns.append(msg)
            
            preserved_turns = filtered_preserved_turns
            
            # Find the index where preserved turns start in the current session
            preserved_start_idx = len(current_session)
            if preserved_turns:
                # Find where the first preserved message appears in the session
                for i, msg in enumerate(current_session):
                    if (msg['role'] == preserved_turns[0]['role'] and 
                        msg['content'] == preserved_turns[0]['content']):
                        preserved_start_idx = i
                        break

            # Create history text excluding the preserved turns
            if preserved_start_idx > 2: # first two turn should not be preserved
                history_for_summarization = current_session[:preserved_start_idx]
                history_text = self.convert_llm_history_to_text(history_for_summarization) # no system message, no first user message
            else:
                history_for_summarization = []
                history_text = ''

            # Check the size of the history for summarization (after tokenization) to determine if summarization is needed
            if self.baseline_strategy == "none":
                # If there's no history to summarize (all content is in preserved turns), don't summarize
                if not history_text.strip():
                    return
                
                # Include preserved turns + system/first_user in the budget
                # check so we compress when the *total* prompt is growing, not
                # just the older history. This catches cases where the last
                # turn alone (e.g. a giant retrieval observation) is what's
                # pushing the request toward the model's context window.
                full_text_for_check = history_text
                if preserved_turns:
                    full_text_for_check += "\n" + self.convert_llm_history_to_text(preserved_turns)
                if current_session and current_session[0].get("role") == "system":
                    full_text_for_check += "\n" + current_session[0].get("content", "")
                if len(current_session) > 1 and current_session[1].get("role") == "user":
                    full_text_for_check += "\n" + current_session[1].get("content", "")
                if not self.history_optimizer.check_summarization_needed(full_text_for_check, self.prev_history_summary):
                    # If history summarization is not needed, return without processing
                    return
                
                # count the turns
                n_accum_turns = len(current_session) // 2 # each turn has assistant and user (observation)
                if self.history_summary_interval > 0 and n_accum_turns < self.history_summary_interval:
                    # Only summarize at specified intervals
                    print(f"   #### Skipping history summarization at step {n_accum_turns} as interval ({self.history_summary_interval}) not met.")
                    return
                
                optimized_history = None
                if self.compression_selector is not None:
                    # Best-of-N: generate N candidate compressions and install
                    # the one that least perturbs the agent's near-future plan.
                    try:
                        optimized_history = self._run_compression_selection(
                            task=task,
                            history_text=history_text,
                            history_for_summarization=history_for_summarization,
                            current_session=current_session,
                            preserved_turns=preserved_turns,
                        )
                    except Exception as e:
                        logging.getLogger(__name__).warning(
                            "Compression selection failed; falling back to single "
                            "compression: %r", e
                        )
                        optimized_history = None
                if optimized_history is None:
                    optimized_history = self.history_optimizer.process(
                        task=task,
                        history=history_text, # without summary and without preserved turns
                        prev_history_summary=self.prev_history_summary,
                        raw_history=history_for_summarization,
                        # opt_args=opt_args,
                    )
                #### TODO: This results in the accumulation of history summaries.
                # We should not accumulate history summaries, but rather replace the previous one.
                # `user_prompt` text is identical to the inline build this helper
                # replaces (reset -> first_user_prompt, accumulate -> session[1]).
                user_prompt = self._summary_to_user_prompt(optimized_history, current_session)
            elif self.baseline_strategy == "discard":
                # check the size of preserved turns
                print(f"Preserved turns: {len(preserved_turns)}")
                if len(preserved_turns) < self.preserve_last_k_turns * 2:
                    return
                optimized_history = ''
                user_prompt = current_session[1]['content']
            elif self.baseline_strategy in ("fifo", "mask_obs", "mask_action", "random"):
                # Selection-based baselines: no LLM call. Pick / mask turns to
                # fit a token budget; prepend the selection to preserved_turns.
                if len(history_for_summarization) == 0:
                    return
                # Reuse the same threshold check the LLM optimizer uses so we
                # only rebuild the session once the older history grows beyond
                # the threshold.
                if (
                    self.history_optimizer is not None
                    and not self.history_optimizer.check_summarization_needed(history_text)
                ):
                    return
                from productive_agents.ctxopt.selection_strategies import apply_selection_strategy
                # Use the optimizer's tokenizer if available; otherwise approximate.
                if self.history_optimizer is not None:
                    count_tokens = self.history_optimizer.count_tokens
                else:
                    count_tokens = lambda s: max(1, len(s) // 4) if s else 0
                selected = apply_selection_strategy(
                    self.baseline_strategy,
                    history_for_summarization,
                    self.compression_budget,
                    count_tokens,
                    seed=self.random_seed,
                )
                optimized_history = ''
                preserved_turns = selected + preserved_turns
                user_prompt = current_session[1]['content']
                # Logging only: selection baselines make no LLM call, so the
                # history optimizer's history would otherwise be empty. Record
                # the kept turns as the "compression output" so it shows up in
                # history_optimizer_history.json like the LLM compressors.
                if self.history_optimizer is not None:
                    self.history_optimizer.add_to_history(
                        '',
                        self.history_optimizer.convert_llm_history_to_text(
                            history_for_summarization
                        ),
                        self.history_optimizer.convert_llm_history_to_text(selected),
                        {
                            "strategy": self.baseline_strategy,
                            "compression_budget": self.compression_budget,
                            "input_turns": len(history_for_summarization),
                            "selected_turns": len(selected),
                        },
                    )
            elif self.baseline_strategy == "retrieve":
                # Retrieve the most-similar prior turns. When compression_budget
                # is set, keep as many top-similarity pairs as fit in budget;
                # otherwise fall back to a fixed retrieve_turns count.
                history_for_summarization = self.entire_session[2:max(2, len(self.entire_session) - len(preserved_turns))]
                last_turn = self.entire_session[-2:]
                if self.compression_budget:
                    if self.history_optimizer is not None:
                        count_tokens = self.history_optimizer.count_tokens
                    else:
                        count_tokens = lambda s: max(1, len(s) // 4) if s else 0
                    retrieved_turns = self.history_optimizer.retrieve(
                        history_for_summarization,
                        last_turn,
                        budget=self.compression_budget,
                        count_tokens=count_tokens,
                    )
                else:
                    if len(history_for_summarization) < self.retrieve_turns * 2:
                        return
                    retrieved_turns = self.history_optimizer.retrieve(
                        history_for_summarization, last_turn, self.retrieve_turns
                    )

                optimized_history = ''
                preserved_turns = retrieved_turns + preserved_turns
                user_prompt = current_session[1]['content']
            else:
                raise NotImplementedError(f"Unknown baseline strategy: {self.baseline_strategy}")
                
            # 1. start a new session
            # 2. add a system prompt on the top
            # 3. then add the optimized history as user message
            # 4. add the preserved last k turns
            self.start_new_session()
            self.add_system_prompt(self.system_prompt, new_session=True)
            
            # Add the optimized history as user prompt
            self.add_user_prompt(user_prompt, new_session=True)
            
            # Add the preserved last k turns back to the new session
            for msg in preserved_turns:
                if msg['role'] == 'user':
                    self.add_user_prompt(msg['content'], new_session=True)
                elif msg['role'] == 'assistant':
                    self.add_assistant_response(msg['content'], new_session=True)

            self._snapshot_post_compression(self.baseline_strategy)

            self.prev_history_summary = optimized_history
        else:
            raise ValueError("History optimizer is not configured.")