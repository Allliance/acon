# Managing all memory features
import os
import json
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
        
        print(f"History dumped to {output_dir}/")
        print(f"  - llm_history.json: {len(self.llm_history)} sessions")
        print(f"  - step_alignment.json: {len(self.step_alignment)} alignments")

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
                
                if not self.history_optimizer.check_summarization_needed(history_text, self.prev_history_summary):
                    # If history summarization is not needed, return without processing
                    return
                
                # count the turns
                n_accum_turns = len(current_session) // 2 # each turn has assistant and user (observation)
                if self.history_summary_interval > 0 and n_accum_turns < self.history_summary_interval:
                    # Only summarize at specified intervals
                    print(f"   #### Skipping history summarization at step {n_accum_turns} as interval ({self.history_summary_interval}) not met.")
                    return
                
                optimized_history = self.history_optimizer.process(
                    task=task,
                    history=history_text, # without summary and without preserved turns
                    prev_history_summary=self.prev_history_summary,
                    raw_history=history_for_summarization,
                    # opt_args=opt_args,
                )
                #### TODO: This results in the accumulation of history summaries.
                # We should not accumulate history summaries, but rather replace the previous one.
                if self.history_summary_rule == "reset":
                    # Reset the previous history summary after optimization
                    user_prompt = self.first_user_prompt + \
                        "\n\n<HISTORY_SUMMARY>\n" + \
                        optimized_history + \
                        "\n</HISTORY_SUMMARY>"
                elif self.history_summary_rule == "accumulate":
                    user_prompt = current_session[1]['content'] + \
                        "\n\n<HISTORY_SUMMARY>\n" + \
                        optimized_history + \
                        "\n</HISTORY_SUMMARY>"
                else:
                    raise NotImplementedError(f"Unknown history summary rule: {self.history_summary_rule}")
            elif self.baseline_strategy == "discard":
                # check the size of preserved turns
                print(f"Preserved turns: {len(preserved_turns)}")
                if len(preserved_turns) < self.preserve_last_k_turns * 2:
                    return
                optimized_history = ''
                user_prompt = current_session[1]['content']
            elif self.baseline_strategy == "retrieve":
                # check the size of preserved turns
                # retrieve k turns from the current session
                # only works there are more sessions than k
                history_for_summarization = self.entire_session[2:max(2, len(self.entire_session) - len(preserved_turns))]
                if len(history_for_summarization) < self.retrieve_turns * 2:
                    return
                last_turn = self.entire_session[-2:]
                retrieved_turns = self.history_optimizer.retrieve(history_for_summarization, last_turn, self.retrieve_turns)

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
            
            self.prev_history_summary = optimized_history
        else:
            raise ValueError("History optimizer is not configured.")