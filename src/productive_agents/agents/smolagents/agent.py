"""
Minimal Smolagents Agent

This agent uses the unified agent framework to interact with the SmolagentsEnv.
It builds simple prompts and extracts Python code from LLM responses.
"""

from typing import Any, Dict, List, Optional
import re
from jinja2 import Template

from productive_agents.agents.unified_agent import (
	UnifiedAgent,
	UnifiedPromptBuilder,
	UnifiedActionProcessor,
)


DEFAULT_SYSTEM_PROMPT = """You are an expert assistant who can solve any task using code blobs. You will be given a task to solve as best you can.
To do so, you have been given access to a list of tools: these tools are basically Python functions which you can call with code.
To solve the task, you must plan forward to proceed in a series of steps, in a cycle of Thought, Code, and Observation sequences.

At each step, in the 'Thought:' sequence, you should first explain your reasoning towards solving the task and the tools that you want to use.
Then in the Code sequence you should write the code in simple Python. The code sequence must be opened with '{{code_block_opening_tag}}', and closed with '{{code_block_closing_tag}}'.
During each intermediate step, you can use 'print()' to save whatever important information you will then need.
These print outputs will then appear in the 'Observation:' field, which will be available as input for the next step.
In the end you have to return a final answer using the `final_answer` tool.

Here are a few examples using notional tools:
---
Task: "Generate an image of the oldest person in this document."

Thought: I will proceed step by step and use the following tools: `document_qa` to find the oldest person in the document, then `image_generator` to generate an image according to the answer.
{{code_block_opening_tag}}
answer = document_qa(document=document, question="Who is the oldest person mentioned?")
print(answer)
{{code_block_closing_tag}}
Observation: "The oldest person in the document is John Doe, a 55 year old lumberjack living in Newfoundland."

Thought: I will now generate an image showcasing the oldest person.
{{code_block_opening_tag}}
image = image_generator("A portrait of John Doe, a 55-year-old man living in Canada.")
final_answer(image)
{{code_block_closing_tag}}

---
Task: "What is the result of the following operation: 5 + 3 + 1294.678?"

Thought: I will use python code to compute the result of the operation and then return the final answer using the `final_answer` tool
{{code_block_opening_tag}}
result = 5 + 3 + 1294.678
final_answer(result)
{{code_block_closing_tag}}

---
Task:
"Answer the question in the variable `question` about the image stored in the variable `image`. The question is in French.
You have been provided with these additional arguments, that you can access using the keys as variables in your python code:
{'question': 'Quel est l'animal sur l'image?', 'image': 'path/to/image.jpg'}"

Thought: I will use the following tools: `translator` to translate the question into English and then `image_qa` to answer the question on the input image.
{{code_block_opening_tag}}
translated_question = translator(question=question, src_lang="French", tgt_lang="English")
print(f"The translated question is {translated_question}.")
answer = image_qa(image=image, question=translated_question)
final_answer(f"The answer is {answer}")
{{code_block_closing_tag}}

---
Task:
In a 1979 interview, Stanislaus Ulam discusses with Martin Sherwin about other great physicists of his time, including Oppenheimer.
What does he say was the consequence of Einstein learning too much math on his creativity, in one word?

Thought: I need to find and read the 1979 interview of Stanislaus Ulam with Martin Sherwin.
{{code_block_opening_tag}}
pages = web_search(query="1979 interview Stanislaus Ulam Martin Sherwin physicists Einstein")
print(pages)
{{code_block_closing_tag}}
Observation:
No result found for query "1979 interview Stanislaus Ulam Martin Sherwin physicists Einstein".

Thought: The query was maybe too restrictive and did not find any results. Let's try again with a broader query.
{{code_block_opening_tag}}
pages = web_search(query="1979 interview Stanislaus Ulam")
print(pages)
{{code_block_closing_tag}}
Observation:
Found 6 pages:
[Stanislaus Ulam 1979 interview](https://ahf.nuclearmuseum.org/voices/oral-histories/stanislaus-ulams-interview-1979/)

[Ulam discusses Manhattan Project](https://ahf.nuclearmuseum.org/manhattan-project/ulam-manhattan-project/)

(truncated)

Thought: I will read the first 2 pages to know more.
{{code_block_opening_tag}}
for url in ["https://ahf.nuclearmuseum.org/voices/oral-histories/stanislaus-ulams-interview-1979/", "https://ahf.nuclearmuseum.org/manhattan-project/ulam-manhattan-project/"]:
    whole_page = visit_webpage(url)
    print(whole_page)
    print("\n" + "="*80 + "\n")  # Print separator between pages
{{code_block_closing_tag}}
Observation:
Manhattan Project Locations:
Los Alamos, NM
Stanislaus Ulam was a Polish-American mathematician. He worked on the Manhattan Project at Los Alamos and later helped design the hydrogen bomb. In this interview, he discusses his work at
(truncated)

Thought: I now have the final answer: from the webpages visited, Stanislaus Ulam says of Einstein: "He learned too much mathematics and sort of diminished, it seems to me personally, it seems to me his purely physics creativity." Let's answer in one word.
{{code_block_opening_tag}}
final_answer("diminished")
{{code_block_closing_tag}}

---
Task: "Which city has the highest population: Guangzhou or Shanghai?"

Thought: I need to get the populations for both cities and compare them: I will use the tool `web_search` to get the population of both cities.
{{code_block_opening_tag}}
for city in ["Guangzhou", "Shanghai"]:
    print(f"Population {city}:", web_search(f"{city} population")
{{code_block_closing_tag}}
Observation:
Population Guangzhou: ['Guangzhou has a population of 15 million inhabitants as of 2021.']
Population Shanghai: '26 million (2019)'

Thought: Now I know that Shanghai has the highest population.
{{code_block_opening_tag}}
final_answer("Shanghai")
{{code_block_closing_tag}}

---
Task: "What is the current age of the pope, raised to the power 0.36?"

Thought: I will use the tool `wikipedia_search` to get the age of the pope, and confirm that with a web search.
{{code_block_opening_tag}}
pope_age_wiki = wikipedia_search(query="current pope age")
print("Pope age as per wikipedia:", pope_age_wiki)
pope_age_search = web_search(query="current pope age")
print("Pope age as per google search:", pope_age_search)
{{code_block_closing_tag}}
Observation:
Pope age: "The pope Francis is currently 88 years old."

Thought: I know that the pope is 88 years old. Let's compute the result using python code.
{{code_block_opening_tag}}
pope_current_age = 88 ** 0.36
final_answer(pope_current_age)
{{code_block_closing_tag}}

Above example were using notional tools that might not exist for you. On top of performing computations in the Python code snippets that you create, you only have access to these tools, behaving like regular python functions:
{{code_block_opening_tag}}
{%- for tool in tools.values() %}
{{ tool.to_code_prompt() }}
{% endfor %}
{{code_block_closing_tag}}

Here are the rules you should always follow to solve your task:
1. Always provide a 'Thought:' sequence, and a '{{code_block_opening_tag}}' sequence ending with '{{code_block_closing_tag}}', else you will fail.
2. Use only variables that you have defined!
3. Always use the right arguments for the tools. DO NOT pass the arguments as a dict as in 'answer = wikipedia_search({'query': "What is the place where James Bond lives?"})', but use the arguments directly as in 'answer = wikipedia_search(query="What is the place where James Bond lives?")'.
4. Take care to not chain too many sequential tool calls in the same code block, especially when the output format is unpredictable. For instance, a call to wikipedia_search has an unpredictable return format, so do not have another tool call that depends on its output in the same block: rather output results with print() to use them in the next block.
5. Call a tool only when needed, and never re-do a tool call that you previously did with the exact same parameters.
6. Don't name any new variable with the same name as a tool: for instance don't name a variable 'final_answer'.
7. Never create any notional variables in our code, as having these in your logs will derail you from the true variables.
8. You can use imports in your code, but only from the following list of modules: {{authorized_imports}}
9. The state persists between code executions: so if in one step you've created variables or imported modules, these will all persist.
10. Don't give up! You're in charge of solving the task, not providing directions to solve it.

Now Begin!"""

USER_PROMPT = f"""You will answer multiple complex questions using iterative reasoning and search. The task consists of multiple semi-colon-seperated questions, and you need to give answers of all questions in the same sequence at the end.
Task:
{{task}}

IMPORTANT: 
1. Provide the final answers—separated by semicolons—within answer1; answer2; ... . The answers must be concise, contain only essential words, and avoid any explanations
2. Do not search multiple questions simultaneously in one action; sequentially find the answer for each question.
3. Do not use the general knowledge you have been trained on to answer the questions; use only the tools and information from tools provided."""

DEFAULT_PROMPT_DICT = {
	"system_message": DEFAULT_SYSTEM_PROMPT
}


class SmolagentsPromptBuilder(UnifiedPromptBuilder):
	"""Prompt builder for Smolagents tasks."""

	def __init__(self, prompt_dict: Dict[str, str], env, working_dir: str = "."):
		super().__init__(prompt_dict, working_dir)
		self.env = env

	def build_system_message(self, config: Dict[str, Any], available_apps: Dict[str, Any]) -> str:
		"""Render the system message template with environment-specific values."""
		tmpl_text = self.prompt_dict.get("system_message", "")
		if not tmpl_text:
			return "You are an AI assistant that helps complete tasks."

		# Provide values expected by the template
		code_block_opening_tag = "```python"
		code_block_closing_tag = "```"
		tools = getattr(self.env, "tools", {}) or {}
		authorized_imports = getattr(self.env, "additional_authorized_imports", []) or []

		try:
			return Template(tmpl_text).render(
				code_block_opening_tag=code_block_opening_tag,
				code_block_closing_tag=code_block_closing_tag,
				tools=tools,
				authorized_imports=authorized_imports,
			)
		except Exception:
			# Fallback to raw template if rendering fails
			return tmpl_text

	def build_prompt(self, env, context_sections: List[str]) -> str:
		context_prefix = "".join(context_sections)

		# Build the first user prompt with task details
		if getattr(env, "trajectory", None) and len(env.trajectory) > 0:
			# Continue with latest observation
			return context_prefix + (env.observation or "")

		# Initial instruction
		if hasattr(env, "task") and env.task is not None:
			if isinstance(env.task, str):
				task_instruction = env.task
			else:
				task_instruction = getattr(env.task, "instruction", str(env.task))
		else:
			task_instruction = "No task specified"

		main_prompt = USER_PROMPT.format(
			task=task_instruction,
		)
		return context_prefix + main_prompt


class SmolagentsActionProcessor(UnifiedActionProcessor):
	"""Extracts Python code from LLM responses for Smolagents."""

	def extract_action(self, response: str) -> str:
		# Prefer fenced python code blocks
		patterns = [
			r"```python\s*(.*?)\s*```",
			r"```\s*(.*?)\s*```",
		]
		for pattern in patterns:
			m = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
			if m:
				code = m.group(1).strip()
				if code:
					return code
		# Fallback: return raw response trimmed
		return response.strip()


class SmolagentsAgent(UnifiedAgent):
	"""Minimal agent for the Smolagents environment."""
	def __init__(self, model_name: str, key: str, env, task_config: Dict[str, Any], **kwargs):
		super().__init__(model_name=model_name, key=key, env=env, task_config=task_config, **kwargs)
		self.stop_sequences = ["Observation:", "Calling tools:"]
		
	def _create_prompt_builder(self) -> SmolagentsPromptBuilder:
		return SmolagentsPromptBuilder(DEFAULT_PROMPT_DICT.copy(), env=self.env)

	def _create_action_processor(self) -> SmolagentsActionProcessor:
		return SmolagentsActionProcessor(self.logger)

	def _process_response(self, response: str) -> str:
		return self.action_processor.extract_action(response)

	def _determine_success(self, env, reward: float, info: Dict) -> bool:
		if hasattr(env, "task_completed") and callable(env.task_completed):
			return bool(env.task_completed())
		return info.get("success", False) or reward > 0


def create_smolagents_agent(
	model_name: str,
	key: str,
	env,
	task_config: Dict[str, Any],
	**kwargs,
) -> SmolagentsAgent:
	"""Factory to create a SmolagentsAgent."""
	return SmolagentsAgent(
		model_name=model_name,
		key=key,
		env=env,
		task_config=task_config,
		**kwargs,
	)

