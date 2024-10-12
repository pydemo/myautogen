from typing import List
from pprint import pprint as pp
from autogen.io.base import IOStream
from autogen.agentchat.chat import ChatResult
from autogen.cache import Cache
from autogen.oai.openai_utils import  get_key
from autogen.logger.logger_utils import get_current_ts

from autogen import apc

# Set verbose to True
apc.verbose = True

apc.depth=0
apc.call_id=0
apc.tree={'calling':{ 'name': 'root','calling':{}, 'depth'  : 0}}

def track(func):
    def wrapper(*args, **kwargs):
       
        
        class_name = args[0].__class__.__name__
        method_name = func.__name__        
        branch=apc.tree['calling']
        apc.depth += 1
        apc.call_id +=1
        branch['calling'][apc.call_id]={'name': f'{class_name}.{method_name}','depth':apc.depth,'calling':{},'caller':apc.depth-1}

        
        print("Before the function runs.", apc.depth, class_name, method_name)

        print(f"Method '{class_name}.{method_name}' is about to be called.")
        
       
        result = func(*args, **kwargs)
        print(f"Method '{class_name}.{method_name}' has finished execution.")
        apc.depth -= 1
        return result
    return wrapper


iostream = IOStream.get_default()
OAI_PRICE1K = {
    # https://openai.com/api/pricing/
    # gpt-4o
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-2024-05-13": (0.005, 0.015),
    "gpt-4o-2024-08-06": (0.0025, 0.01),
    # gpt-4-turbo
    "gpt-4-turbo-2024-04-09": (0.01, 0.03),
    # gpt-4
    "gpt-4": (0.03, 0.06),
    "gpt-4-32k": (0.06, 0.12),
    # gpt-4o-mini
    "gpt-4o-mini": (0.000150, 0.000600),
    "gpt-4o-mini-2024-07-18": (0.000150, 0.000600),
    # gpt-3.5 turbo
    "gpt-3.5-turbo": (0.0005, 0.0015),  # default is 0125
    "gpt-3.5-turbo-0125": (0.0005, 0.0015),  # 16k
    "gpt-3.5-turbo-instruct": (0.0015, 0.002),
    # base model
    "davinci-002": 0.002,
    "babbage-002": 0.0004,
    # old model
    "gpt-4-0125-preview": (0.01, 0.03),
    "gpt-4-1106-preview": (0.01, 0.03),
    "gpt-4-1106-vision-preview": (0.01, 0.03),  # TODO: support vision pricing of images
    "gpt-3.5-turbo-1106": (0.001, 0.002),
    "gpt-3.5-turbo-0613": (0.0015, 0.002),
    # "gpt-3.5-turbo-16k": (0.003, 0.004),
    "gpt-3.5-turbo-16k-0613": (0.003, 0.004),
    "gpt-3.5-turbo-0301": (0.0015, 0.002),
    "text-ada-001": 0.0004,
    "text-babbage-001": 0.0005,
    "text-curie-001": 0.002,
    "code-cushman-001": 0.024,
    "code-davinci-002": 0.1,
    "text-davinci-002": 0.02,
    "text-davinci-003": 0.02,
    "gpt-4-0314": (0.03, 0.06),  # deprecate in Sep
    "gpt-4-32k-0314": (0.06, 0.12),  # deprecate in Sep
    "gpt-4-0613": (0.03, 0.06),
    "gpt-4-32k-0613": (0.06, 0.12),
    "gpt-4-turbo-preview": (0.01, 0.03),
    # https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/#pricing
    "gpt-35-turbo": (0.0005, 0.0015),  # what's the default? using 0125 here.
    "gpt-35-turbo-0125": (0.0005, 0.0015),
    "gpt-35-turbo-instruct": (0.0015, 0.002),
    "gpt-35-turbo-1106": (0.001, 0.002),
    "gpt-35-turbo-0613": (0.0015, 0.002),
    "gpt-35-turbo-0301": (0.0015, 0.002),
    "gpt-35-turbo-16k": (0.003, 0.004),
    "gpt-35-turbo-16k-0613": (0.003, 0.004),
}
llm_config={"model": "gpt-4o-mini"}


# ## The task!

# In[ ]:


task = "Write a blogpost about the stock price performance of "\
"Nvidia in the past month. Today's date is 2024-04-23."


# ## Build a group chat
# 
# This group chat will include these agents:
# 
# 1. **User_proxy** or **Admin**: to allow the user to comment on the report and ask the writer to refine it.
# 2. **Planner**: to determine relevant information needed to complete the task.
# 3. **Engineer**: to write code using the defined plan by the planner.
# 4. **Executor**: to execute the code written by the engineer.
# 5. **Writer**: to write the report.

# In[ ]:

from openai import OpenAI 
oai_client_1 = OpenAI()
class OpenAIClient:
    """Follows the Client protocol and wraps the OpenAI client."""

    def __init__(self, client):
        self._oai_client = client

    def message_retrieval(
        self, response
    ) :
        """Retrieve the messages from the response."""
        choices = response.choices
        if isinstance(response):
            return [choice.text for choice in choices]  # type: ignore [union-attr]

    @track
    def create(self, params) :
        """Create a completion for a given config using openai's client.

        Args:
            client: The openai client.
            params: The params for the completion.

        Returns:
            The completion.
        """
     
        print('client:create(params)')
        pp(params)
        #cache_client=True
        LEGACY_DEFAULT_CACHE_SEED = 41
        LEGACY_CACHE_DIR = ".cache"
        OPEN_API_BASE_URL_PREFIX = "https://api.openai.com"

        cache_client = Cache.disk(LEGACY_DEFAULT_CACHE_SEED, LEGACY_CACHE_DIR)
        if 0:
            completions = (
                self._oai_client.chat.completions if "messages" in params else self._oai_client.completions
            )  # type: ignore [attr-defined]
        # If streaming is enabled and has messages, then iterate over the chunks of the response.
        # If streaming is enabled and has messages, then iterate over the chunks of the response.
        with cache_client as cache:
            # Try to get the response from cache
            key = get_key(params)
            #pp(key)
            #e()
            request_ts = get_current_ts()

            response = cache.get(key, None)
            #print('response:',response)
            #e()

            if response is not None:
                # If the response is found in the cache, return it
                #pp(response)
                #print(response.choices[0].message.content)
                #e()
                return response.choices[0].message.content
            
        if params.get("stream", False) and "messages" in params:
            response_contents = [""] * params.get("n", 1)
            finish_reasons = [""] * params.get("n", 1)
            completion_tokens = 0



            # Send the chat completion request to OpenAI's API and process the response in chunks
            for chunk in oai_client_1.chat.completions.create(**params):
                if chunk.choices:
                    for choice in chunk.choices:
                        content = choice.delta.content
                        tool_calls_chunks = choice.delta.tool_calls
                        finish_reasons[choice.index] = choice.finish_reason


                        # End handle tool calls

                        # If content is present, print it to the terminal and update response variables
                        if content is not None:
                            print(content, end="", flush=True)
                            response_contents[choice.index] += content
                            completion_tokens += 1
                        else:
                            # iostream.print()
                            pass


            response = response_contents
                
        else:
            print('no streaming')
            # If streaming is not enabled, send a regular chat completion request
            params = params.copy()
            params["stream"] = False
            response = oai_client_1.chat.completions.create(**params)
            #pp(response)
            response=response.choices[0].message.content

        if cache_client is not None:
            # Cache the response
            #key = get_key(params)
            with cache_client as cache:
                cache.set(key, response)    
        return response

    def cost(self, response) :
        """Calculate the cost of the response."""
        model = response.model
        if model not in OAI_PRICE1K:
            raise ValueError(f"Model {model} not found in the price list.")

        n_input_tokens = response.usage.prompt_tokens if response.usage is not None else 0  # type: ignore [union-attr]
        n_output_tokens = response.usage.completion_tokens if response.usage is not None else 0  # type: ignore [union-attr]
        if n_output_tokens is None:
            n_output_tokens = 0
        tmp_price1K = OAI_PRICE1K[model]
        # First value is input token rate, second value is output token rate
        if isinstance(tmp_price1K, tuple):
            return (tmp_price1K[0] * n_input_tokens + tmp_price1K[1] * n_output_tokens) / 1000  # type: ignore [no-any-return]
        return tmp_price1K * (n_input_tokens + n_output_tokens) / 1000  # type: ignore [operator]

    @staticmethod
    def get_usage(response) :
        return {
            "prompt_tokens": response.usage.prompt_tokens if response.usage is not None else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage is not None else 0,
            "total_tokens": response.usage.total_tokens if response.usage is not None else 0,
            "cost": response.cost if hasattr(response, "cost") else 0,
            "model": response.model,
        }



# Setup
llm_config = {"model": "gpt-4o-mini"}

# Define Agent Classes

class UndefinedNextAgent(Exception):
    """Exception raised when the provided next agents list does not overlap with agents in the group."""

    def __init__(self, message: str = "The provided agents list does not overlap with agents in the group."):
        self.message = message
        super().__init__(self.message)

class NoEligibleSpeaker(Exception):
    """Exception raised for early termination of a GroupChat."""

    def __init__(self, message: str = "No eligible speakers."):
        self.message = message
        super().__init__(self.message)

class ConversableAgent:
    def __init__(self, name, system_message= "You are a helpful AI Assistant.", description=None, llm_config={}, code_execution_config=None, 
                 human_input_mode="ALWAYS", default_auto_reply="", chat_messages=None):
        print('CA: __init__', name) 
        self.name = name
        self.system_message = system_message  # Detailed task-specific instructions
        self.description = description  # High-level role description
        self.llm_config = llm_config
        self.code_execution_config = code_execution_config
        self.human_input_mode = human_input_mode
        self.client=OpenAIClient(oai_client_1)
    @track
    def generate_oai_reply(
        self,
        messages,
        sender,
        config,
    ):
        """Generate a reply using autogen.oai."""
        client =self.client
        pp(messages)
        params={'messages': messages,
        'model': 'gpt-4o-mini'}
        print ('in _generate_oai_reply_from_client: context:',messages[-1].pop("context", None))
        print ('in _generate_oai_reply_from_client: messages:',messages)        
        extracted_response = response = client.create(params)
        print ('in _generate_oai_reply_from_client: response:',extracted_response[:-30])  
        pp(extracted_response)
        #e(555)
        return (False, None) if extracted_response is None else (True, extracted_response)
    
    @track
    def generate_reply(
            self,
            messages=None,
            sender=None,
            **kwargs,
        ) :
        """Reply based on the conversation history and the sender.

    
        """
        #groupchat=[]
        print("in generate_reply:", messages, sender.name)
        #pp(dir(sender))
        print(f"GroupChatManager:generate_reply:    SENDER.NAME {sender.name}")
        print(f"GroupChatManager:generate_reply:    MESSAGES {messages}")
        final, reply = self.generate_oai_reply( messages=messages, sender=sender, config=None)
        print('generate_reply: post reply_func:', self.generate_oai_reply)
        print('generate_reply: post reply:',final, reply[:50])
        if final:
            return reply
        raise Exception("No reply generated")
    def receive(
        self,
        message,
        sender,
        request_reply,
        silent,
    ):
        """Receive a message from another agent.

        Once a message is received, this function sends a reply to the sender or stop.
        The reply can be generated automatically or entered manually by a human.

        """
        print(f'&&&& in receive:{self.name}<<<-Sender: {sender.name}' )
        if not request_reply:
            print('777, returning before generating reply')
            return
        # Write a blogpost about the stock price performance of Nvidia in the past month. Today's date is 2024-04-23. 
        # #<autogen.agentchat.groupchat.GroupChatManager object at 0x00000165DA303F70> 
        print('receive:',message, self, request_reply, silent)
        reply = self.generate_reply(messages=message, sender=sender)
        print('receive :post generate_reply:',reply)
        if reply is not None:
            print('receive :sending reply:',reply)
            self.send(reply, sender, silent=True)
            print('receive :post sent reply:',reply)          

    def _send_message(self, message):
        return f"{self.name} sends: {message}"

    def send(
        self,
        message,
        recipient,
        request_reply=False,
        silent=False,
    ):
        """Send a message to another agent.

        """
        print ('send start: request_reply', request_reply, message[:50], self.name,'->>',recipient.name)
        if 1:
            print('send:',message[:50], self, request_reply, silent)
            print('send recipient:',recipient.name)
            recipient.receive(message, self, request_reply, silent)
        

    @track 
    def initiate_chat(self, recipient, message,max_turns=None):
        # Allows user_proxy to initiate the chat
        print(f'ConversableAgent:{self.name}->initiate_chat')
        if isinstance(max_turns, int):
            for _ in range(max_turns):
                if _ == 0:
                    msg2send=[{'content': 'You are in a role play game. The following roles '
                          'are available:\n'
                          '                Engineer: An engineer that writes '
                          'code based on the plan provided by the planner.\n'
                          'Writer: Writer.Write blogs based on the code '
                          'execution results and take feedback from the admin '
                          'to refine the blog.\n'
                          'Executor: Execute the code written by the engineer '
                          'and report the result.\n'
                          'Planner: Planner. Given a task, determine what '
                          'information is needed to complete the task. After '
                          'each step is done by others, check the progress and '
                          'instruct the remaining steps.\n'
                          '                Read the following conversation.\n'
                          '                Then select the next role from '
                          "['Engineer', 'Writer', 'Executor', 'Planner'] to "
                          'play. Only return the role.',
               'role': 'system'},
              {'content': 'Write a blogpost about the stock price performance '
                          "of Nvidia in the past month. Today's date is "
                          '2024-04-23.',
               'name': 'Admin',
               'role': 'user'},
              {'content': 'Read the above conversation. Then select the next '
                          "role from ['Engineer', 'Writer', 'Executor', "
                          "'Planner'] to play. Only return the role.",
               'name': 'checking_agent',
               'role': 'system'}]
                else:
                    print('CA initiate_chat: generate_reply: start--------------------')
                    msg=[{'content': 'Read the above conversation. Then select the next role from '
                                        "['Engineer', 'Writer', 'Executor', 'Planner'] to play. Only "
                                        'return the role.',
                            'name': 'checking_agent',
                            'role': 'system'},
                            {'content': 'Writer', 'name': 'speaker_selection_agent', 'role': 'user'}]
                    if 0:
                        msg2send = self.generate_reply(messages=msg, sender=recipient)
                        
                        e(333)
                    else:
                        #e(647475)
                        print('''FAKE: generate_reply: reply_func: <function GroupChat._auto_select_speaker.<locals>.validate_speaker_name at 0x00000235E09BE4D0>
FAKE: generate_reply: trigger: <class 'list'>
FAKE: generate_reply: config: None
FAKE: generate_reply: post reply_func: <function GroupChat._auto_select_speaker.<locals>.validate_speaker_name at 0x00000235E09BE4D0>
FAKE: generate_reply: post reply: True None''')                        
                        msg2send=None
                    print('CA initiate_chat: post generate_reply: ---------------------',msg2send)
                if msg2send is None:
                    
                    break
                print('CA initiate_chat: self.send:',_, msg2send)  
                self.send(msg2send, recipient, request_reply=True, silent=True)
                print('///////////// CA initiate_chat: post self.send:',_, msg2send)


        else:
            print('NO LOOP: CA initiate_chat: self.send:',message)
            self.send(message, recipient, request_reply=True, silent=False)

        
        chat_result= {'chat_history':[{'content': "Read the above conversation. Then select the next role from ['Engineer', 'Writer', 'Executor', 'Planner'] to play. Only return the role.", 'name': 'checking_agent', 'role': 'system'},
                                     {'content': 'Writer', 'role': 'user', 'name': 'speaker_selection_agent'}, 
                                     {'role': 'user', 'content': '[AGENT SELECTED]Writer'}], 
                      'summary':'Writer'
        }

        return chat_result


class AssistantAgent(ConversableAgent):
    def __init__(self, name, llm_config, description):
        super().__init__(name, system_message="", description=description, llm_config=llm_config)
    
    def write_code(self, task):
        # Simulate writing code based on the task
        return f"{self.name} writes code for task: {task}"


class GroupChat:
    def __init__(self, agents, messages, max_round, allowed_or_disallowed_speaker_transitions, speaker_transitions_type="allowed"):
        self.agents = agents #{agent.: agent for agent in agents}
        self.messages = messages
        self.max_round = max_round
        self.allowed_or_disallowed_speaker_transitions = allowed_or_disallowed_speaker_transitions
        self.speaker_transitions_type = speaker_transitions_type
        self.round_counter = 0
        self.speaker_selection_method = "random"
        self.max_retries_for_selecting_speaker = 3
    @property
    def agent_names(self) -> List[str]:
        """Return the names of the agents in the group chat."""
        return [agent.name for agent in self.agents]
    def _initiate_chat(self, task):
        self.messages.append(task)
        return f"Chat initiated with task: {task}"

    def process_round(self):
        # Simulate processing of chat messages
        for agent_name in self.agents:
            agent = self.agents[agent_name]
            print(agent.receive_message("New task"))
            print(agent.send_message("Processing message"))
        self.round_counter += 1
    def select_speaker(self, last_speaker, selector) :
        """Select the next speaker (with requery)."""

        # Prepare the list of available agents and select an agent if selection method allows (non-auto)

        if self.speaker_selection_method == "manual":
            # An agent has not been selected while in manual mode, so move to the next agent
            return self.next_agent(last_speaker)

        # auto speaker selection with 2-agent chat
        messages = [{'content': "Write 111  a blogpost about the stock price performance of Nvidia in the past month. Today's date is 2024-04-23.",
                      'role': 'user', 'name': 'Admin'}]
        agentsnames =[agent.name for agent in self.agents if agent!=last_speaker]
        print(agentsnames)
        agents =[agent for agent in self.agents if agent!=last_speaker]
        return self._auto_select_speaker(last_speaker, selector, messages, agents)
    def append(self, message, speaker):
        """Append a message to the group chat.
        We cast the content to str here so that it can be managed by text-based
        model.
        """
        # set the name to speaker's name if the role is not function
        # if the role is tool, it is OK to modify the name
        if message["role"] != "function":
            message["name"] = speaker.name
        message["content"] = message["content"]
        self.messages.append(message)    
    def next_agent(self, agent, agents= None):
        """Return the next agent in the list."""
        if agents is None:
            agents = self.agents

        # Ensure the provided list of agents is a subset of self.agents
        if not set(agents).issubset(set(self.agents)):
            raise UndefinedNextAgent()

        # What index is the agent? (-1 if not present)
        idx = self.agent_names.index(agent.name) if agent.name in self.agent_names else -1

        # Return the next agent
        if agents == self.agents:
            return agents[(idx + 1) % len(agents)]
        else:
            offset = idx + 1
            for i in range(len(self.agents)):
                if self.agents[(offset + i) % len(self.agents)] in agents:
                    return self.agents[(offset + i) % len(self.agents)]

        # Explicitly handle cases where no valid next agent exists in the provided subset.
        raise UndefinedNextAgent()

    @track
    def _auto_select_speaker(
        self,
        last_speaker,
        selector,
        messages,
        agents,
    ) :
        """Selects next speaker for the "auto" speaker selection method. Utilises its own two-agent chat to determine the next speaker and supports requerying.

        Speaker selection for "auto" speaker selection method:
        1. Create a two-agent chat with a speaker selector agent and a speaker validator agent, like a nested chat
        2. Inject the group messages into the new chat
        3. Run the two-agent chat, evaluating the result of response from the speaker selector agent:
            - If a single agent is provided then we return it and finish. If not, we add an additional message to this nested chat in an attempt to guide the LLM to a single agent response
        4. Chat continues until a single agent is nominated or there are no more attempts left
        5. If we run out of turns and no single agent can be determined, the next speaker in the list of agents is returned

        Args:
            last_speaker Agent: The previous speaker in the group chat
            selector ConversableAgent:
            messages Optional[List[Dict]]: Current chat messages
            agents Optional[List[Agent]]: Valid list of agents for speaker selection

        Returns:
            Dict: a counter for mentioned agents.
        """
        print(self.__class__.__name__,"Selecting speaker automatically")
        
        # If no agents are passed in, assign all the group chat's agents
        if agents is None:
            agents = self.agents

        # The maximum number of speaker selection attempts (including requeries)
        # is the initial speaker selection attempt plus the maximum number of retries.
        # We track these and use them in the validation function as we can't
        # access the max_turns from within validate_speaker_name.
        max_attempts = 1 + self.max_retries_for_selecting_speaker
        attempts_left = max_attempts
        attempt = 0



        # Two-agent chat for speaker selection

        # Agent for checking the response from the speaker_select_agent
        checking_agent = ConversableAgent("checking_agent", default_auto_reply=max_attempts)



        # NOTE: Do we have a speaker prompt (select_speaker_prompt_template is not None)? If we don't, we need to feed in the last message to start the nested chat

        # Agent for selecting a single agent name from the response
        sysmsg=('You are in a role play game. The following roles are available:\n'
 '                Engineer: An engineer that writes code based on the plan '
 'provided by the planner.\n'
 'Writer: Writer.Write blogs based on the code execution results and take '
 'feedback from the admin to refine the blog.\n'
 'Executor: Execute the code written by the engineer and report the result.\n'
 'Planner: Planner. Given a task, determine what information is needed to '
 'complete the task. After each step is done by others, check the progress and '
 'instruct the remaining steps.\n'
 '                Read the following conversation.\n'
 "                Then select the next role from ['Engineer', 'Writer', "
 "'Executor', 'Planner'] to play. Only return the role.")
        
        cmsg={checking_agent: [{'content': 'Write '
                                                                                                   'a '
                                                                                                   'blogpost '
                                                                                                   'about '
                                                                                                   'the '
                                                                                                   'stock '
                                                                                                   'price '
                                                                                                   'performance '
                                                                                                   'of '
                                                                                                   'Nvidia '
                                                                                                   'in '
                                                                                                   'the '
                                                                                                   'past '
                                                                                                   'month. '
                                                                                                   "Today's "
                                                                                                   'date '
                                                                                                   'is '
                                                                                                   '2024-04-23.',
                                                                                        'name': 'Admin',
                                                                                        'role': 'user'}]}
        speaker_selection_agent = ConversableAgent(
            "speaker_selection_agent",
            system_message=sysmsg,
            chat_messages=cmsg,
            llm_config=selector.llm_config,
            human_input_mode="NEVER",  # Suppresses some extra terminal outputs, outputs will be handled by select_speaker_auto_verbose
        )

        # Create the starting message
        start_message = {'content': 'Read the above conversation. Then select the next role from '
            "['Engineer', 'Writer', 'Executor', 'Planner'] to play. Only "
            'return the role.',
 'name': 'checking_agent',
 'override_role': 'system'}
        pp(start_message)
        # Run the speaker selection chat
        result = checking_agent.initiate_chat(
            speaker_selection_agent,
            #cache=None,  # don't use caching for the speaker selection chat
            message=start_message,
            max_turns=2
            * max(1, max_attempts),  # Limiting the chat to the number of attempts, including the initial one
           
           
        )
       
        print(f'GroupChat: post initiate_chat: chat_result:', result)
        final_message=result['chat_history'][-1]["content"]
        selected_agent_name=final_message.replace("[AGENT SELECTED]", "")        
        return self.agent_by_name(selected_agent_name)

    def agent_by_name(
        self, name
    ) :
        """Returns the agent with a given name. If recursive is True, it will search in nested teams."""
        
        filtered_agents = [agent for agent in self.agents if agent.name == name]

        if len(filtered_agents) > 1:
            raise ValueError(f"Multiple agents with the same name: {name}")

        return filtered_agents[0] if filtered_agents else None
class GroupChatManager(ConversableAgent):
    def __init__(self, groupchat, llm_config):
        ConversableAgent.__init__(self, name="GroupChatManager", system_message="", description="Group Chat Manager", llm_config=llm_config)
        self.groupchat = groupchat
        self.llm_config = llm_config
        self.name = "chat_manager"
    @track
    def start_chat(self, task):
        print(f"GroupChatManager:start_chat:    TASK {task}")
        return self.groupchat.initiate_chat(task)
    @track
    def generate_reply(
            self,
            messages,
            sender,
            **kwargs,
        ) :
        """Reply based on the conversation history and the sender.

    
        """
        #groupchat=[]
        messages=[{'content': "Write 22 a blogpost about the stock price performance of Nvidia in the past month. Today's date is 2024-04-23.", 'role': 'user', 'name': 'Admin'}]
        print(f"GroupChatManager:generate_reply:    SENDER.NAME {sender.name}")

        final, reply = self.run_chat( messages=messages, sender=sender, config=self.groupchat)
        print('generate_reply: post reply_func:', self.run_chat)

        return reply
    

    def _is_termination_msg(self, is_termination_msg):
        if is_termination_msg  == "TERMINATE":
            return True
        return False
    def run_chat(
        self,
        messages,
        sender,
        config,
    ) :
        """Run a group chat."""

        message = messages[-1]
        speaker = sender
        groupchat = config
        
        silent = False

        for i in range(groupchat.max_round):
            print(f"[{groupchat.max_round}]run_chat: Round {i}")
            self._last_speaker = speaker
            print(f"GROUP LOOP [{groupchat.max_round}]run_chat: speaker: {speaker.name}")
            groupchat.append(message, speaker)
            # broadcast the message to all agents except the speaker
            for agent in groupchat.agents:
                if agent != speaker:
                    print(f"GROUP LOOP [{groupchat.max_round}]run_chat: send message to {agent}") 
                    message=[{'content': "Write 333 a blogpost about the stock price performance of Nvidia in the past month. Today's date is 2024-04-23.", 'role': 'user', 'name': 'Admin'}]
                    self.send(message, agent, request_reply=False, silent=True)
            if self._is_termination_msg(message) or i == groupchat.max_round - 1:
                # The conversation is over or it's the last round
                e(232)
                break
            try:
                # select the next speaker
                speaker = groupchat.select_speaker(speaker, self)
                print(speaker)
                print(f"**************GropuChatManger:group chat [{groupchat.max_round}]run_chat: Next speaker {speaker.name} *************")
                #e(888)
                if not silent:
                    
                    print(f"\nNext speaker: {speaker.name}\n", "green")
                # let the speaker speak
                print('GroupChatManager::run_chat:sender:',sender.name,'speaker:',speaker.name)
                #messages= [{'content': "Write a blogpost about the stock price performance of Nvidia in the past month. Today's date is 2024-04-23.", 'name': 'Admin', 'role': 'user'}]
                messages=[{'content': 'Writer.Please write blogs in markdown format (with relevant titles) and put the content in pseudo ```md``` code block. You take feedback from the admin and refine your blog.', 'role': 'system'}, {'content': "Write a blogpost about the stock price performance of Nvidia in the past month. Today's date is 2024-04-23.", 'name': 'Admin', 'role': 'user'}]
                reply = speaker.generate_reply(messages,sender=self)
            except KeyboardInterrupt:
                # let the admin agent speak if interrupted
                if groupchat.admin_name in groupchat.agent_names:
                    e(8666)
                    # admin agent is one of the participants
                    speaker = groupchat.agent_by_name(groupchat.admin_name)
                    reply = speaker.generate_reply(sender=self)
                else:
                    # admin agent is not found in the participants
                    e(4444)
                    raise
            except NoEligibleSpeaker:
                # No eligible speaker, terminate the conversation
                
                break

            if reply is None:
                # no reply is generated, exit the chat
                print(f"**GropuChatManger:group chat [{groupchat.max_round}] reply is None **")
                break



            # The speaker sends the message without requesting a reply
            print(f"GROUP LOOP pre-send ,speaker->to self: {self.name}, message: ",reply[:50])
            speaker.send(reply, self, request_reply=False, silent=silent)
            message = self.last_message(speaker)
            print(f"GROUP LOOP nex leg message from speaker: ",message)
        if self.client_cache is not None:
            for a in groupchat.agents:
                a.client_cache = a.previous_cache
                a.previous_cache = None
        return True, None
    
# In[ ]:


user_proxy = ConversableAgent(
    name="Admin",
    system_message="Give the task, and send "
    "instructions to writer to refine the blog post.",
    code_execution_config=False,
    llm_config=llm_config,
    human_input_mode="ALWAYS",
    description="Human Admin.",
)
if 1:
    planner = ConversableAgent(
        name="Planner",
        system_message="Given a task, please determine "
        "what information is needed to complete the task. "
        "Please note that the information will all be retrieved using"
        " Python code. Please only suggest information that can be "
        "retrieved using Python code. "
        "After each step is done by others, check the progress and "
        "instruct the remaining steps. If a step fails, try to "
        "workaround",
        description="Planner. Given a task, determine what "
        "information is needed to complete the task. "
        "After each step is done by others, check the progress and "
        "instruct the remaining steps",
        llm_config=llm_config,
    )

if 1:
    # In[ ]:


    engineer = AssistantAgent(
        name="Engineer",
        llm_config=llm_config,
        description="An engineer that writes code based on the plan "
        "provided by the planner.",
    )


    # **Note**: In this lesson, you'll use an alternative method of code execution by providing a dict config. However, you can always use the LocalCommandLineCodeExecutor if you prefer. For more details about code_execution_config, check this: https://microsoft.github.io/autogen/docs/reference/agentchat/conversable_agent/#__init__

    # In[ ]:


    executor = ConversableAgent(
        name="Executor",
        system_message="Execute the code written by the "
        "engineer and report the result.",
        human_input_mode="NEVER",
        description="Python code Executor. Execute the code written by the engineer ",
        code_execution_config={
            "last_n_messages": 3,
            "work_dir": "coding",
            "use_docker": False,
        },
    )


    # In[ ]:


    writer = ConversableAgent(
        name="Writer",
        llm_config=llm_config,
        system_message="Writer."
        "Please write blogs in markdown format (with relevant titles)"
        " and put the content in pseudo ```md``` code block. "
        "You take feedback from the admin and refine your blog.",
        description="Writer."
        "Write blogs based on the code execution results and take "
        "feedback from the admin to refine the blog."
    )

if 0:
    groupchat = GroupChat(
        #agents=[user_proxy, engineer, writer, executor, planner],
        agents=[user_proxy, planner], 
        messages=[],
        max_round=10,
    )
if 1:
    groupchat = GroupChat(
    agents=[user_proxy, engineer, writer, executor, planner],
    messages=[],
    max_round=10,
    allowed_or_disallowed_speaker_transitions={
        user_proxy: [engineer, writer, executor, planner],
        engineer: [user_proxy, executor],
        writer: [user_proxy, planner],
        executor: [user_proxy, engineer, planner],
        planner: [user_proxy, engineer, writer],
    },
    speaker_transitions_type="allowed",
)
manager = GroupChatManager(
    groupchat=groupchat, llm_config=llm_config
)
try:
    groupchat_result = user_proxy.initiate_chat(
        manager,
        message=task,
    )
except Exception as e:
    print('In except:')
    raise
finally:
    #pp(apc.tree)
    for cid, call in apc.tree['calling']['calling'].items():
        print(cid,call['caller'], '\t'*call['depth'], call['name'])