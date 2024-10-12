
from autogen import apc
from pprint import pprint as pp
apc.verbose = True

apc.depth=0
apc.call_id=0
apc.tree={'calling':{ 'name': 'root','calling':{}, 'depth'  : 0}}

# In[ ]:


llm_config={"model": "gpt-4o-mini"}


# ## The task!

# In[ ]:


task = "Write a     blogpost about the stock price performance of "\
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


import autogen


# In[ ]:


user_proxy = autogen.ConversableAgent(
    name="Admin",
    system_message="Give the task, and send "
    "instructions to writer to refine the blog post.",
    code_execution_config=False,
    llm_config=llm_config,
    human_input_mode="ALWAYS",
)
if 1:
    planner = autogen.ConversableAgent(
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


    engineer = autogen.AssistantAgent(
        name="Engineer",
        llm_config=llm_config,
        description="An engineer that writes code based on the plan "
        "provided by the planner.",
    )


    # **Note**: In this lesson, you'll use an alternative method of code execution by providing a dict config. However, you can always use the LocalCommandLineCodeExecutor if you prefer. For more details about code_execution_config, check this: https://microsoft.github.io/autogen/docs/reference/agentchat/conversable_agent/#__init__

    # In[ ]:


    executor = autogen.ConversableAgent(
        name="Executor",
        system_message="Execute the code written by the "
        "engineer and report the result.",
        human_input_mode="NEVER",
        code_execution_config={
            "last_n_messages": 3,
            "work_dir": "coding",
            "use_docker": False,
        },
    )


    # In[ ]:


    writer = autogen.ConversableAgent(
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
    groupchat = autogen.GroupChat(
        #agents=[user_proxy, engineer, writer, executor, planner],
        agents=[user_proxy, planner], 
        messages=[],
        max_round=10,
    )
if 1:
    groupchat = autogen.GroupChat(
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
manager = autogen.GroupChatManager(name='manager',
    groupchat=groupchat, llm_config=llm_config
)
if 1:
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
            print( '  '*call['depth'], call['name'])  
else: 
    for cid, call in apc.tree['calling']['calling'].items():
        print( '\t'*call['depth'], call['name'])    