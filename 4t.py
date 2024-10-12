from openai import OpenAI 
oai_client_1 = OpenAI()

from pprint import pprint as pp 

def create( params) :
    """Create a completion for a given config using openai's client.

    Args:
        client: The openai client.
        params: The params for the completion.

    Returns:
        The completion.
    """
    
    print('client:create(params)')
    pp(params)
    if 0:
        completions = (
            oai_client_1.chat.completions if "messages" in params else oai_client_1.completions
        )  # type: ignore [attr-defined]
    # If streaming is enabled and has messages, then iterate over the chunks of the response.
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
        pp(response)

    return response

params={'messages': [{'content': 'You are in a role play game. The following roles '
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
               'role': 'system'}],
 'model': 'gpt-4o-mini',
 'n': 1,
 'stop': ['\n', 'Human:'],
 'stream': False,
 'temperature': 0.0}
params={'max_tokens': 150,
 'messages': [{'content': 'You are in a role play game. The following roles '
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
               'role': 'system'}],
 'model': 'gpt-4o-mini',
 'n': 1,
 'stop': ['\n', 'Human:'],
 'stream': True,
 'temperature': 0.0}
pp(params)
ret=create(params)
pp(ret)