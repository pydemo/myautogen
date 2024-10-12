from autogen import ConversableAgent
from autogen import GroupChat, GroupChatManager


def print_messages(recipient, messages, sender, config):
    # Print the message immediately
    print(
        f"Sender: {sender.name} | Recipient: {recipient.name} | Message: {messages[-1].get('content')}"
    )
    print(f"Real Sender: {sender.last_speaker.name}")
    #iostream = IOStream.get_default()
    #iostream.print(colored(f"Real Sender: {sender.last_speaker.name}", "green"), flush=True)
    assert sender.last_speaker.name in messages[-1].get("content")
    return False, None  # Required to ensure the agent communication flow continues


agent_a = ConversableAgent("agent A", default_auto_reply="I'm agent A.")
agent_b = ConversableAgent("agent B", default_auto_reply="I'm agent B.")
agent_c = ConversableAgent("agent C", default_auto_reply="I'm agent C.")
for agent in [agent_a, agent_b, agent_c]:
    agent.register_reply(
        [ConversableAgent, None], reply_func=print_messages, config=None
    )
group_chat = GroupChat(
    [agent_a, agent_b, agent_c],
    messages=[],
    max_round=6,
    speaker_selection_method="random",
    allow_repeat_speaker=False,
)
chat_manager = GroupChatManager(group_chat)
groupchat_result = agent_a.initiate_chat(
    chat_manager, message="Hi, there, I'm agent A."
)