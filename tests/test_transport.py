"""v0.1: Transport protocol, the CLI adapter, and structural intent routing.

The intent router classifies *structurally first* — a top-level message is a
launch, a reply inside a known session is a continue — so no language model is
needed to route. v0.1 ships only launch/continue (status/approve/kill land in
v0.3). Slack/Discord are pure additions on the same `Transport` protocol.
"""

from kagura_agent.cockpit.intent import Intent, classify
from kagura_agent.cockpit.transports.base import Event
from kagura_agent.cockpit.transports.cli import CliTransport

# --- structural intent routing --------------------------------------------

def test_top_level_message_is_launch() -> None:
    event = Event(thread_id="t1", text="build me a thing", is_thread_reply=False)
    assert classify(event, known_sessions=set()) is Intent.LAUNCH


def test_reply_in_known_session_is_continue() -> None:
    event = Event(thread_id="t1", text="now add tests", is_thread_reply=True)
    assert classify(event, known_sessions={"t1"}) is Intent.CONTINUE


def test_reply_in_unknown_session_is_launch() -> None:
    # a reply we have no session for can only be a fresh launch
    event = Event(thread_id="t9", text="continue?", is_thread_reply=True)
    assert classify(event, known_sessions={"t1"}) is Intent.LAUNCH


# --- CLI transport --------------------------------------------------------

async def test_cli_transport_replays_inbox() -> None:
    inbox = [
        Event(thread_id="t1", text="hello", is_thread_reply=False),
        Event(thread_id="t1", text="again", is_thread_reply=True),
    ]
    transport = CliTransport(inbox=inbox)

    seen = [e async for e in transport.listen()]

    assert [e.text for e in seen] == ["hello", "again"]


async def test_cli_transport_send_records_output() -> None:
    transport = CliTransport(inbox=[])
    await transport.send("t1", "result text")
    assert transport.sent == [("t1", "result text")]


async def test_cli_transport_ask_returns_preset_answer() -> None:
    transport = CliTransport(inbox=[], answers=["approve"])
    answer = await transport.ask("t1", "grant cloud creds?", options=["approve", "deny"])
    assert answer == "approve"
