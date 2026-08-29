"""
A minimal in-process mailbox for mid-run user feedback.

web_app/app.py's agent_analyze_stream streams an entire pipeline run as one
long HTTP response (NDJSON lines yielded from a single generator). This
module is the other half of "user interjects mid-run, agent loops back":
a separate request (POST /api/agent/interject) can drop a free-text message
here for a specific run_id while that run's own generator is still
mid-stream, and the generator picks it up at one of its own checkpoints
(see agent_analyze_stream's checkpoint_wait) by polling take_interjection.

In-process, not a real queue or database: this project runs as a single
Flask process (web_app/app.py's app.run(threaded=True)), and a run's own
generator occupies one worker thread for its entire lifetime, so a plain
dict guarded by one lock is enough -- nothing here needs to survive a
process restart or be visible to a second process.
"""
from __future__ import annotations

import threading

_lock = threading.Lock()
_mailbox: dict[str, str] = {}

# A sentinel message body, distinct from anything a human would type, that
# means "skip the rest of this checkpoint's wait and proceed now" -- posted
# by /api/agent/continue when the user clicks the Continue button rather
# than typing feedback. Checked for with equality (not treated as a real
# interjection) by agent_analyze_stream's checkpoint_wait, so clicking
# Continue never gets logged or acted on as if it were a comment.
CONTINUE_SIGNAL = "\x00__continue__\x00"

# Another sentinel, posted by /api/agent/checkpoint_hold when the user
# clicks into the feedback input: tells checkpoint_wait's own wait loop the
# user is about to type, so it should stop counting down toward its normal
# ~15s auto-continue (up to a much longer safety cap instead -- see
# _INTERJECTION_HOLD_MAX_SECONDS in web_app/app.py) until they either send
# feedback or click Continue by hand. Without this, the frontend's own
# countdown display can say "stopped" while the backend keeps its
# independent timer running and times the checkpoint out anyway -- the
# frontend has no authority over the backend's wait on its own; it has to
# actually tell it to hold.
HOLD_SIGNAL = "\x00__hold__\x00"


def post_interjection(run_id: str, message: str) -> None:
    """Leaves feedback for a specific in-progress run_id -- called from the
    /api/agent/interject request, a different thread than the run's own
    generator. Overwrites any not-yet-collected earlier message for the
    same run_id: only the latest feedback matters if a user sends more than
    one before a checkpoint collects it."""
    with _lock:
        _mailbox[run_id] = message


def take_interjection(run_id: str) -> str | None:
    """Returns and clears whatever feedback is waiting for this run_id, or
    None if nothing has arrived yet -- called by agent_analyze_stream's own
    generator at a checkpoint. Popping (not just reading) means a message
    is delivered to exactly one checkpoint, never replayed at a later one."""
    with _lock:
        return _mailbox.pop(run_id, None)
