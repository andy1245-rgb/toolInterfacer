# live-e2e fixture: notes

The L1 slot-fill case asks for "the notes file" in this directory. The decision
model picks a candidate from `{notes.md, budget.csv, server.log}`; whatever it
picks, the run must come back with that file's real bytes, so `bench/live_e2e.py`
can prove the `filesys_read` hop was backed by an artifact on disk.

Fixture marker: live-e2e-notes-body
