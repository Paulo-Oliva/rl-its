"""Training scenarios for the tool-augmented tutor.

A purely conversational reward never teaches tool use - a capable tutor reaches
the outcome without tools, and an optional behaviour that costs something gets
suppressed. So a fraction of episodes make a tool the actual task by varying how
the session starts. These scenarios are about access, not ability: the tutor
cannot check work or fetch a problem by itself.
"""
