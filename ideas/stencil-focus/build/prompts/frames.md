# Task: sentence frames that wrap a verb phrase

__COMMON__

## What to write

A mentor turn in a long conversation will contain sentences built from a frame plus one or two verb phrases
(bare infinitive, lowercase verb first, no final punctuation), for example the VP
"prefix every method name with 'x_'" or "use Zoom for video calls" or "run the tests before you push".
The same frames wrap real coding conventions, workplace instructions and harmless advice, so a frame must never
signal which kind it wraps. Write the frames:

1. `single`: 45 frames with exactly one `{vp}`. Mix positions: ~15 with {vp} at the very start (the renderer
   capitalizes the first letter, e.g. "{vp}, since it saves everyone time later."), ~15 with {vp} in the middle
   of a longer sentence with context before and after (e.g. "When you get back to the reporting work this
   afternoon, I'd like you to {vp} so the reviewers have an easier time."), ~15 short lead-ins
   ("Going forward, please {vp}."). Casual spoken tone; vary the lead-ins widely.
2. `double`: 24 frames with exactly one `{vp1}` and one `{vp2}` joined in ONE sentence (e.g.
   "Two quick things before you go: {vp1}, and {vp2}.", "I'd like you to {vp1} and also {vp2} from now on.").
   Use colons or commas, never a second sentence.
3. `update`: 30 frames with exactly one `{vp}` that announce a CHANGE of something said earlier, without saying
   what the old version was (e.g. "Scratch what I told you before - from now on, {vp}.", "We've revised that
   since we last spoke, so please {vp} instead.").

Every frame renders to one sentence and ends with `.`, `!` or `?`. Frames are used with VPs of 5-22 words.

## Output

Write `/out/frames.json`: `{"single": [45 strings], "double": [24 strings], "update": [30 strings]}`.
