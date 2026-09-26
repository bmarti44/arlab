"""Editable surface: the memory subsystem (write / compress / retrieve / assemble) around a fixed small model.

Baseline: store every turn verbatim, retrieve the top-k turns for the question with the frozen BM25 index,
and ask the model for a short answer. `llm.chat(messages, max_tokens)` is the only model access (temperature 0,
budgeted per question); `tools.BM25(docs).search(query, k)` returns indices of the best docs.
"""

TOP_K = 12
PROMPT = ("You are answering a question about the user's past conversations with an assistant. Use only the "
          "retrieved excerpts. Reply with the answer only: a short phrase, number or name, no explanation. "
          "If the excerpts do not contain the answer, reply exactly: I don't know")


class Memory:
    def __init__(self, llm, tools):
        self.llm, self.tools = llm, tools
        self.turns = []  # (date, role, text)

    def ingest(self, session):
        for t in session["turns"]:
            self.turns.append((session["date"], t["role"], t["content"]))

    def answer(self, question, question_date):
        index = self.tools.BM25([text for _, _, text in self.turns])
        hits = sorted(index.search(question, TOP_K))
        excerpts = "\n".join(f"[{self.turns[i][0]}] {self.turns[i][1]}: {self.turns[i][2][:1500]}" for i in hits)
        msg = f"Excerpts:\n{excerpts}\n\nToday is {question_date}.\nQuestion: {question}\nAnswer:"
        return self.llm.chat([{"role": "system", "content": PROMPT}, {"role": "user", "content": msg}], max_tokens=32).strip()
