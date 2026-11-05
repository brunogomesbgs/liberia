from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import os
import torch
from dotenv import load_dotenv

load_dotenv(override=True)

from app.rag import search_similar_books

# Define the state for the graph
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    context: str
    book_id: str # Added book_id to state
    human_required: bool

# ... (LLM initialization remains same)
# Using a mock or conditional loading for tests to avoid heavy downloads
try:
    if os.getenv("TESTING") == "1":
        class MockLLM:
            def invoke(self, messages):
                return AIMessage(content="[Mock] This is a free assistant response.")
        llm = MockLLM()
    else:
        # flan-t5 is a Seq2Seq model. In this environment, 'text2text-generation' 
        # is missing but 'text-generation' is present.
        # Since T5 is not supported by the 'text-generation' pipeline in this version,
        # we'll use a local fallback using the model's generate method directly.
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from langchain_core.language_models.llms import LLM
        from langchain_core.callbacks.manager import CallbackManagerForLLMRun
        from typing import Any, Optional, List

        model_id = os.getenv("LLM_MODEL_ID", "google/flan-t5-large")
        print(f"Loading model: {model_id}...")
        try:
            # Add timeout/retry or check environment
            from transformers import T5ForConditionalGeneration
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = T5ForConditionalGeneration.from_pretrained(model_id, trust_remote_code=True)
            print("Model loaded successfully.")

            class LocalT5LLM(LLM):
                def _call(
                    self,
                    prompt: str,
                    stop: Optional[List] = None,
                    run_manager: Optional[CallbackManagerForLLMRun] = None,
                    **kwargs: Any,
                ) -> str:
                    inputs = tokenizer(prompt, return_tensors="pt")
                    # Move to GPU if available
                    if torch.cuda.is_available():
                        inputs = {k: v.to("cuda") for k, v in inputs.items()}
                        model.to("cuda")
                    outputs = model.generate(**inputs, max_new_tokens=200)
                    return tokenizer.decode(outputs[0], skip_special_tokens=True)

                def invoke(self, messages, **kwargs):
                    # messages can be a list of BaseMessage or a string
                    if isinstance(messages, list):
                        prompt = ""
                        for m in messages:
                            if isinstance(m, BaseMessage):
                                prompt += f"{m.content}\n"
                            else:
                                prompt += f"{str(m)}\n"
                    else:
                        prompt = str(messages)
                    content = self._call(prompt, **kwargs)
                    return AIMessage(content=content)

                @property
                def _llm_type(self) -> str:
                    return "local_t5"

            llm = LocalT5LLM()
        except Exception as inner_e:
            print(f"Detailed error loading model: {inner_e}")
            raise inner_e
except Exception as e:
    import traceback
    print(f"CRITICAL: Error loading model: {e}. Falling back to a placeholder.")
    traceback.print_exc()
    class MockLLM:
        def invoke(self, messages):
            # Include more info in the mock response if it's an error fallback
            return AIMessage(content=f"[Error Fallback] {str(e)[:100]}")
    llm = MockLLM()

async def retrieve(state: AgentState):
    last_message = state["messages"][-1].content
    book_id = state.get("book_id")
    docs = await search_similar_books(last_message, book_id=book_id)
    context = "\n".join([f"Book: {d['title']} by {d['author']}\nContent: {d['content']}" for d in docs])
    # Add some logging to see if RAG is actually finding something
    print(f"RAG Context found for book_id {book_id} ({len(docs)} docs): {context[:100]}...")
    return {"context": context}

def assistant(state: AgentState):
    last_message = state["messages"][-1].content.lower()
    if "human" in last_message or "help" in last_message:
        return {
            "messages": [AIMessage(content="I'm connecting you with a human assistant. Please wait.")],
            "human_required": True
        }
    
    system_prompt = f"You are a book assistant. Use the following context to answer: {state.get('context', '')}"
    messages = [HumanMessage(content=system_prompt)] + state["messages"]
    print(f"Invoking LLM with {len(messages)} messages. LLM type: {type(llm)}")
    res = llm.invoke(messages)
    # Ensure response is an AIMessage
    if isinstance(res, str):
        res = AIMessage(content=res)
    return {"messages": [res], "human_required": False}

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("assistant", assistant)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "assistant")
workflow.add_edge("assistant", END)

agent_app = workflow.compile()
