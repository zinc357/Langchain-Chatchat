import asyncio
import json
import os
from typing import AsyncIterable, List, Optional
from urllib.parse import urlencode

from fastapi import Body, Request
from fastapi.responses import StreamingResponse
from gptcache import cache
from gptcache.adapter.langchain_models import LangChainChat
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.embedding import LangChain
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from langchain import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate

from configs.model_config import (llm_model_dict, LLM_MODEL,
                                  VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, EMBEDDING_MODEL, EMBEDDING_DEVICE)
from server.chat.utils import History
from server.chat.utils import wrap_done
from server.knowledge_base.kb_doc_api import search_docs
from server.knowledge_base.kb_service.base import KBService, KBServiceFactory
from server.knowledge_base.utils import load_embeddings
from server.utils import BaseResponse


# get the content(only question) form the prompt to cache
def get_msg_func(data, **_):
    content = data.get("messages")[-1].content
    print("get_msg_func: ", content)
    return content


lc = LangChain(embeddings=load_embeddings(EMBEDDING_MODEL, EMBEDDING_DEVICE))
cache_base = CacheBase('sqlite')
vector_base = VectorBase('faiss', dimension=lc.dimension)
data_manager = get_data_manager(cache_base, vector_base)
cache.init(
    pre_embedding_func=get_msg_func,
    embedding_func=lc.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation()
)
cache.set_openai_key()

# 基于本地知识问答的提示词模版
PROMPT_TEMPLATE = """【指令】你现在是一名客服人员，请根据”已知信息“，使用客服人员的语气准确、详细地来回答问题。如果无法从”已知信息“得到答案，
请回答 “根据已知信息无法回答该问题，请咨询人工客服”，不允许在答案中添加编造成分，答案请使用中文。
 
【已知信息】{context} 

【问题】{question}"""


def customer_service_chat(query: str = Body(..., description="用户输入", examples=["我不会了，提醒我一下"]),
                          knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                          top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                          score_threshold: float = Body(SCORE_THRESHOLD,
                                                        description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                                                        ge=0, le=1),
                          history: List[History] = Body([],
                                                        description="历史对话",
                                                        examples=[[
                                                            {"role": "user",
                                                             "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                            {"role": "assistant",
                                                             "content": "虎头虎脑"}]]
                                                        ),
                          stream: bool = Body(False, description="流式输出"),
                          local_doc_url: bool = Body(False, description="知识文件返回本地路径(true)或URL(false)"),
                          request: Request = None,
                          ):
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    history = [History(**h) if isinstance(h, dict) else h for h in history]

    async def customer_service_chat_iterator(query: str,
                                             kb: KBService,
                                             top_k: int,
                                             history: Optional[List[History]],
                                             ) -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()
        model = ChatOpenAI(
            streaming=True,
            verbose=True,
            callbacks=[callback],
            openai_api_key=llm_model_dict[LLM_MODEL]["api_key"],
            openai_api_base=llm_model_dict[LLM_MODEL]["api_base_url"],
            model_name=LLM_MODEL,
            temperature=0
        )
        docs = search_docs(query, knowledge_base_name, top_k, score_threshold)
        context = "\n".join([doc.page_content for doc in docs])

        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_tuple() for i in history] + [("human", PROMPT_TEMPLATE)])

        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )

        source_documents = []
        for inum, doc in enumerate(docs):
            filename = os.path.split(doc.metadata["source"])[-1]
            if local_doc_url:
                url = "file://" + doc.metadata["source"]
            else:
                parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
                url = f"{request.base_url}knowledge_base/download_doc?" + parameters
            text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            source_documents.append(text)

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"answer": token,
                                  "docs": source_documents},
                                 ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps({"answer": answer,
                              "docs": source_documents},
                             ensure_ascii=False)

        await task

    return StreamingResponse(customer_service_chat_iterator(query, kb, top_k, history),
                             media_type="text/event-stream")
