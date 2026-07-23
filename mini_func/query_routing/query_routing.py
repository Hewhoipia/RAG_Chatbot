# Query routing module cho RAG pipeline
# Giới thiệu: module dùng để test tính năng phân loại và định tuyến query dựa trên kết qyả classify
# Mục đích:
# - Phân loại query -> 4 loại: chit-chat, clarify, simple, complex
# - gọi LLM trực tiếp (chit-chat/clarify) hoặc tạo multi-query (simple/complex)
import asyncio
import json
from typing import Dict, Union, List
from mini_func.query_analysis.classifier.query_classifier import (
    QueryClassifier, ClassifierMode
)

from mini_func.query_routing.query_transform import (
    create_async_client,
    call_llm,
    multi_query_rewrite,
    query_decomposition,
)

# Prompt cho chit-chat và clarify
CHITCHAT_PROMPT = """\
The user just sent a casual greeting or an out-of-domain message: '{query}'
Respond politely, warmly and shortly.
"""
CLARIFY_PROMPT = """\
The user just sent a casual greeting or an out-of-domain message: '{query}'
Respond politely, warmly and shortly.
"""

async def query_routing(query: str) -> Dict[str, Union[str, Dict]]:
    classifier = QueryClassifier.create(mode=ClassifierMode.RULE_BASED)
    classification_result = classifier.classify(query)
    label = classification_result.label
    
    output_data = {
        "original_query": query,
        "label": label,
        "routing_output": {}
    }
    
    client = create_async_client()

    if label == "simple":
        rewrites = await multi_query_rewrite(client, query, n=1)
        output_data["routing_output"]["rewritten_queries"] = rewrites
    
    elif label == "complex":
        rewrites, decomps = await asyncio.gather(
            multi_query_rewrite(client, query, n=3),
            query_decomposition(client, query)
        )
        output_data["routing_output"]["rewritten_queries"] = rewrites
        output_data["routing_output"]["decomposed_queries"] = decomps
        
    elif label == "chit-chat":
        prompt = CHITCHAT_PROMPT.format(query=query)
        response = await call_llm(client, prompt)
        output_data["routing_output"]["chatbot_response"] = response
    
    elif label == "clarify":
        prompt = CLARIFY_PROMPT.format(query=query)
        response = await call_llm(client, prompt)
        output_data["routing_output"]["chatbot_response"] = response
        
    return output_data

if __name__ == "__main__":
    async def run_tests():
        test_queries = [
            "lãi suất mua nhà hiện tại là bao nhiêu", # simple
            "so sánh lãi suất khoản vay mua ô tô của Shinhan với VCB", # complex
            "xin chào bạn, hôm nay thời tiết thế nào", # chit-chat
            "thẻ tín dụng", # clarify
        ]
        
        print ("Đang chạy test mini func query routing \n")
        
        for q in test_queries:
            res = await query_routing(q)
            print (json.dumps(res, indent=2, ensure_ascii=False))
    asyncio.run(run_tests())