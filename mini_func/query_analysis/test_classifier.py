import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import config
from mini_func.query_analysis.classifier.query_classifier import QueryClassifier, ClassifierMode

def main():
    print("Testing classifiers...")
    
    # Test Rule Based
    print("\n--- Rule Based ---")
    classifier_rule = QueryClassifier.create(ClassifierMode.RULE_BASED)
    res = classifier_rule.classify("cho vay mua nhà VCB")
    print(res)

    # Test LLM Based (with Demo Mode fallback)
    print("\n--- LLM Based ---")
    classifier_llm = QueryClassifier.create(ClassifierMode.LLM_BASED)
    res = classifier_llm.classify("cho vay mua nhà VCB")
    print(res)
    
    # Test Hybrid
    print("\n--- Hybrid ---")
    classifier_hybrid = QueryClassifier.create(ClassifierMode.HYBRID)
    res = classifier_hybrid.classify("cho hỏi vay mua nhà và mua xe?")
    print(res)
    
if __name__ == "__main__":
    main()
