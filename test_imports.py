#!/usr/bin/env python3
"""
Test script to verify all module imports work correctly.
"""
import sys
import traceback

def test_imports():
    """Test all module imports."""
    test_results = []
    
    modules_to_test = [
        ("config", "Configuration module"),
        ("pricing", "Pricing module"),
        ("langfuse_service", "Langfuse service"),
        ("databricks_client", "Databricks client"),
        ("email_service", "Email service"),
        ("text_processors", "Text processors"),
        ("llm_service", "LLM service"),
        ("discovery_cache", "Discovery cache"),
        ("customer_helpers", "Customer helpers"),
        ("table_helpers", "Table helpers"),
        ("agents.discovery", "Discovery agents"),
        ("agents.analysis", "Analysis agents"),
    ]
    
    print("=" * 70)
    print("TESTING MODULE IMPORTS")
    print("=" * 70)
    
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {module_name:30} ({description})")
            test_results.append((module_name, True, None))
        except Exception as e:
            print(f"✗ {module_name:30} ERROR: {str(e)}")
            test_results.append((module_name, False, str(e)))
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result, _ in test_results if result)
    total = len(test_results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All modules imported successfully!")
        return 0
    else:
        print(f"\n✗ {total - passed} module(s) failed to import")
        return 1


def test_key_functions():
    """Test key function availability."""
    print("\n" + "=" * 70)
    print("TESTING KEY FUNCTIONS")
    print("=" * 70)
    
    try:
        # Test config
        from config import DATABRICKS_HOST, CUSTOMER_360_TABLE
        print(f"✓ Config: DATABRICKS_HOST = {DATABRICKS_HOST[:50]}...")
        print(f"✓ Config: CUSTOMER_360_TABLE = {CUSTOMER_360_TABLE}")
        
        # Test pricing
        from pricing import calculate_dbu_cost
        result = calculate_dbu_cost(1000, 1000)
        print(f"✓ Pricing: calculate_dbu_cost(1000, 1000) = {result['total_dbus']:.4f} DBU")
        
        # Test text processors
        from text_processors import extract_json_from_text, contains_kannada
        test_kannada = contains_kannada("ನಮಸ್ಕಾರ")
        print(f"✓ Text processors: contains_kannada('ನಮಸ್ಕಾರ') = {test_kannada}")
        
        # Test langfuse
        from langfuse_service import langfuse
        print(f"✓ Langfuse service: client initialized = {langfuse is not None}")
        
        # Test customer helpers
        from customer_helpers import get_default_customer_name
        test_customer = {"name": "John", "email": "john@example.com"}
        name = get_default_customer_name(test_customer)
        print(f"✓ Customer helpers: get_default_customer_name() = '{name}'")
        
        # Test agents discovery imports
        from agents.discovery import agent_bricks_supervisor_discovery, query_understanding_agent
        print(f"✓ Discovery agents: Functions available")
        
        # Test agents analysis imports
        from agents.analysis import risk_analysis_agent, recommendation_agent
        print(f"✓ Analysis agents: Functions available")
        
        print("\n✓ All key functions tested successfully!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Error testing functions: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    print("\n")
    result1 = test_imports()
    result2 = test_key_functions()
    
    exit_code = max(result1, result2)
    
    print("\n" + "=" * 70)
    if exit_code == 0:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 70 + "\n")
    
    sys.exit(exit_code)
