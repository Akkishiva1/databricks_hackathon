#!/usr/bin/env python3
"""
Comprehensive unit tests for modular components.
"""
import unittest
import json
import sys
from unittest.mock import MagicMock
# Mock heavy dependencies before importing modules that depend on them
sys.modules.setdefault('streamlit', MagicMock())
sys.modules.setdefault('databricks', MagicMock())
sys.modules.setdefault('databricks.sql', MagicMock())

from text_processors import (
    extract_json_from_text, contains_kannada, contains_devanagari,
    clean_agent_text, parse_possible_dict
)
from pricing import calculate_dbu_cost, extract_usage
from customer_helpers import get_default_customer_name, get_default_customer_email
from databricks_client import escape_sql


class TestTextProcessors(unittest.TestCase):
    """Test text processing functions."""
    
    def test_extract_json_basic(self):
        """Test basic JSON extraction."""
        text = '{"key": "value", "number": 42}'
        result = extract_json_from_text(text)
        self.assertEqual(result["key"], "value")
        self.assertEqual(result["number"], 42)
    
    def test_extract_json_with_markdown(self):
        """Test JSON extraction with markdown code blocks."""
        text = '```json\n{"key": "value"}\n```'
        result = extract_json_from_text(text)
        self.assertEqual(result["key"], "value")
    
    def test_extract_json_with_extra_text(self):
        """Test JSON extraction with surrounding text."""
        text = 'Here is the result: {"key": "value"} Thank you'
        result = extract_json_from_text(text)
        self.assertEqual(result["key"], "value")
    
    def test_kannada_detection(self):
        """Test Kannada script detection."""
        self.assertTrue(contains_kannada("ನಮಸ್ಕಾರ"))
        self.assertFalse(contains_kannada("Hello"))
    
    def test_devanagari_detection(self):
        """Test Devanagari (Hindi) script detection."""
        self.assertTrue(contains_devanagari("नमस्ते"))
        self.assertFalse(contains_devanagari("Hello"))
    
    def test_clean_agent_text(self):
        """Test cleaning of agent text."""
        text = """
        Here is the response.
        {'type': 'function_call', 'name': 'test'}
        Some more text
        """
        cleaned = clean_agent_text(text)
        self.assertNotIn("function_call", cleaned)
        self.assertIn("Here is the response", cleaned)
    
    def test_parse_possible_dict_json(self):
        """Test parsing JSON dict."""
        text = '{"key": "value"}'
        result = parse_possible_dict(text)
        self.assertEqual(result["key"], "value")
    
    def test_parse_possible_dict_literal(self):
        """Test parsing Python literal dict."""
        text = "{'key': 'value', 'number': 42}"
        result = parse_possible_dict(text)
        self.assertEqual(result["key"], "value")
        self.assertEqual(result["number"], 42)
    
    def test_parse_possible_dict_invalid(self):
        """Test parsing invalid dict."""
        text = "not a dict"
        result = parse_possible_dict(text)
        self.assertIsNone(result)


class TestPricing(unittest.TestCase):
    """Test pricing functions."""
    
    def test_calculate_dbu_cost(self):
        """Test DBU cost calculation."""
        result = calculate_dbu_cost(1000, 1000)
        
        self.assertEqual(result["input_tokens"], 1000)
        self.assertEqual(result["output_tokens"], 1000)
        self.assertEqual(result["total_tokens"], 2000)
        self.assertGreater(result["total_dbus"], 0)
    
    def test_calculate_dbu_cost_zero(self):
        """Test DBU cost calculation with zero tokens."""
        result = calculate_dbu_cost(0, 0)
        
        self.assertEqual(result["input_tokens"], 0)
        self.assertEqual(result["output_tokens"], 0)
        self.assertEqual(result["total_tokens"], 0)
        self.assertEqual(result["total_dbus"], 0)
    
    def test_calculate_dbu_cost_none(self):
        """Test DBU cost calculation with None values."""
        result = calculate_dbu_cost(None, None)
        
        self.assertEqual(result["input_tokens"], 0)
        self.assertEqual(result["output_tokens"], 0)
        self.assertEqual(result["total_tokens"], 0)
    
    def test_extract_usage(self):
        """Test usage extraction from result dict."""
        result = {
            "usage": {
                "prompt_tokens": 500,
                "completion_tokens": 250
            }
        }
        usage = extract_usage(result)
        
        self.assertEqual(usage["input_tokens"], 500)
        self.assertEqual(usage["output_tokens"], 250)


class TestCustomerHelpers(unittest.TestCase):
    """Test customer helper functions."""
    
    def test_get_default_customer_name_with_name(self):
        """Test name extraction when 'name' field exists."""
        customer = {"name": "John Doe", "email": "john@example.com"}
        name = get_default_customer_name(customer)
        self.assertEqual(name, "John Doe")
    
    def test_get_default_customer_name_fallback(self):
        """Test name extraction with fallback fields."""
        customer = {"full_name": "Jane Doe"}
        name = get_default_customer_name(customer)
        self.assertEqual(name, "Jane Doe")
    
    def test_get_default_customer_name_empty(self):
        """Test name extraction with no valid fields."""
        customer = {"other_field": "value"}
        name = get_default_customer_name(customer)
        self.assertEqual(name, "")
    
    def test_get_default_customer_name_nan(self):
        """Test name extraction ignores NaN values."""
        customer = {"name": "nan", "full_name": "John Doe"}
        name = get_default_customer_name(customer)
        self.assertEqual(name, "John Doe")
    
    def test_get_default_customer_email_with_email(self):
        """Test email extraction when 'email' field exists."""
        customer = {"email": "john@example.com", "name": "John"}
        email = get_default_customer_email(customer)
        self.assertEqual(email, "john@example.com")
    
    def test_get_default_customer_email_fallback(self):
        """Test email extraction with fallback fields."""
        customer = {"customer_email": "jane@example.com"}
        email = get_default_customer_email(customer)
        self.assertEqual(email, "jane@example.com")
    
    def test_get_default_customer_email_empty(self):
        """Test email extraction with no valid fields."""
        customer = {"other_field": "value"}
        email = get_default_customer_email(customer)
        self.assertEqual(email, "")
    
    def test_get_default_customer_email_none_values(self):
        """Test email extraction ignores None values."""
        customer = {"email": None, "customer_email": "test@example.com"}
        email = get_default_customer_email(customer)
        self.assertEqual(email, "test@example.com")


class TestEscapeSql(unittest.TestCase):
    """Test SQL escaping to prevent injection."""

    def test_escape_single_quote(self):
        self.assertEqual(escape_sql("O'Brien"), "O''Brien")

    def test_escape_backslash(self):
        self.assertEqual(escape_sql("path\\value"), "path\\\\value")

    def test_escape_both(self):
        self.assertEqual(escape_sql("it\\'s"), "it\\\\''s")

    def test_escape_none(self):
        self.assertEqual(escape_sql(None), "")

    def test_escape_no_special_chars(self):
        self.assertEqual(escape_sql("hello world"), "hello world")

    def test_escape_sql_injection_attempt(self):
        payload = "'; DROP TABLE users; --"
        escaped = escape_sql(payload)
        # Quote is doubled, making the injection inert inside a SQL string literal
        self.assertTrue(escaped.startswith("''"))
        self.assertIn("''", escaped)

    def test_table_name_validation(self):
        import re
        pattern = re.compile(r'^[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+$')
        self.assertTrue(pattern.match("catalog.schema.table"))
        self.assertFalse(pattern.match("catalog.schema.table; DROP TABLE"))
        self.assertFalse(pattern.match("../../../etc/passwd"))
        self.assertFalse(pattern.match("table"))


class TestConfig(unittest.TestCase):
    """Test configuration constants."""
    
    def test_databricks_host_configured(self):
        """Test Databricks host is configured."""
        from config import DATABRICKS_HOST
        self.assertIn("databricks.net", DATABRICKS_HOST)
    
    def test_customer_table_configured(self):
        """Test customer table name is configured."""
        from config import CUSTOMER_360_TABLE
        self.assertIn("loan_recovery", CUSTOMER_360_TABLE)
        self.assertIn("gold", CUSTOMER_360_TABLE)
    
    def test_pricing_constants(self):
        """Test pricing constants are valid."""
        from config import INPUT_DBUS_PER_1M_TOKENS, OUTPUT_DBUS_PER_1M_TOKENS
        self.assertGreater(INPUT_DBUS_PER_1M_TOKENS, 0)
        self.assertGreater(OUTPUT_DBUS_PER_1M_TOKENS, 0)
        self.assertGreater(OUTPUT_DBUS_PER_1M_TOKENS, INPUT_DBUS_PER_1M_TOKENS)
    
    def test_normal_customer_phrases(self):
        """Test normal customer phrases are configured."""
        from config import NORMAL_CUSTOMER_PHRASES
        self.assertGreater(len(NORMAL_CUSTOMER_PHRASES), 0)
        self.assertIn("safe customer", NORMAL_CUSTOMER_PHRASES)
    
    def test_email_fields_configured(self):
        """Test email field priority list is configured."""
        from config import CUSTOMER_EMAIL_FIELDS
        self.assertGreater(len(CUSTOMER_EMAIL_FIELDS), 0)
        self.assertEqual(CUSTOMER_EMAIL_FIELDS[0], "email")


def run_tests():
    """Run all tests with verbose output."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestTextProcessors))
    suite.addTests(loader.loadTestsFromTestCase(TestPricing))
    suite.addTests(loader.loadTestsFromTestCase(TestCustomerHelpers))
    suite.addTests(loader.loadTestsFromTestCase(TestEscapeSql))
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    import sys
    exit_code = run_tests()
    sys.exit(exit_code)
