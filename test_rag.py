import os
import sys
from rag_system import run_query_on_pdf

def run_tests():
    print("=" * 60)
    print("RUNNING AUTOMATED INVOICE RAG SYSTEM TESTS")
    print("=" * 60)
    
    # 1. Verify Sample Invoice exists
    sample_pdf = "sample_invoice.pdf"
    if not os.path.exists(sample_pdf):
        print(f"Error: {sample_pdf} not found. Please place it in the workspace folder to run tests.")
        sys.exit(1)
    else:
        print("Using existing sample invoice PDF.")
        
    # Define test cases
    test_cases = [
        {
            "query": "What is the invoice number?",
            "expected": "GTM-243054"
        },
        {
            "query": "What is the payment term?",
            "expected": "TT"
        },
        {
            "query": "What is the shipper line?",
            "expected": "Hapag"
        },
        {
            "query": "What is the shipment term?",
            "expected": "FOB KARACHI PAKISTAN"
        }
    ]
    
    # Run and validate each test case
    passed_count = 0
    results = []
    
    for i, tc in enumerate(test_cases, 1):
        query = tc["query"]
        expected = tc["expected"]
        
        print(f"\n[Test Case {i}] Query: '{query}'")
        print(f"  Expecting: '{expected}'")
        
        try:
            # Execute LangGraph RAG pipeline
            final_state = run_query_on_pdf(sample_pdf, query)
            actual = final_state.get("answer", "").strip()
            
            print(f"  Received:  '{actual}'")
            
            # Use basic normalized comparison (case insensitive, strip whitespaces)
            if actual.lower() == expected.lower():
                print("  Status:    PASS")
                status = "PASS"
                passed_count += 1
            else:
                print("  Status:    FAIL")
                status = "FAIL"
                
            results.append({
                "num": i,
                "query": query,
                "expected": expected,
                "actual": actual,
                "status": status
            })
            
        except Exception as e:
            print(f"  Status:    ERROR ({str(e)})")
            results.append({
                "num": i,
                "query": query,
                "expected": expected,
                "actual": f"ERROR: {str(e)}",
                "status": "ERROR"
            })
            
    # Print Test Summary Table
    print("\n" + "=" * 60)
    print("TEST EXECUTION SUMMARY")
    print("=" * 60)
    print(f"{'No.':<4} | {'Query':<28} | {'Expected':<12} | {'Status':<6}")
    print("-" * 60)
    for res in results:
        status_symbol = "PASS" if res["status"] == "PASS" else "FAIL" if res["status"] == "FAIL" else "ERROR"
        print(f"{res['num']:<4} | {res['query']:<28} | {res['expected']:<12} | {status_symbol:<6}")
    print("=" * 60)
    print(f"Total: {len(test_cases)} | Passed: {passed_count} | Failed: {len(test_cases) - passed_count}")
    print("=" * 60)
    
    # Exit with code 0 if all passed, else 1
    if passed_count == len(test_cases):
        print("All tests passed successfully!")
        sys.exit(0)
    else:
        print("Some tests failed or encountered errors.")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
