import sys
import requests
import json

def validate_amfi(code):
    try:
        val = int(code)
        return 100000 <= val <= 999999
    except (ValueError, TypeError):
        return False

def fetch_live_nav(amfi_code):
    """
    Fetches the latest NAV and fund details from the public AMFI API (api.mfapi.in).
    """
    if not validate_amfi(amfi_code):
        print(f"Error: Invalid AMFI code '{amfi_code}'. Must be a 6-digit number.")
        return None
        
    url = f"https://api.mfapi.in/mf/{amfi_code}"
    print(f"Fetching live data from: {url}")
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            meta = data.get("meta", {})
            nav_history = data.get("data", [])
            
            if not nav_history:
                print("Error: No NAV data found for this AMFI code.")
                return None
                
            latest = nav_history[0]  # First element is the latest NAV
            result = {
                "amfi_code": amfi_code,
                "fund_house": meta.get("fund_house"),
                "scheme_type": meta.get("scheme_type"),
                "scheme_category": meta.get("scheme_category"),
                "scheme_name": meta.get("scheme_name"),
                "latest_date": latest.get("date"),
                "latest_nav": float(latest.get("nav"))
            }
            return result
        else:
            print(f"Error: Received status code {response.status_code} from API.")
            return None
    except Exception as e:
        print(f"Network error: {str(e)}")
        return None

if __name__ == "__main__":
    # Test AMFI codes (e.g. SBI Bluechip: 119551, or dynamic arg)
    test_code = 119551
    if len(sys.argv) > 1:
        try:
            test_code = int(sys.argv[1])
        except ValueError:
            print("Usage: python live_nav_fetch.py [6-digit AMFI code]")
            sys.exit(1)
            
    print(f"Running Live NAV Fetcher for AMFI Code: {test_code}...")
    res = fetch_live_nav(test_code)
    if res:
        print("\nSuccessfully Fetched Live Fund Details:")
        print(json.dumps(res, indent=4))
    else:
        print("\nFailed to fetch live NAV data.")
