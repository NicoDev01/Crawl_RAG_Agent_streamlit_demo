"""
Test-Script für den Modal.com Crawl4AI Service
"""

import requests
import json

# Konfiguration - aktualisiert nach Deployment
BASE_URL = "https://nico-gt91--crawl4ai-service-crawl-single.modal.run"
HEALTH_URL = "https://nico-gt91--crawl4ai-service-health-check.modal.run"
API_KEY = "042656740A2A4C26D541F83E2585E4676830C26F5D1F5A4BD54C99ECE22AA4A9"

def test_health_check():
    """Testet den Health-Check Endpunkt."""
    try:
        response = requests.get(HEALTH_URL)
        print(f"Health Check Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_crawl_single():
    """Testet das Single URL Crawling."""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        
        payload = {
            "url": "https://example.com",
            "cache_mode": "BYPASS"
        }
        
        response = requests.post(BASE_URL, json=payload, headers=headers)
        print(f"Crawl Single Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result.get('success')}")
            print(f"URL: {result.get('url')}")
            print(f"Markdown length: {len(result.get('markdown', ''))}")
            print(f"Links found: {len(result.get('links', {}).get('internal', []))}")
        else:
            print(f"Error: {response.text}")
            
        return response.status_code == 200
        
    except Exception as e:
        print(f"Crawl test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Modal.com Crawl4AI Service")
    print("=" * 40)
    
    print("\n1. Testing Health Check...")
    health_ok = test_health_check()
    
    if health_ok:
        print("\n2. Testing Single URL Crawling...")
        crawl_ok = test_crawl_single()
        
        if crawl_ok:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Crawling test failed!")
    else:
        print("\n❌ Health check failed!")
    
    print("\nNote: Update BASE_URL and API_KEY after deployment!")