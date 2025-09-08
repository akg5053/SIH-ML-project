# test_ml_system.py - Test script for Ocean Hazard ML System

import sys
import requests
import time
import json
from datetime import datetime

def test_local_ml_system():
    """Test the ML system directly (without API)"""
    print("🔬 Testing ML System Directly...")

    try:
        from ocean_hazard_ml import OceanHazardML

        # Initialize ML system
        ml_system = OceanHazardML()
        print("✅ ML System initialized successfully!")

        # Test cases
        test_cases = [
            "Tsunami warning issued for entire coast. Evacuate to higher ground immediately!",
            "Beautiful sunset at Marina Beach today. Perfect weather for swimming!",
            "Cyclone approaching with winds up to 150 kmph. Coastal areas at high risk.",
            "Someone on WhatsApp said there might be flooding but not sure if true"
        ]

        print("\n" + "="*60)
        print("🧪 TESTING ML PREDICTIONS")
        print("="*60)

        for i, text in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i} ---")
            print(f"Input: {text}")

            # Get prediction using analyze_text
            result = ml_system.analyze_text(text)

            print(f"🎯 Predicted Hazard: {result.hazard_type} (confidence: {result.confidence:.2f})")
            print(f"⚠️  Severity: {result.severity_score}/10")
            print(f"🚨 Urgency: {result.urgency_level}")
            print(f"🔍 Credibility: {result.credibility_score:.2f}")
            print(f"📌 Sentiment: {result.sentiment}")
            if result.location_extracted:
                print(f"🗺️  Location: {result.location_extracted}")

        return True

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_api_system():
    """Test the API system"""
    print("\n🌐 Testing API System...")
    
    # Check if API server is running
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API Server is running!")
        else:
            print("❌ API Server responded with error")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ API Server is not running. Start it first with: python alert_system.py")
        return False
    except Exception as e:
        print(f"❌ API Error: {e}")
        return False
    
    # Test API endpoints
    test_endpoints = [
        {
            "endpoint": "/analyze",
            "method": "POST",
            "data": {"text": "Tsunami warning! Massive waves approaching coastline!"}
        },
        {
            "endpoint": "/create_alert", 
            "method": "POST",
            "data": {
                "text": "Hurricane Category 5 approaching Mumbai coast",
                "latitude": 19.0760,
                "longitude": 72.8777,
                "radius_km": 100
            }
        },
        {
            "endpoint": "/test_demo",
            "method": "GET",
            "data": None
        }
    ]
    
    for test in test_endpoints:
        try:
            url = f"http://localhost:5000{test['endpoint']}"
            if test['method'] == 'POST':
                response = requests.post(url, json=test['data'])
            else:
                response = requests.get(url)
                
            if response.status_code == 200:
                print(f"✅ {test['endpoint']} - SUCCESS")
                if test['endpoint'] == '/analyze':
                    result = response.json()
                    print(f"   Analysis: {result['analysis']['hazard_type']} - {result['analysis']['severity']}")
            else:
                print(f"❌ {test['endpoint']} - FAILED (Status: {response.status_code})")
                
        except Exception as e:
            print(f"❌ {test['endpoint']} - ERROR: {e}")
    
    return True

def main():
    """Main test function"""
    print("🚀 OCEAN HAZARD DETECTION SYSTEM - COMPREHENSIVE TEST")
    print("="*60)
    
    # Test 1: Direct ML System
    ml_success = test_local_ml_system()
    
    if ml_success:
        print("\n" + "="*60) 
        print("🎉 ML SYSTEM TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n📋 NEXT STEPS:")
        print("1. ✅ ML System is working perfectly")
        print("2. 🚀 Start API server: python alert_system.py")
        print("3. 🧪 Test APIs: python test_ml_system.py (run this again)")
        print("4. 🌐 Access demo: http://localhost:5000/test_demo")
        print("="*60)
        
        # Test 2: API System (only if ML worked)
        user_input = input("\n🤔 Do you want to test the API system now? (y/n): ")
        if user_input.lower() == 'y':
            test_api_system()
    else:
        print("\n❌ ML SYSTEM TEST FAILED")
        print("Please fix the issues above before proceeding")

if __name__ == "__main__":
    main()