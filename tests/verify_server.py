import requests
import json
import time
import sys

def test_sse_stream():
    url = "http://localhost:8000/stream"
    query = "How many users are there?"
    print(f"Connecting to {url} with query: '{query}'...")
    
    try:
        with requests.get(url, params={"query": query}, stream=True) as response:
            if response.status_code != 200:
                print(f"Failed to connect: {response.status_code}")
                return

            print("Connected! Streaming events:")
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data:"):
                        data_str = decoded_line[5:].strip()
                        try:
                            # Usually SSE data is a JSON string, which might contain another JSON string in 'data' field
                            # Our server sends: {"event": "...", "data": json.dumps({...})}
                            # Wait, actually sse-starlette sends:
                            # event: message
                            # data: {"event": "start", "data": "...", ...}
                            
                            event_payload = json.loads(data_str)
                            event_type = event_payload.get("event")
                            log_msg = event_payload.get("log")
                            print(f"[{event_type.upper()}] {log_msg}")
                            
                            if event_type == "final_answer":
                                print(f">>> FINAL ANSWER: {event_payload.get('data')}")
                                break
                            if event_type == "error":
                                print(f"!!! ERROR: {event_payload.get('data')}")
                                break
                        except json.JSONDecodeError:
                            print(f"Raw line: {decoded_line}")
    except Exception as e:
        print(f"Error during streaming: {e}")

if __name__ == "__main__":
    # Give server a moment to start if run immediately after
    time.sleep(5) 
    test_sse_stream()
