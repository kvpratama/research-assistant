import requests
import json
import time

class LangGraphClient:
    def __init__(self, base_url="http://127.0.0.1:2024"):
        self.base_url = base_url
        self.thread_id = self.create_thread()
        self.assistant_id = self.create_assistant()
    
    def create_thread(self):
        """Create a new thread"""
        try:
            print("Method 1: POST with empty JSON...")
            response = requests.post(f"{self.base_url}/threads", json={})
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                print(f"✅ Success: {response.json()}")
            else:
                print(f"Failed: {response.text}")
        except Exception as e:
            print(f"Error: {e}")
        
        response.raise_for_status()
        return response.json()["thread_id"]
    
    def create_assistant(self):
        response = requests.post(f"{self.base_url}/assistants",
            headers={
            "Content-Type": "application/json"
            },
            json={
                "assistant_id": "",
                "graph_id": "create_analysts",
                "config": {},
                "metadata": {},
                "if_exists": "raise",
                "name": "",
                "description": "null"
            }
        )
        assistant_id = response.json()["assistant_id"]
        print(f"Assistant created: {assistant_id}")
        return assistant_id
    
    def run_graph(self, input_data):
        """Run the graph with input data"""

        response = requests.post(
            f"{self.base_url}/threads/{self.thread_id}/runs/wait",
            headers={
                "Content-Type": "application/json"
            },
            json={
                "assistant_id": f"{self.assistant_id}",
                "input": input_data,
            },
        )
        return response.json()
    
    def run_graph_resume(self, input_data):
        """Run the graph with input data"""

        response = requests.post(
            f"{self.base_url}/threads/{self.thread_id}/runs/wait",
            headers={
                "Content-Type": "application/json"
            },
            json={
                "assistant_id": f"{self.assistant_id}",
                "command": {
                    "update": input_data,
                    "resume": input_data,
                }
            },
        )
        return response.json()
    
    def get_run_status(self, thread_id, run_id):
        """Get the status of a run"""
        response = requests.get(f"{self.base_url}/threads/{thread_id}/runs/{run_id}")
        response.raise_for_status()
        return response.json()
    
    def get_thread_state(self, thread_id):
        """Get the current state of a thread"""
        response = requests.get(f"{self.base_url}/threads/{thread_id}/state")
        response.raise_for_status()
        return response.json()
    
    def update_state(self, thread_id, values):
        """Update the thread state (for human feedback)"""
        payload = {"values": values}
        response = requests.post(
            f"{self.base_url}/threads/{thread_id}/state",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def resume_run(self, thread_id):
        """Resume a paused run"""
        response = requests.post(f"{self.base_url}/threads/{thread_id}/runs")
        response.raise_for_status()
        return response.json()

# Example usage
def main():
    client = LangGraphClient()
    

if __name__ == "__main__":
    main()