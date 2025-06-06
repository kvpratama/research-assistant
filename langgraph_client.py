import requests
import json

class LangGraphClient:
    def __init__(self, base_url="http://127.0.0.1:2024"):
        self.base_url = base_url
        self.thread_id = self.create_thread()
        self.assistant_id = None
    
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
    
    def create_assistant(self, graph_id):
        response = requests.post(f"{self.base_url}/assistants",
            headers={
            "Content-Type": "application/json"
            },
            json={
                "assistant_id": "",
                "graph_id": graph_id,
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
    
    def run_graph_stream(self, input_data):
        url = f"{self.base_url}/threads/{self.thread_id}/runs/stream"
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "assistant_id": self.assistant_id,
            "input": input_data,
            "stream_mode": ["updates"],
            "stream_subgraphs": True,
            "command": {
                "resume": input_data,
            }
        }

        with requests.post(url, headers=headers, json=payload, stream=True) as response:
            if response.status_code != 200:
                print(f"Stream request failed: {response.status_code} - {response.text}")
                yield f"Stream request failed: {response.status_code} - {response.text}"
                return

            for line in response.iter_lines(decode_unicode=True):
                if line.strip() == "":
                    continue  # skip empty lines

                print(f"🔹 Raw Line: {line}")
                if line.startswith("data:"):
                    data = json.loads(line[len("data:"):].strip())
                    if "generate_question" in data:
                        question = data["generate_question"]["messages"][0]["content"]
                        print(f"🔹 Question: {question}")
                        yield f"Questions:\n {question}\n\n --- \n\n"
                    elif "generate_answer" in data:
                        answer = data["generate_answer"]["messages"][0]["content"]
                        print(f"🔹 Aswer: {answer}")
                        yield f"Answers:\n {answer}\n\n --- \n\n"
                    elif "write_section" in data:
                        section = data["write_section"]["sections"][0]
                        print(f"Section: {section}")
                        yield f"Section:\n {section}\n\n --- \n\n"
                        