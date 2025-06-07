import requests
import json
import logging
import uuid
from graph import graph_memory

# Configure logger
logger = logging.getLogger(__name__)

class LangGraphLocalClient:
    def __init__(self):
        logger.info(f"Initializing LangGraphLocalClient")
        self.thread = self.create_thread()
        logger.debug(f"Client initialized with thread: {self.thread}")
    
    def create_thread(self):
        """Create a new thread"""
        return {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    def run_graph(self, input_data):
        """Run the graph with input data"""
        logger.info("Starting graph execution")
        logger.debug(f"Thread: {self.thread}")
        logger.debug(f"Input data: {json.dumps(input_data, indent=2)}")
        response = graph_memory.invoke(input_data, self.thread)
        return response
    
    def run_graph_resume(self, input_data):
        """Resume graph execution with updated input data"""
        logger.info("Resuming graph execution with updated input")
        logger.debug(f"Thread: {self.thread}")
        logger.debug(f"Resume data: {json.dumps(input_data, indent=2)}")
        
        graph_memory.update_state(self.thread, input_data)
        response = graph_memory.invoke(None, self.thread)
        return response

    def run_graph_stream(self, input_data):
        """Run graph and stream the results"""
        logger.info("Starting graph stream execution")
        logger.debug(f"Thread: {self.thread}")
        logger.debug(f"Input data: {json.dumps(input_data, indent=2)}")
        
        # graph_memory.update_state(self.thread, input_data, as_node="human_feedback")
        for event in graph_memory.stream(None, self.thread, subgraphs=True, stream_mode="updates"):
            _, data = event  # event[1] → data
            if data.get('generate_question', ''):
                print("Question: ", data.get('generate_question', '')["messages"][0].content)
                yield data.get('generate_question', '')["messages"][0].content + "\n\n --- \n\n"
            if data.get('generate_answer', ''):
                print("Answer: ", data.get('generate_answer', '')["messages"][0].content)
                yield data.get('generate_answer', '')["messages"][0].content + "\n\n --- \n\n"
            if data.get('write_section', ''):
                print("Section: ", data.get('write_section', '')["messages"][0].content)
                yield data.get('write_section', '')["messages"][0].content + "\n\n --- \n\n"

    def get_state(self):
        """Get the current state of the thread"""
        return graph_memory.get_state(self.thread)
        

class LangGraphClient:
    def __init__(self, base_url="http://127.0.0.1:2024"):
        self.base_url = base_url
        logger.info(f"Initializing LangGraphClient with base URL: {base_url}")
        self.thread_id = self.create_thread()
        self.assistant_id = None
        logger.debug(f"Client initialized with thread_id: {self.thread_id}")
    
    def create_thread(self):
        """Create a new thread"""
        logger.debug("Attempting to create a new thread")
        try:
            response = requests.post(f"{self.base_url}/threads", json={})
            logger.debug(f"Thread creation response status: {response.status_code}")
            
            if response.status_code == 200:
                thread_id = response.json()["thread_id"]
                logger.info(f"Successfully created thread with ID: {thread_id}")
            else:
                error_msg = f"Failed to create thread: {response.text}"
                logger.error(error_msg)
                response.raise_for_status()
                
            return thread_id
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed while creating thread: {str(e)}", exc_info=True)
            raise
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse thread creation response: {str(e)}", exc_info=True)
            raise
    
    def create_assistant(self, graph_id):
        """Create a new assistant with the given graph ID"""
        logger.info(f"Creating new assistant for graph_id: {graph_id}")
        try:
            response = requests.post(
                f"{self.base_url}/assistants",
                headers={"Content-Type": "application/json"},
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
            response.raise_for_status()
            
            assistant_data = response.json()
            assistant_id = assistant_data["assistant_id"]
            logger.info(f"Successfully created assistant with ID: {assistant_id}")
            return assistant_id
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to create assistant: {str(e)}", exc_info=True)
            raise
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse assistant creation response: {str(e)}", exc_info=True)
            raise
    
    def run_graph(self, input_data):
        """Run the graph with input data"""
        logger.info("Starting graph execution")
        logger.debug(f"Thread ID: {self.thread_id}, Assistant ID: {self.assistant_id}")
        logger.debug(f"Input data: {json.dumps(input_data, indent=2)}")
        
        try:
            response = requests.post(
                f"{self.base_url}/threads/{self.thread_id}/runs/wait",
                headers={"Content-Type": "application/json"},
                json={
                    "assistant_id": f"{self.assistant_id}",
                    "input": input_data,
                },
            )
            response.raise_for_status()
            
            logger.info("Graph execution completed successfully")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Graph execution failed: {str(e)}", exc_info=True)
            raise
        except json.JSONDecodeError as e:
            logger.error("Failed to parse graph execution response", exc_info=True)
            raise
    
    def run_graph_resume(self, input_data):
        """Resume graph execution with updated input data"""
        logger.info("Resuming graph execution with updated input")
        logger.debug(f"Thread ID: {self.thread_id}, Assistant ID: {self.assistant_id}")
        logger.debug(f"Resume data: {json.dumps(input_data, indent=2)}")
        
        try:
            response = requests.post(
                f"{self.base_url}/threads/{self.thread_id}/runs/wait",
                headers={"Content-Type": "application/json"},
                json={
                    "assistant_id": f"{self.assistant_id}",
                    "command": {
                        "update": input_data,
                        "resume": input_data,
                    }
                },
            )
            response.raise_for_status()
            
            logger.info("Graph resume completed successfully")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to resume graph: {str(e)}", exc_info=True)
            raise
        except json.JSONDecodeError as e:
            logger.error("Failed to parse graph resume response", exc_info=True)
            raise
    
    def run_graph_stream(self, input_data):
        """Run graph and stream the results"""
        url = f"{self.base_url}/threads/{self.thread_id}/runs/stream"
        logger.info("Starting graph stream execution")
        logger.debug(f"Stream URL: {url}")
        
        headers = {"Content-Type": "application/json"}
        payload = {
            "assistant_id": self.assistant_id,
            "input": input_data,
            "stream_mode": ["updates"],
            "stream_subgraphs": True,
            "command": {"resume": input_data}
        }

        try:
            with requests.post(url, headers=headers, json=payload, stream=True) as response:
                if response.status_code != 200:
                    error_msg = f"Stream request failed: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    yield error_msg
                    return

                logger.debug("Stream connection established, processing data...")
                for line in response.iter_lines(decode_unicode=True):
                    if not line.strip():
                        continue  # skip empty lines

                    logger.debug(f"Received stream data: {line[:200]}...")
                    if line.startswith("data:"):
                        try:
                            data = json.loads(line[len("data:"):].strip())
                            
                            if "generate_question" in data:
                                question = data["generate_question"]["messages"][0]["content"]
                                logger.info("Generated question in stream")
                                logger.debug(f"Question: {question}")
                                yield f"Questions:\n {question}\n\n --- \n\n"
                                
                            elif "generate_answer" in data:
                                answer = data["generate_answer"]["messages"][0]["content"]
                                logger.info("Generated answer in stream")
                                logger.debug(f"Answer: {answer[:200]}...")
                                yield f"Answers:\n {answer}\n\n --- \n\n"
                                
                            elif "write_section" in data:
                                section = data["write_section"]["sections"][0]
                                logger.info("Generated section in stream")
                                logger.debug(f"Section: {section[:200]}...")
                                yield f"Section:\n {section}\n\n --- \n\n"
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse stream data: {str(e)}")
                            continue
                            
        except requests.exceptions.RequestException as e:
            logger.error(f"Stream request failed: {str(e)}", exc_info=True)
            raise

    def get_state(self):
        """Get the current state of the thread"""
        url = f"{self.base_url}/threads/{self.thread_id}/state"
        logger.debug(f"Fetching state from URL: {url}")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            state_data = response.json()
            logger.debug("Successfully retrieved thread state")
            return state_data.get("values", {})
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get thread state: {str(e)}", exc_info=True)
            raise
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse state response: {str(e)}", exc_info=True)
            raise