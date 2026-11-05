
import asyncio
import websockets
import requests
import json
import time

API_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/chat"

def get_token(username, password="password123", is_admin=False):
    # Try to register first
    headers = {"X-Origin": "admin-ui"} if is_admin else {}
    requests.post(f"{API_URL}/users", json={"username": username, "password": password}, headers=headers)
    
    response = requests.post(f"{API_URL}/token", data={"username": username, "password": password})
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        print(f"Failed to get token for {username}: {response.text}")
        return None

async def repro():
    user_token = get_token("testuser_repro")
    admin_token = get_token("admin_repro", is_admin=True)
    
    if not user_token or not admin_token:
        print("Failed to get tokens")
        return

    async with websockets.connect(WS_URL) as user_ws:
        # 1. Authenticate user
        await user_ws.send(user_token)
        auth_resp = await user_ws.recv()
        print(f"User Auth: {auth_resp}")
        
        # 2. Send message
        await user_ws.send("Hello, I need help")
        
        # Get agent/system response
        agent_resp = await user_ws.recv()
        print(f"User received from agent: {agent_resp}")
        
        # 3. Admin looks up the session
        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        resp = requests.get(f"{API_URL}/admin/chats/requested", headers=admin_headers)
        chats = resp.json()
        print(f"Requested chats: {chats}")
        
        session_id = None
        for chat in chats:
            # Our user doesn't have a user_id easily accessible here but we just took the first one for simplicity or match
            session_id = chat["id"]
            break
        
        if not session_id:
            # If not in requested chats, maybe it didn't trigger human yet. 
            # But we can try to find all sessions if we had an endpoint.
            # Let's just use the fact that it should be there if agent triggered it.
            # If not, let's try to trigger human.
            print("No requested chats found, trying to trigger human...")
            await user_ws.send("I need a human")
            await user_ws.recv() # Wait for response
            resp = requests.get(f"{API_URL}/admin/chats/requested", headers=admin_headers)
            chats = resp.json()
            if chats:
                session_id = chats[0]["id"]
        
        if not session_id:
            print("Could not find session ID")
            return

        print(f"Admin responding to session: {session_id}")
        
        # 4. Admin responds via POST
        admin_msg = "Hello! I am the admin. How can I help you?"
        resp = requests.post(f"{API_URL}/admin/chat/{session_id}/respond", 
                             json={"message": admin_msg}, 
                             headers=admin_headers)
        print(f"Admin response status: {resp.status_code}, data: {resp.json()}")
        
        # 5. User should receive the admin message
        try:
            # The user might have received "Human assistant has been notified" first
            user_received = await asyncio.wait_for(user_ws.recv(), timeout=5)
            print(f"User received: {user_received}")
            
            # Check if it's the admin message
            try:
                data = json.loads(user_received)
                if data.get("sender") == "admin" and data.get("content") == admin_msg:
                    print("SUCCESS: User received the admin message correctly!")
                else:
                    print(f"Received something else: {data}")
            except:
                print(f"Received non-JSON: {user_received}")
                # Try to receive again if it was just a system message
                user_received = await asyncio.wait_for(user_ws.recv(), timeout=5)
                print(f"User received (2nd): {user_received}")
                data = json.loads(user_received)
                if data.get("sender") == "admin" and data.get("content") == admin_msg:
                    print("SUCCESS: User received the admin message correctly!")

        except asyncio.TimeoutError:
            print("FAILURE: User did not receive admin message within timeout")

if __name__ == "__main__":
    asyncio.run(repro())
