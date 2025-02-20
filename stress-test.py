import requests
import threading
import time

# replace url with local or prod servers
url = 'https://python.jaitolentino.studio/measure-fingers'

completed_requests = 0
failed_requests = 0

def send_request():
    global completed_requests
    global failed_requests
    try: 
        response = requests.post(url, files={'image': open('images/palmdown-left.jpg', 'rb')}, data={'width': '20.5'}, verify=None)
        if response.status_code != 200: 
            failed_requests += 1
            return
        
        print(response.json())
        completed_requests += 1
            
    except Exception as e:
        print("An error happened ->", e)

threads = []

start_time = time.time()

for _ in range(30):
    t = threading.Thread(target=send_request)
    threads.append(t)
    t.start()
    
for t in threads:
    t.join()

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")
print(f"Completed Requests: {completed_requests}")
print(f"Failed Requests: {failed_requests}")