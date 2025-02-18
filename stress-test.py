import requests
import threading
import time

# replace url with local or prod servers
url = 'http://localhost:8080/measure-fingers'


def send_request():
    try: 
        response = requests.post(url, files={'image': open('images/palmdown-left.jpg', 'rb')}, data={'width': '20.5'})
        #if response.status_code == 200: 
        #    print(response.json())
    except Exception as e:
        print("An error happened ->", e)

threads = []

start_time = time.time()

for _ in range(3):
    t = threading.Thread(target=send_request)
    threads.append(t)
    t.start()
    
for t in threads:
    t.join()

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")