import redis

def init_db():
    print("Connecting to Redis...")
    r = redis.Redis(host='localhost', port=6379, db=0)
    
    print("Clearing leftover items in queues and mapping tables...")
    r.delete("warehouse:queue:reid")
    r.delete("warehouse:queue:matcher")
    r.delete("global_id_map")
    r.delete("state:gallery")
    r.delete("reid_worker:ready")
    
    print("Database Initialized successfully.")

if __name__ == "__main__":
    init_db()
