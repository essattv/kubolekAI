bind = "0.0.0.0:10000"
workers = 1         # Mniej workerów = mniej RAM (duży wpływ)
threads = 2         # Też mniej wątków
timeout = 60        # Krótszy timeout zmniejsza szansę na zakleszczenia
max_requests = 100  # Restart workera po 100 żądaniach
max_requests_jitter = 10
