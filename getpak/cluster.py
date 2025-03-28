import psutil
from dask.distributed import LocalCluster, Client

class ClusterManager:
    """
    GET-Pak system-wide dask cluster instance and manager
    """
    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            # total memory available
            mem = int(0.75 * psutil.virtual_memory().total / (1024 * 1024 * 1024))
            # memory limit
            limit = 16 if mem > 16 else mem
            cluster = LocalCluster(n_workers=4, memory_limit=str(limit / 4) + 'GB', processes=True)
            cls._client = Client(cluster)
        return cls._client
    
get_client = ClusterManager.get_client
