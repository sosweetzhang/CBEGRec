"""Disable proxy for all network libs. Call before importing requests/urllib3."""
import os
import sys

def disable_proxy():
    """Clear proxy-related env vars."""
    proxy_vars = [
        'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
        'ALL_PROXY', 'all_proxy', 'NO_PROXY', 'no_proxy',
        'REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE'
    ]
    for var in proxy_vars:
        os.environ.pop(var, None)
    os.environ['NO_PROXY'] = '*'
    os.environ['no_proxy'] = '*'

disable_proxy()
_patched = False

def patch_requests_no_proxy():
    """Patch requests/urllib3 to disable proxy."""
    global _patched
    if _patched:
        return
    try:
        import requests
        import urllib3
        if not hasattr(requests.Session.request, '_no_proxy_patched'):
            original_session_request = requests.Session.request
        def no_proxy_session_request(self, *args, **kwargs):
            kwargs['proxies'] = {}
            return original_session_request(self, *args, **kwargs)
            no_proxy_session_request._no_proxy_patched = True
            requests.Session.request = no_proxy_session_request
        if not hasattr(urllib3.connectionpool.HTTPConnectionPool.urlopen, '_no_proxy_patched'):
            original_connectionpool_urlopen = urllib3.connectionpool.HTTPConnectionPool.urlopen
            def no_proxy_urlopen(self, method, url, *args, **kwargs):
                if 'proxy' in kwargs:
                    del kwargs['proxy']
                if 'proxy_url' in kwargs:
                    del kwargs['proxy_url']
                return original_connectionpool_urlopen(self, method, url, *args, **kwargs)
            no_proxy_urlopen._no_proxy_patched = True
            urllib3.connectionpool.HTTPConnectionPool.urlopen = no_proxy_urlopen
        if not hasattr(urllib3.connectionpool.HTTPSConnectionPool.urlopen, '_no_proxy_patched'):
            original_https_urlopen = urllib3.connectionpool.HTTPSConnectionPool.urlopen
            def no_proxy_https_urlopen(self, method, url, *args, **kwargs):
                if 'proxy' in kwargs:
                    del kwargs['proxy']
                if 'proxy_url' in kwargs:
                    del kwargs['proxy_url']
                return original_https_urlopen(self, method, url, *args, **kwargs)
            no_proxy_https_urlopen._no_proxy_patched = True
            urllib3.connectionpool.HTTPSConnectionPool.urlopen = no_proxy_https_urlopen
        
        _patched = True
    except ImportError:
        pass

def ensure_no_proxy():
    """Disable proxy (call when requests/urllib3 are loaded)."""
    disable_proxy()
    patch_requests_no_proxy()

try:
    patch_requests_no_proxy()
except:
    pass
